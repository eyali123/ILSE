"""
ILSEClassifier: sklearn-style API for training ILSE encoders on top of frozen LLMs.
"""
from typing import List, Optional, Union
import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split

from .configs import CayleyConfig, FCConfig, SetEncoderConfig
from ._internal.llm import LLMLayerExtractor
from ._internal.nn_modules import GNNEncoder, SetEncoder, ClassificationHead
from ._internal.dataset import GraphDataset, TensorDataset, graph_collate, tensor_collate
from ._internal.training import train_model

# Type alias for any ILSE config
ILSEConfig = Union[CayleyConfig, FCConfig, SetEncoderConfig]


class ILSEClassifier:
    """
    Scikit-learn-style classifier that aggregates frozen LLM layer embeddings
    using GNN (Cayley / FC) or DeepSet encoders.

    Example:
        from ilse import ILSEClassifier, CayleyConfig

        clf = ILSEClassifier(
            base_model="EleutherAI/pythia-410m",
            config=CayleyConfig(),     # defaults from Optuna search
            num_classes=6,
        )
        clf.fit(train_texts, train_labels)
        preds = clf.predict(test_texts)
        acc = clf.score(test_texts, test_labels)
    """

    def __init__(
        self,
        base_model: str,
        config: ILSEConfig,
        num_classes: int,
        device: Optional[str] = None,
        torch_dtype=None,
        trust_remote_code: bool = False,
        max_length: int = 2048,
    ):
        """
        Args:
            base_model: HuggingFace model name or path (e.g. "EleutherAI/pythia-410m").
            config: One of CayleyConfig, FCConfig, or SetEncoderConfig.
            num_classes: Number of output classes.
            device: "cuda", "cpu", or None (auto-detect).
            torch_dtype: Optional dtype for LLM loading (e.g. torch.float16).
            trust_remote_code: Pass to HuggingFace model loading.
            max_length: Maximum token length for the tokenizer.
        """
        self.base_model = base_model
        self.config = config
        self.num_classes = num_classes
        self.torch_dtype = torch_dtype
        self.trust_remote_code = trust_remote_code
        self.max_length = max_length

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)

        # Built during fit()
        self._encoder = None
        self._head = None
        self._num_layers = None
        self._hidden_dim = None
        self._is_gnn = not isinstance(config, SetEncoderConfig)
        self._extractor = None

    def _get_extractor(self) -> LLMLayerExtractor:
        """Lazy-load the LLM extractor."""
        if self._extractor is None:
            self._extractor = LLMLayerExtractor(
                self.base_model,
                device=str(self.device),
                torch_dtype=self.torch_dtype,
                trust_remote_code=self.trust_remote_code,
                max_length=self.max_length,
            )
            self._num_layers = self._extractor.num_layers
            self._hidden_dim = self._extractor.hidden_dim
        return self._extractor

    def _build_encoder(self):
        """Construct the encoder and head from config."""
        cfg = self.config

        if isinstance(cfg, (CayleyConfig, FCConfig)):
            self._encoder = GNNEncoder(
                in_dim=self._hidden_dim,
                hidden_dim=cfg.hidden_dim,
                gnn_layers=cfg.gnn_layers,
                gin_mlp_layers=cfg.gin_mlp_layers if cfg.conv_type == "gin" else 0,
                conv_type=cfg.conv_type,
                pooling=cfg.pooling,
                dropout=cfg.dropout,
            )
            enc_out_dim = cfg.hidden_dim

        elif isinstance(cfg, SetEncoderConfig):
            self._encoder = SetEncoder(
                num_layers=self._num_layers,
                layer_dim=self._hidden_dim,
                hidden_dim=cfg.hidden_dim,
                pre_pooling_layers=cfg.pre_pooling_layers,
                post_pooling_layers=cfg.post_pooling_layers,
                pooling=cfg.pooling,
                dropout=cfg.dropout,
            )
            enc_out_dim = self._encoder.out_dim
        else:
            raise TypeError(f"config must be CayleyConfig, FCConfig, or SetEncoderConfig, got {type(cfg)}")

        self._head = ClassificationHead(enc_out_dim, self.num_classes)

    def _make_loader(self, embeddings, labels, shuffle=False):
        """Create appropriate DataLoader for the config type."""
        cfg = self.config
        if isinstance(cfg, CayleyConfig):
            topology = "cayley"
        elif isinstance(cfg, FCConfig):
            topology = "fully_connected"
        else:
            topology = None

        if self._is_gnn:
            ds = GraphDataset(embeddings, labels, topology=topology)
            return DataLoader(ds, batch_size=cfg.batch_size, shuffle=shuffle, collate_fn=graph_collate)
        else:
            ds = TensorDataset(embeddings, labels)
            return DataLoader(ds, batch_size=cfg.batch_size, shuffle=shuffle, collate_fn=tensor_collate)

    def fit(
        self,
        texts: List[str],
        labels: List[int],
        val_texts: Optional[List[str]] = None,
        val_labels: Optional[List[int]] = None,
        val_fraction: float = 0.15,
        extract_batch_size: int = 32,
        verbose: bool = True,
    ) -> "ILSEClassifier":
        """
        Extract layer embeddings from the LLM and train the ILSE encoder.

        Args:
            texts: Training texts.
            labels: Integer class labels.
            val_texts: Optional validation texts. If None, splits from train.
            val_labels: Optional validation labels.
            val_fraction: Fraction of train to use as validation (if val_texts is None).
            extract_batch_size: Batch size for LLM forward passes.
            verbose: Print training progress.

        Returns:
            self (for chaining).
        """
        extractor = self._get_extractor()

        # Extract embeddings
        if verbose:
            print(f"Extracting layer embeddings from {self.base_model} ...")
            print(f"  {self._num_layers} layers, {self._hidden_dim}-dim")
        all_texts = list(texts)
        all_labels = list(labels)

        if val_texts is not None:
            all_texts += list(val_texts)
            train_emb = extractor.extract(list(texts), batch_size=extract_batch_size, show_progress=verbose)
            val_emb = extractor.extract(list(val_texts), batch_size=extract_batch_size, show_progress=verbose)
            train_labels = list(labels)
            v_labels = list(val_labels)
        else:
            all_emb = extractor.extract(all_texts, batch_size=extract_batch_size, show_progress=verbose)
            # Split
            indices = list(range(len(all_emb)))
            train_idx, val_idx = train_test_split(
                indices, test_size=val_fraction, random_state=42, stratify=all_labels
            )
            train_emb = [all_emb[i] for i in train_idx]
            val_emb = [all_emb[i] for i in val_idx]
            train_labels = [all_labels[i] for i in train_idx]
            v_labels = [all_labels[i] for i in val_idx]

        # Build model
        self._build_encoder()

        # Create loaders
        train_loader = self._make_loader(train_emb, train_labels, shuffle=True)
        val_loader = self._make_loader(val_emb, v_labels, shuffle=False)

        cfg = self.config
        result = train_model(
            encoder=self._encoder,
            head=self._head,
            train_loader=train_loader,
            val_loader=val_loader,
            lr=cfg.lr,
            weight_decay=cfg.weight_decay,
            epochs=cfg.epochs,
            patience=10,
            device=self.device,
            is_gnn=self._is_gnn,
            verbose=verbose,
        )

        if verbose:
            print(f"Training complete: val_acc={result['best_val_acc']:.4f}")

        return self

    @torch.no_grad()
    def predict(self, texts: List[str], batch_size: int = 32) -> List[int]:
        """Predict class labels for texts."""
        proba = self.predict_proba(texts, batch_size=batch_size)
        return proba.argmax(axis=1).tolist()

    @torch.no_grad()
    def predict_proba(self, texts: List[str], batch_size: int = 32) -> np.ndarray:
        """
        Predict class probabilities for texts.

        Returns:
            np.ndarray of shape (len(texts), num_classes).
        """
        if self._encoder is None:
            raise RuntimeError("Model not trained. Call fit() first.")

        extractor = self._get_extractor()
        embeddings = extractor.extract(texts, batch_size=batch_size, show_progress=False)
        dummy_labels = [0] * len(embeddings)
        loader = self._make_loader(embeddings, dummy_labels, shuffle=False)

        self._encoder.to(self.device).eval()
        self._head.to(self.device).eval()

        all_probs = []
        for batch in loader:
            if self._is_gnn:
                batch = batch.to(self.device)
                h = self._encoder(batch)
            else:
                x, _ = batch
                x = x.to(self.device)
                h = self._encoder(x)
            logits = self._head(h)
            probs = torch.softmax(logits, dim=1)
            all_probs.append(probs.cpu().numpy())

        return np.concatenate(all_probs, axis=0)

    def score(self, texts: List[str], labels: List[int], batch_size: int = 32) -> float:
        """Return classification accuracy on texts."""
        preds = self.predict(texts, batch_size=batch_size)
        correct = sum(p == l for p, l in zip(preds, labels))
        return correct / len(labels)

    def save(self, path: str):
        """
        Save trained model to a file.

        Saves encoder weights, head weights, config, and model metadata.
        Does NOT save the LLM — it is reloaded from HuggingFace on load().
        """
        if self._encoder is None:
            raise RuntimeError("Nothing to save. Call fit() first.")

        state = {
            "encoder_state": self._encoder.state_dict(),
            "head_state": self._head.state_dict(),
            "config": self.config,
            "base_model": self.base_model,
            "num_classes": self.num_classes,
            "num_layers": self._num_layers,
            "hidden_dim": self._hidden_dim,
            "max_length": self.max_length,
        }
        torch.save(state, path)

    @classmethod
    def load(
        cls,
        path: str,
        device: Optional[str] = None,
        torch_dtype=None,
        trust_remote_code: bool = False,
    ) -> "ILSEClassifier":
        """
        Load a saved ILSEClassifier.

        The base LLM is re-downloaded/loaded from HuggingFace on first predict().
        """
        state = torch.load(path, map_location="cpu", weights_only=False)

        obj = cls(
            base_model=state["base_model"],
            config=state["config"],
            num_classes=state["num_classes"],
            device=device,
            torch_dtype=torch_dtype,
            trust_remote_code=trust_remote_code,
            max_length=state.get("max_length", 2048),
        )
        obj._num_layers = state["num_layers"]
        obj._hidden_dim = state["hidden_dim"]
        obj._build_encoder()
        obj._encoder.load_state_dict(state["encoder_state"])
        obj._head.load_state_dict(state["head_state"])

        return obj

    def unload_llm(self):
        """Free LLM GPU memory (encoder stays loaded)."""
        if self._extractor is not None:
            self._extractor.unload()
            self._extractor = None
