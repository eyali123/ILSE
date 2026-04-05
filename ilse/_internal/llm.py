"""
Generic HuggingFace LLM wrapper for extracting layer-wise embeddings.

Works with any AutoModel that supports output_hidden_states=True
(GPT-2, Pythia, Llama, Mistral, Phi, etc.).
"""
from typing import List, Optional
import numpy as np
import torch
from tqdm import tqdm


class LLMLayerExtractor:
    """
    Loads a frozen HuggingFace model and extracts per-layer, per-sample embeddings.

    Usage:
        extractor = LLMLayerExtractor("EleutherAI/pythia-410m")
        embeddings = extractor.extract(["Hello world", "Another text"])
        # embeddings: list of np.ndarray, each shape (num_layers, hidden_dim)
    """

    def __init__(
        self,
        model_name_or_path: str,
        device: Optional[str] = None,
        torch_dtype=None,
        trust_remote_code: bool = False,
        max_length: int = 2048,
    ):
        from transformers import AutoModel, AutoTokenizer

        self.model_name = model_name_or_path
        self.max_length = max_length

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)

        load_kwargs = {"trust_remote_code": trust_remote_code}
        if torch_dtype is not None:
            load_kwargs["torch_dtype"] = torch_dtype

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name_or_path, trust_remote_code=trust_remote_code
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model = AutoModel.from_pretrained(
            model_name_or_path, output_hidden_states=True, **load_kwargs
        )
        self.model.to(self.device)
        self.model.eval()

        # Infer model properties
        self.num_layers = self.model.config.num_hidden_layers + 1  # +1 for embedding layer
        self.hidden_dim = self.model.config.hidden_size

    @torch.no_grad()
    def extract(
        self,
        texts: List[str],
        batch_size: int = 32,
        show_progress: bool = True,
    ) -> List[np.ndarray]:
        """
        Extract layer-wise embeddings for each text (mean-pooled over tokens).

        Args:
            texts: Input texts.
            batch_size: Batch size for forward passes.
            show_progress: Show tqdm progress bar.

        Returns:
            List of np.ndarray, each shape (num_layers, hidden_dim). One per text.
        """
        all_embeddings = []
        batches = range(0, len(texts), batch_size)
        if show_progress:
            batches = tqdm(batches, desc="Extracting embeddings", unit="batch")

        for start in batches:
            batch_texts = texts[start : start + batch_size]
            encoded = self.tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt",
            ).to(self.device)

            outputs = self.model(**encoded)
            hidden_states = outputs.hidden_states  # tuple of (B, seq_len, D)

            # Build attention mask for mean pooling (ignore padding tokens)
            mask = encoded["attention_mask"].unsqueeze(-1).float()  # (B, seq_len, 1)

            # Stack all layers: (num_layers, B, seq_len, D)
            stacked = torch.stack(hidden_states, dim=0)

            # Mean pool over non-padding tokens: (num_layers, B, D)
            masked = stacked * mask.unsqueeze(0)
            pooled = masked.sum(dim=2) / mask.unsqueeze(0).sum(dim=2).clamp(min=1e-9)

            # Convert to per-sample arrays: each (num_layers, D)
            pooled_np = pooled.cpu().float().numpy()  # (num_layers, B, D)
            for i in range(pooled_np.shape[1]):
                all_embeddings.append(pooled_np[:, i, :])  # (num_layers, D)

        return all_embeddings

    def unload(self):
        """Free GPU memory by deleting the model."""
        del self.model
        del self.tokenizer
        if self.device.type == "cuda":
            torch.cuda.empty_cache()
