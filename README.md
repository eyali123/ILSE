# ILSE: Intermediate Layer Structure Encoders

Lightweight encoders (~300K-600K trainable params) that learn to aggregate **all layers** of a frozen LLM for classification, instead of using only the last layer.

Works with any HuggingFace causal/masked LM. No fine-tuning of the base model.

## Install

```bash
# PyTorch Geometric requires matching CUDA/PyTorch versions.
# Install PyG first: https://pytorch-geometric.readthedocs.io/en/latest/install/installation.html
pip install torch-geometric

pip install .          # from source
# pip install ilse     # (future) from PyPI
```

## Quick Start

```python
from ilse import ILSEClassifier, CayleyConfig

clf = ILSEClassifier(
    base_model="EleutherAI/pythia-410m",
    config=CayleyConfig(),    # sensible defaults from hyperparameter search
    num_classes=6,
)

clf.fit(train_texts, train_labels)
preds = clf.predict(test_texts)
acc = clf.score(test_texts, test_labels)

clf.save("model.pt")
clf = ILSEClassifier.load("model.pt")
```

## Three Encoder Families

| Encoder | Config class | Graph topology | Key idea |
|---------|-------------|---------------|----------|
| **Cayley** | `CayleyConfig` | SL(2, Z_n) algebraic graph | Sparse, regular GNN message passing over layers. Adapts to any model size via virtual nodes. |
| **FC** | `FCConfig` | Fully-connected | Dense GNN — every layer communicates with every other layer. |
| **SetEncoder** | `SetEncoderConfig` | None (DeepSet) | Permutation-invariant: per-layer MLP -> pool -> MLP. No message passing. |

Both Cayley and FC support two convolution types via `conv_type`:
- `"gin"` (default): GIN convolution with internal MLP — more expressive
- `"gcn"`: GCN convolution — simpler, fewer params

## Configuration

All hyperparameters are exposed with empirically-grounded defaults from an Optuna search across 3 LLMs and 6 classification tasks.

```python
from ilse import CayleyConfig, FCConfig, SetEncoderConfig

# Use defaults (recommended starting point)
config = CayleyConfig()

# Override specific params
config = CayleyConfig(
    conv_type="gcn",      # "gin" (default) or "gcn"
    hidden_dim=256,       # projection dimension (default: 256)
    gnn_layers=1,         # number of GNN layers (default: 1)
    gin_mlp_layers=1,     # MLP layers inside GINConv; ignored for gcn (default: 1)
    pooling="mean",       # "mean" (default), "sum", or "last"
    dropout=0.1,          # dropout rate (default: 0.1)
    lr=1e-3,              # learning rate (default: 1e-3; try 1e-4 for >3B models)
    weight_decay=1e-4,    # L2 regularization (default: 1e-4)
    batch_size=64,        # training batch size (default: 64)
    epochs=50,            # max training epochs (default: 50)
)

# FC encoder (same params as Cayley, different graph topology)
config = FCConfig(pooling="last")  # FC shows a mean/last split; try both

# SetEncoder (no graph, permutation-invariant)
config = SetEncoderConfig(
    pre_pooling_layers=1,   # MLP layers before pooling (default: 1)
    post_pooling_layers=1,  # MLP layers after pooling (default: 1)
    pooling="sum",          # "sum" (default, best for classification) or "mean"
    dropout=0.2,            # slightly higher than GNN variants (default: 0.2)
)
```

### Default Recommendations by Model Size

| Parameter | Models up to ~1B | Models >3B |
|-----------|-----------------|------------|
| `lr` | `1e-3` (default) | `1e-4` |
| `gnn_layers` | `1` | `1` |
| `hidden_dim` | `256` | `256` |

## API Reference

### `ILSEClassifier`

```python
ILSEClassifier(
    base_model: str,              # HuggingFace model name/path
    config: CayleyConfig | FCConfig | SetEncoderConfig,
    num_classes: int,
    device: str = None,           # "cuda"/"cpu"/None (auto)
    torch_dtype = None,           # e.g. torch.float16 for large models
    trust_remote_code: bool = False,
    max_length: int = 2048,       # tokenizer max length
)
```

**Methods:**

| Method | Description |
|--------|-------------|
| `fit(texts, labels, val_texts=None, val_labels=None, val_fraction=0.15)` | Extract embeddings and train. Returns `self`. |
| `predict(texts) -> List[int]` | Predict class labels. |
| `predict_proba(texts) -> np.ndarray` | Predict class probabilities, shape `(N, num_classes)`. |
| `score(texts, labels) -> float` | Classification accuracy. |
| `save(path)` | Save encoder weights + config (not the LLM). |
| `ILSEClassifier.load(path)` | Load a saved model. LLM re-loaded on first predict. |
| `unload_llm()` | Free LLM GPU memory. Encoder stays loaded. |

### Training Details

- **Optimizer**: Adam
- **Early stopping**: patience=10 epochs on validation accuracy
- **LR scheduler**: ReduceLROnPlateau (patience=3, factor=0.5)
- **Loss**: Cross-entropy
- **Validation**: Stratified split (15%) if no val set provided

## How It Works

Standard LLM usage takes only the **last layer** embedding. But intermediate layers contain different linguistic information (syntax, semantics, world knowledge). ILSE trains a small encoder to aggregate **all layers** into a single embedding.

```
Input text -> Frozen LLM -> [layer_0, layer_1, ..., layer_L]  (L+1 embeddings of dim D)
                                          |
                              ILSE Encoder (GNN or DeepSet)
                                          |
                              [aggregated embedding]  (dim 256)
                                          |
                                Linear -> class prediction
```

The encoder has ~300K-600K trainable parameters (vs billions in the LLM). The LLM stays frozen.

### Graph Topologies (Cayley and FC)

For GNN encoders, each text produces a **graph over layers**:
- **Nodes** = layer embeddings (one node per LLM layer)
- **Edges** = defined by topology (Cayley or fully-connected)

GNN message passing lets layers exchange information before pooling into a single vector.

**Cayley graph (SL(2, Z_n))**: An algebraic construction that produces a sparse, regular graph. Automatically adapts to any number of layers by finding the smallest Cayley graph that fits, padding with zero-initialized virtual nodes.

**FC graph**: Every layer connected to every other. Denser, more expensive for many layers, but lets all layers communicate directly.

## Tested Models

Validated on these LLMs (but works with any HuggingFace transformer):

| Model | Layers | Hidden dim | `base_model` string |
|-------|--------|-----------|---------------------|
| Pythia-410m | 25 | 1024 | `"EleutherAI/pythia-410m"` |
| TinyLlama-1.1B | 23 | 2048 | `"TinyLlama/TinyLlama-1.1B-Chat-v1.0"` |
| Llama3-8B | 33 | 4096 | `"meta-llama/Meta-Llama-3-8B"` |

## File Structure

```
ilse/
    __init__.py         # Public API: ILSEClassifier, CayleyConfig, FCConfig, SetEncoderConfig
    configs.py          # Dataclass configs with Optuna-grounded defaults
    classifier.py       # ILSEClassifier (fit/predict/save/load)
    _internal/
        llm.py          # LLMLayerExtractor (generic HF model wrapper)
        nn_modules.py   # GNNEncoder, SetEncoder, ClassificationHead (PyTorch modules)
        graph_ops.py    # Cayley graph construction (SL(2,Z_n)), FC edge index
        dataset.py      # GraphDataset (PyG), TensorDataset (for SetEncoder)
        training.py     # Training loop with early stopping
```

## Citation

Based on the ILSE paper (Intermediate Layer Structure Encoders). If you use this code, please cite:

```bibtex
@article{ilse2025,
    title={ILSE: Intermediate Layer Structure Encoders for LLM Classification},
    year={2025}
}
```
