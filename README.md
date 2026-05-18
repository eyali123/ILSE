# ILSE: Intermediate Layer Structure Encoders

Lightweight encoders (~300K-600K trainable params) that learn to aggregate **all layers** of a frozen LLM, instead of using only the last layer.

Works with any HuggingFace causal/masked LM. No fine-tuning of the base model.

## Install

```bash
pip install -e .                # core (SetEncoder works immediately)
pip install -e ".[gnn]"         # + torch-geometric for Cayley/FC encoders
```

PyG requires matching CUDA/PyTorch versions. See [PyG install guide](https://pytorch-geometric.readthedocs.io/en/latest/install/installation.html) if the `[gnn]` extra fails.

## Three Entry Points

### 1. Sklearn-style classifier (simple text classification)

```python
from ilse import ILSEClassifier, CayleyConfig

clf = ILSEClassifier("EleutherAI/pythia-410m", CayleyConfig(), num_classes=6)
clf.fit(train_texts, train_labels)
preds = clf.predict(test_texts)
```

### 2. `build_aggregator` (custom fine-tuning pipelines)

Pure `nn.Module`. You own the backbone, loss, optimizer, and training loop.

```python
from ilse import build_aggregator, CayleyConfig

agg = build_aggregator(CayleyConfig(), num_layers=37, hidden_in=2560)
# Forward: (N, num_layers, hidden_in) -> (N, agg.out_dim)

# Example: per-residue protein task with frozen ESM-2
with torch.no_grad():
    out = esm_model(input_ids, attention_mask=mask)
hs = torch.stack(out.hidden_states, dim=1)    # (B, L, T, D)
hs = hs.permute(0, 2, 1, 3).reshape(B*T, L, D)  # flatten residues
h = agg(hs)                                    # (B*T, out_dim)
h = h.view(B, T, -1)                          # (B, T, out_dim)
rates = rate_head(h).squeeze(-1)               # (B, T)
```

### 3. `build_per_token_aggregator` (multi-token Cayley graph)

Builds one Cayley graph per sample spanning **all tokens x all layers**. GNN message passing mixes information across both tokens and layers before producing output.

```python
from ilse import build_per_token_aggregator, CayleyConfig

# Per-token output (e.g., per-residue regression)
agg = build_per_token_aggregator(
    CayleyConfig(conv_type="gat", gat_heads=4, gnn_layers=2),
    num_layers=37, hidden_in=2560,
    output_mode="per_token",
)
# Forward: (batch, seq_len, num_layers, hidden_in) -> (batch, seq_len, out_dim)

# Sequence-level output (e.g., text classification)
agg = build_per_token_aggregator(
    CayleyConfig(), num_layers=25, hidden_in=1024,
    output_mode="sequence",
)
# Forward: (batch, seq_len, num_layers, hidden_in) -> (batch, out_dim)
```

## Encoder Families

| Encoder | Config class | Graph topology | PyG required? |
|---------|-------------|----------------|---------------|
| **Cayley** | `CayleyConfig` | SL(2, Z_n) algebraic graph | Yes |
| **FC** | `FCConfig` | Fully-connected | Yes |
| **SetEncoder** | `SetEncoderConfig` | None (DeepSet) | No |

### Convolution types (Cayley and FC)

All three conv types are supported via `conv_type` on `CayleyConfig` and `FCConfig`:

| Conv type | Parameter | Description |
|-----------|-----------|-------------|
| `"gin"` (default) | `gin_mlp_layers` | GIN convolution with internal MLP — most expressive |
| `"gcn"` | — | GCN convolution — simpler, fewer params |
| `"gat"` | `gat_heads` | GAT with multi-head attention — data-dependent mixing |

```python
CayleyConfig(conv_type="gat", gat_heads=4)  # GAT with 4 attention heads
CayleyConfig(conv_type="gin", gin_mlp_layers=2)  # GIN with 2-layer MLP
CayleyConfig(conv_type="gcn")  # GCN
```

## Hyperparameter Helpers

### `recommended_config` — zero-search defaults

Empirically-validated defaults from an Optuna search across 3 LLMs and 6 tasks.

```python
from ilse.tuning import recommended_config

cfg = recommended_config("cayley")                            # lr=1e-3 (small models)
cfg = recommended_config("cayley", model_size_hint="large")   # lr=1e-4 (>3B models)
cfg = recommended_config("set_encoder")                       # DeepSet defaults
```

### `suggest_config` — Optuna search space

Call inside an Optuna objective to search over the paper's validated hyperparameter space.

```python
import optuna
from ilse.tuning import suggest_config
from ilse import build_aggregator

def objective(trial):
    cfg = suggest_config(trial, "cayley")   # samples conv_type, hidden_dim, gnn_layers, etc.
    agg = build_aggregator(cfg, num_layers=37, hidden_in=2560)
    model = MyModel(agg)
    return train_and_eval(model)

study = optuna.create_study(direction="minimize")
study.optimize(objective, n_trials=100)
best_cfg = suggest_config(study.best_trial, "cayley")
```

SQLite storage is sufficient for single-machine search (no PostgreSQL needed).

## Configuration Reference

```python
from ilse import CayleyConfig

CayleyConfig(
    conv_type="gin",       # "gin" | "gcn" | "gat"
    hidden_dim=256,        # projection dimension
    gnn_layers=1,          # number of GNN layers
    gin_mlp_layers=1,      # MLP layers inside GINConv (gin only)
    gat_heads=4,           # attention heads (gat only)
    pooling="mean",        # "mean" | "sum" | "last"
    dropout=0.1,
    lr=1e-3,               # try 1e-4 for >3B models
    weight_decay=1e-4,     # GAT empirically prefers 1e-3
    batch_size=64,
    epochs=50,
)
```

`FCConfig` has the same parameters. `SetEncoderConfig` has `pre_pooling_layers`, `post_pooling_layers` instead of GNN-specific params.

### Defaults by model size

| Parameter | Models up to ~1B | Models >3B |
|-----------|-----------------|------------|
| `lr` | `1e-3` (default) | `1e-4` |
| `gnn_layers` | `1` | `1` |
| `hidden_dim` | `256` | `256` |

## API Summary

### Aggregators

| Factory | Input | Output | Use case |
|---------|-------|--------|----------|
| `build_aggregator(config, L, D)` | `(N, L, D)` | `(N, out_dim)` | Per-sample layer aggregation |
| `build_per_token_aggregator(..., output_mode="per_token")` | `(B, T, L, D)` | `(B, T, out_dim)` | Per-token with cross-token Cayley mixing |
| `build_per_token_aggregator(..., output_mode="sequence")` | `(B, T, L, D)` | `(B, out_dim)` | Sequence-level with multi-token Cayley |

All aggregators expose `.out_dim` for sizing downstream heads.

### ILSEClassifier

| Method | Description |
|--------|-------------|
| `fit(texts, labels, val_texts=None, val_labels=None)` | Extract embeddings and train |
| `predict(texts) -> List[int]` | Predict class labels |
| `predict_proba(texts) -> np.ndarray` | Class probabilities `(N, num_classes)` |
| `score(texts, labels) -> float` | Classification accuracy |
| `save(path)` / `load(path)` | Persist encoder weights + config (not the LLM) |
| `unload_llm()` | Free LLM GPU memory |

## How It Works

Standard LLM usage takes only the **last layer** embedding. But intermediate layers contain different information (syntax, semantics, world knowledge). ILSE trains a small encoder to aggregate **all layers**.

```
Input -> Frozen LLM -> [layer_0, layer_1, ..., layer_L]   (L+1 vectors of dim D)
                                      |
                          ILSE Encoder (GNN or DeepSet)
                                      |
                          [aggregated embedding]           (dim 256)
                                      |
                            Task head -> prediction
```

**Cayley graph**: An algebraic construction (SL(2, Z_n)) that produces a sparse, regular expander graph. Adapts to any number of layers by finding the smallest Cayley graph >= num_layers and padding with virtual nodes.

**Vectorized batching**: The GNN path constructs PyG-compatible batches via pure tensor ops (tiled edge_index + offsets). No Python loop, no per-sample `Data` objects. Handles N=4000+ samples per step efficiently.

## Tested Models

Works with any HuggingFace transformer. Validated on:

| Model | Layers | Hidden dim | `base_model` string |
|-------|--------|-----------|---------------------|
| Pythia-410m | 25 | 1024 | `"EleutherAI/pythia-410m"` |
| TinyLlama-1.1B | 23 | 2048 | `"TinyLlama/TinyLlama-1.1B-Chat-v1.0"` |
| Llama3-8B | 33 | 4096 | `"meta-llama/Meta-Llama-3-8B"` |
| Gemma2-2B | 27 | 2304 | `"google/gemma-2-2b"` |

## File Structure

```
ilse/
    __init__.py          # Public API exports
    configs.py           # CayleyConfig, FCConfig, SetEncoderConfig
    aggregator.py        # Aggregator, PerTokenAggregator, build_* factories
    classifier.py        # ILSEClassifier (sklearn-style)
    tuning.py            # recommended_config, suggest_config
    _internal/
        nn_modules.py    # GNNEncoder (GIN/GCN/GAT), SetEncoder
        graph_ops.py     # Cayley graph construction, FC edge index
        llm.py           # LLMLayerExtractor (HF model wrapper)
        dataset.py       # GraphDataset, TensorDataset
        training.py      # Training loop with early stopping
```

## Citation

```bibtex
@article{ilse2025,
    title={ILSE: Intermediate Layer Structure Encoders for LLM Classification},
    year={2025}
}
```
