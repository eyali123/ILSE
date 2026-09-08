# ILSE: Intermediate Layer Structure Encoders

Implementation of [**Improving LLM Final Representations with Inter-Layer Geometry**](https://arxiv.org/abs/2603.22665) (Blyachman, Ulanovski, Bechler-Speicher, 2026).

Lightweight encoders (~300K-600K trainable params) that learn to aggregate **all layers** of a frozen LLM, instead of using only the last layer.

Works with any HuggingFace causal/masked LM. No fine-tuning of the base model.

## Install

```bash
pip install -e .                # core (SetEncoder works immediately)
pip install -e ".[gnn]"         # + torch-geometric for Cayley/FC encoders
```

PyG requires matching CUDA/PyTorch versions. See [PyG install guide](https://pytorch-geometric.readthedocs.io/en/latest/install/installation.html) if the `[gnn]` extra fails.

## Three Entry Points

Which one do you need?

| Use case | Entry point |
|----------|-------------|
| Plain text classification, want a working baseline fast | **`ILSEClassifier`** |
| Your own backbone / loss / training loop; one aggregated vector per input | **`build_aggregator`** |
| Per-token output *and* you want tokens to exchange information through the graph (e.g. per-residue protein tasks) | **`build_per_token_aggregator`** |

`num_layers` and `hidden_in` are the number of hidden layers **+ 1** (the embedding layer) and the backbone hidden size. Don't hardcode them — use `num_layers_for(model)` / `hidden_size_for(model)`.

### 1. Sklearn-style classifier (simple text classification)

Extracts frozen layer embeddings and trains the encoder + a linear head for you. Best when you just want text classification without writing a training loop.

```python
from ilse import ILSEClassifier, CayleyConfig

clf = ILSEClassifier("EleutherAI/pythia-410m", CayleyConfig(), num_classes=6)
clf.fit(train_texts, train_labels)
preds = clf.predict(test_texts)
acc = clf.score(test_texts, test_labels)
```

### 2. `build_aggregator` (custom fine-tuning pipelines)

Pure `nn.Module` mapping `(N, num_layers, hidden_in) -> (N, out_dim)`. You own the backbone, loss, optimizer, and training loop. Use it for any task where each input collapses to a single aggregated vector (classification, regression, ordinal, retrieval, per-residue tasks where you flatten `batch*seq_len` into `N`).

```python
import torch
from ilse import build_aggregator, CayleyConfig, num_layers_for, hidden_size_for

name = "EleutherAI/pythia-410m"
agg = build_aggregator(CayleyConfig(),
                       num_layers=num_layers_for(name),   # 25 for pythia-410m
                       hidden_in=hidden_size_for(name))   # 1024
# Forward: (N, num_layers, hidden_in) -> (N, agg.out_dim)

# Example: per-residue protein task with a frozen PLM (e.g. ESM-2), treating
# each residue as an independent sample (no cross-residue mixing here).
with torch.no_grad():
    out = esm_model(input_ids, attention_mask=mask)
hs = torch.stack(out.hidden_states, dim=1)      # (B, L, T, D)
hs = hs.permute(0, 2, 1, 3).reshape(B * T, L, D)  # flatten residues -> (B*T, L, D)
h = agg(hs)                                     # (B*T, out_dim)
h = h.view(B, T, -1)                            # (B, T, out_dim)
rates = rate_head(h).squeeze(-1)                # (B, T)
```

### 3. `build_per_token_aggregator` (multi-token Cayley graph)

Builds **one Cayley graph per sample spanning all tokens × all layers**, so GNN message passing mixes information across both tokens and layers before producing output. This is the version we used for **per-residue protein tasks with frozen PLMs**, where a residue's prediction should depend on its neighbours, not only its own layer stack.

> **Cost:** the graph has `seq_len × num_layers` nodes and the Cayley graph is sized to fit that (`|SL(2, Zₙ)| ~ n³`). Long sequences make very large graphs that are slow to build and memory-heavy. Keep `seq_len` modest or chunk long sequences; a warning is emitted past a few thousand nodes.

```python
import torch
from ilse import build_per_token_aggregator, CayleyConfig

# Per-token output (per-residue regression over a short window)
agg = build_per_token_aggregator(
    CayleyConfig(conv_type="gat", gat_heads=4, gnn_layers=2),
    num_layers=13, hidden_in=480,          # e.g. a small ESM-2
    output_mode="per_token",
)
x = torch.randn(2, 16, 13, 480)            # (batch, seq_len, num_layers, hidden_in)
y = agg(x)                                 # (2, 16, 256) -> (batch, seq_len, out_dim)

# Sequence-level output (short-text classification)
agg = build_per_token_aggregator(
    CayleyConfig(), num_layers=13, hidden_in=480,
    output_mode="sequence",
)
y = agg(torch.randn(4, 16, 13, 480))       # (4, 256) -> (batch, out_dim)
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
    agg = build_aggregator(cfg, num_layers=25, hidden_in=1024)
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
    pooling="mean",        # "mean" | "sum" (over real layer-nodes only;
                           #   Cayley virtual padding nodes are excluded)
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

**Cayley graph**: An algebraic construction (SL(2, Z_n)) that produces a sparse, regular expander graph. Adapts to any number of layers by finding the smallest Cayley graph >= num_layers and padding with virtual nodes. Virtual nodes participate in message passing but are excluded from the final pooling, so the aggregated vector reflects only the real LLM layers.

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

If you use this code, please cite:

```bibtex
@misc{2026improvingllmfinalrepresentations,
      title={Improving LLM Final Representations with Inter-Layer Geometry}, 
      author={Eyal Blyachman and Tom Ulanovski and Maya Bechler-Speicher},
      year={2026},
      eprint={2603.22665},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2603.22665}, 
}
```
