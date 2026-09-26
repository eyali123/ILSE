# ILGE: Inter-Layer Geometry Encoders

[![PyPI](https://img.shields.io/pypi/v/ilge)](https://pypi.org/project/ilge/)
[![NeurIPS 2026](https://img.shields.io/badge/NeurIPS-2026-blue)](https://arxiv.org/abs/2603.22665)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

An easy-to-use package for the classification methods from [**Improving LLM Final Representations with Inter-Layer Geometry**](https://arxiv.org/abs/2603.22665) (Blyachman, Ulanovski, Bechler-Speicher), **NeurIPS 2026**.

Lightweight encoders (~300K-600K trainable params) that learn to aggregate **all layers** of a frozen LLM, instead of using only the last layer. No fine-tuning of the base model.

Works with transformer-based models that expose per-layer hidden states via `output_hidden_states=True`. It has been validated on the [models below](#tested-models).

## How It Works

![ILGE overview](https://raw.githubusercontent.com/eyali123/ILGE/main/assets/ilge_fig.png)

Standard LLM usage takes only the **last layer** representation, but intermediate layers carry complementary information. ILGE runs the input through a **frozen** LLM, takes the representation from every layer, and learns to combine them with a small encoder:

1. Each layer's representation becomes a **node** in a graph.
2. The nodes are connected by a **Cayley graph** over the group SL(2, Z_n): a sparse, regular expander graph. The smallest such graph with at least `num_layers` nodes is used; extra nodes (red in the figure) are zero-initialized **virtual nodes**.
3. A **GNN** (GIN, GCN or GAT) passes messages over the graph, then pools the **real** layer nodes into the final representation. Virtual nodes help carry messages but are excluded from the pooling.
4. A task head (e.g. a linear classifier) sits on top. Only the encoder and head are trained, ~300K-600K parameters.

Every node in the Cayley graph has the same small degree (4 neighbours), so the number of edges grows **linearly** with the number of nodes, whereas a fully-connected graph (`FCConfig`) grows quadratically.

## Install

```bash
pip install ilge
```

This installs everything, including [PyTorch Geometric](https://pytorch-geometric.readthedocs.io/) for the Cayley and FC graph encoders. Install PyTorch for your CUDA version first if you need GPU support; see the [PyG install guide](https://pytorch-geometric.readthedocs.io/en/latest/install/installation.html) if the `torch-geometric` install has trouble.

From source (for development):

```bash
git clone https://github.com/eyali123/ILGE.git && cd ILGE
pip install -e ".[dev]"
pytest                          # run the test suite
```

## Three Entry Points

Which one do you need?

| Use case | Entry point |
|----------|-------------|
| Plain text classification, want a working baseline fast | **`ILGEClassifier`** |
| Your own backbone / loss / training loop; one aggregated vector per input | **`build_aggregator`** |
| Aggregate information from **all tokens across all layers** in one graph (per-token or sequence-level output) | **`build_per_token_aggregator`** |

`num_layers` and `hidden_in` are the number of hidden layers **+ 1** (the embedding layer) and the backbone hidden size. Don't hardcode them — use `num_layers_for(model)` / `hidden_size_for(model)`.

### 1. Sklearn-style classifier (simple text classification)

Extracts frozen layer embeddings and trains the encoder + a linear head for you. Best when you just want text classification without writing a training loop.

```python
from ilge import ILGEClassifier, CayleyConfig

clf = ILGEClassifier("Qwen/Qwen3-8B-Base", CayleyConfig(lr=1e-4), num_classes=6)  # lr=1e-4 for >3B models
clf.fit(train_texts, train_labels)
preds = clf.predict(test_texts)
acc = clf.score(test_texts, test_labels)
```

Layer embeddings are mean-pooled over tokens by default; pass
`token_pooling="last" | "first" | "mean_including_padding"` to the constructor
to change that.

### 2. `build_aggregator` (custom fine-tuning pipelines)

Pure `nn.Module` mapping `(N, num_layers, hidden_in) -> (N, out_dim)`. You own the backbone, loss, optimizer, and training loop. Use it for any task where each input collapses to a single aggregated vector (classification, regression, ordinal, retrieval). Reach for it over `ILGEClassifier` when you need a custom head, loss, or training schedule.

```python
import torch
import torch.nn as nn
from ilge import build_aggregator, CayleyConfig, num_layers_for, hidden_size_for

name, num_classes = "Qwen/Qwen3-8B-Base", 6
agg = build_aggregator(CayleyConfig(),
                       num_layers=num_layers_for(name),   # 37 for Qwen3-8B
                       hidden_in=hidden_size_for(name))   # 4096
head = nn.Linear(agg.out_dim, num_classes)               # your own task head

# X: per-sample layer embeddings, shape (N, num_layers, hidden_in). y: (N,) labels.
# Produce X from the frozen LLM's hidden states, mean-pooled over tokens, e.g.:
#   out = llm(**tokenizer(texts, ...), output_hidden_states=True)
#   X = torch.stack(out.hidden_states, dim=1).mean(dim=2)   # (N, num_layers, hidden_in)

opt = torch.optim.Adam(list(agg.parameters()) + list(head.parameters()), lr=1e-4)  # 1e-4 for >3B models
loss_fn = nn.CrossEntropyLoss()
for epoch in range(50):                          # your training loop
    logits = head(agg(X))                        # (N, num_classes)
    loss = loss_fn(logits, y)
    opt.zero_grad(); loss.backward(); opt.step()
```

### 3. `build_per_token_aggregator` (multi-token Cayley graph)

Instead of first pooling each layer over tokens, this builds **one Cayley graph per sample whose nodes are every (token, layer) pair**. GNN message passing then aggregates information from all tokens across all layers at once. You can read the result out per token (`output_mode="per_token"`) or as a single vector per sequence (`output_mode="sequence"`).

*Side note:* a cool use case for per-token output is residue-level tasks on protein language model (PLM) embeddings, where each residue's prediction can draw on its neighbours as well as its own layer stack.

> **Graph size:** here the graph has one node per (token, layer) pair, i.e. `seq_len × num_layers` nodes, before padding. Edges still grow linearly with the number of nodes, but long sequences make big graphs. Keep `seq_len` modest or chunk long sequences; a warning is printed above 2,000 nodes.

```python
import torch
from ilge import build_per_token_aggregator, CayleyConfig, num_layers_for, hidden_size_for

name = "Qwen/Qwen3-8B-Base"
L, D = num_layers_for(name), hidden_size_for(name)   # 37, 4096

# x: all hidden states of a short sequence, (batch, seq_len, num_layers, hidden_in)
x = torch.randn(4, 16, L, D)

# Sequence-level output: one vector per sequence (e.g. short-text classification)
agg = build_per_token_aggregator(
    CayleyConfig(conv_type="gcn"), num_layers=L, hidden_in=D,
    output_mode="sequence",
)
y = agg(x)                                 # (4, 256) -> (batch, out_dim)

# Per-token output: one vector per token (e.g. token- or residue-level tasks)
agg = build_per_token_aggregator(
    CayleyConfig(conv_type="gcn", gnn_layers=2), num_layers=L, hidden_in=D,
    output_mode="per_token",
)
y = agg(x)                                 # (4, 16, 256) -> (batch, seq_len, out_dim)
```

## Encoder Families

| Encoder | Config class | Layer interaction |
|---------|-------------|-------------------|
| **Cayley** (GNN) | `CayleyConfig` | Message passing over an SL(2, Z_n) Cayley expander graph |
| **FC** (GNN) | `FCConfig` | Message passing over a fully-connected graph |
| **DeepSet** | `DeepSetConfig` | None — permutation-invariant per-layer MLP, pool, MLP |

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
from ilge.tuning import recommended_config

cfg = recommended_config("cayley")                            # lr=1e-3 (small models)
cfg = recommended_config("cayley", model_size_hint="large")   # lr=1e-4 (>3B models)
cfg = recommended_config("deepset")                           # DeepSet defaults
```

### `suggest_config` — Optuna search space

Call inside an Optuna objective to search over the paper's validated hyperparameter space.

```python
import optuna
from ilge.tuning import suggest_config
from ilge import build_aggregator, num_layers_for, hidden_size_for

name = "Qwen/Qwen3-8B-Base"

def objective(trial):
    cfg = suggest_config(trial, "cayley")   # samples conv_type, hidden_dim, gnn_layers, etc.
    agg = build_aggregator(cfg, num_layers=num_layers_for(name),
                           hidden_in=hidden_size_for(name))
    model = MyModel(agg)
    return train_and_eval(model)

study = optuna.create_study(direction="minimize")
study.optimize(objective, n_trials=100)
best_cfg = suggest_config(study.best_trial, "cayley")
```

SQLite storage is sufficient for single-machine search (no PostgreSQL needed).

## Configuration Reference

```python
from ilge import CayleyConfig

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

`FCConfig` has the same parameters. `DeepSetConfig` has `pre_pooling_layers`, `post_pooling_layers` instead of GNN-specific params.

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

### ILGEClassifier

| Method | Description |
|--------|-------------|
| `fit(texts, labels, val_texts=None, val_labels=None)` | Extract embeddings and train |
| `predict(texts) -> List[int]` | Predict class labels |
| `predict_proba(texts) -> np.ndarray` | Class probabilities `(N, num_classes)` |
| `score(texts, labels) -> float` | Classification accuracy |
| `save(path)` / `load(path)` | Persist encoder weights + config (not the LLM) |
| `unload_llm()` | Free LLM GPU memory |

## Tested Models

Validated on the following transformer models. The `Layers` column is `num_hidden_layers + 1` (the count ILGE aggregates, including the embedding layer) — i.e. `num_layers_for(model)`.

| Model | Layers | Hidden dim | `base_model` string |
|-------|--------|-----------|---------------------|
| Pythia-410m\* | 25 | 1024 | `"EleutherAI/pythia-410m"` |
| TinyLlama-1.1B | 23 | 2048 | `"TinyLlama/TinyLlama-1.1B-Chat-v1.0"` |
| Gemma2-2B | 27 | 2304 | `"google/gemma-2-2b"` |
| Llama3-8B | 33 | 4096 | `"meta-llama/Meta-Llama-3-8B"` |
| Qwen3-8B\* | 37 | 4096 | `"Qwen/Qwen3-8B-Base"` |

\* Also validated on other sizes from the same family: Pythia-14m to Pythia-2.8B, and Qwen3-1.7B to Qwen3-14B.

## File Structure

```
ilge/
    __init__.py          # Public API exports
    configs.py           # CayleyConfig, FCConfig, DeepSetConfig
    aggregator.py        # Aggregator, PerTokenAggregator, build_* factories
    classifier.py        # ILGEClassifier (sklearn-style)
    tuning.py            # recommended_config, suggest_config
    model_utils.py       # num_layers_for, hidden_size_for
    _internal/
        nn_modules.py    # GNNEncoder (GIN/GCN/GAT), DeepSetEncoder, build_gnn_encoder
        graph_ops.py     # Cayley graph construction, FC edge index
        llm.py           # LLMLayerExtractor (HF model wrapper, token pooling)
        dataset.py       # GraphDataset, TensorDataset
        training.py      # Training loop with early stopping
tests/                   # pytest suite (excluded from the built package)
assets/                  # README figure
pyproject.toml           # Package metadata and dependencies
LICENSE                  # MIT
```

## Citation

If you use this code, please cite:

```bibtex
@misc{ulanovski2026improvingllmfinalrepresentations,
      title={Improving LLM Final Representations with Inter-Layer Geometry},
      author={Tom Ulanovski and Eyal Blyachman and Maya Bechler-Speicher},
      year={2026},
      eprint={2603.22665},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2603.22665},
}
```
