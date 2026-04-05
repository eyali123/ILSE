"""
ILSE: Intermediate Layer Structure Encoders

Lightweight encoders (~300K-600K params) that aggregate frozen LLM layer embeddings
for classification. No fine-tuning of the base model required.

Three encoder families:
- CayleyConfig: GNN with algebraic Cayley graph topology (SL(2, Z_n))
- FCConfig: GNN with fully-connected graph topology
- SetEncoderConfig: DeepSet (permutation-invariant, no message passing)

Quick start:
    from ilse import ILSEClassifier, CayleyConfig

    clf = ILSEClassifier("EleutherAI/pythia-410m", CayleyConfig(), num_classes=6)
    clf.fit(train_texts, train_labels)
    preds = clf.predict(test_texts)
"""

__version__ = "0.1.0"

from .configs import CayleyConfig, FCConfig, SetEncoderConfig
from .classifier import ILSEClassifier

__all__ = [
    "ILSEClassifier",
    "CayleyConfig",
    "FCConfig",
    "SetEncoderConfig",
]
