"""RW-ML trainers for DeepAnT anomaly detection.

Implements Afonso Baldo's thesis (2025) methods faithfully, then adds
an AutoCEGAR-guided variant.

Classes
-------
RW1Trainer       -- correction tensor + L1 regularisation (epoch-wise)
RW2Trainer       -- RW-1 + Savitzky-Golay smoothing on correction
RWCegarTrainer   -- RW-2 + CEGAR gate scales the correction gradient

Anomaly score in all three methods: magnitude of the correction tensor,
NOT the prediction error (that is what makes the RW mechanism detect
anomalies: the algorithm corrects anomalous points more aggressively).
"""
from .rw1 import RW1Trainer
from .rw2 import RW2Trainer
from .rw_autocegar import RWCegarTrainer

__all__ = ["RW1Trainer", "RW2Trainer", "RWCegarTrainer"]
