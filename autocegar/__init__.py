"""CEGAR gate machinery, extracted from the Auto-CEGAR (``autocegar``) repo.

Modules:
    scale_grad        - ``ScaleGrad`` autograd function (backward-only rescale).
    gate              - generalized ``gate(confidence, wrongness)`` interface.
    controllers       - valley-detection (tau) and tail-ratio (lambda) controllers.
    residual_signals  - residual-domain E_t / C_t signals for DeepAnT (placeholder formulas).
    rw_cegar          - ``CNN_RW_CEGAR`` = reproduced RW-1 + CEGAR gate.
"""
from .scale_grad import ScaleGrad
from .gate import gate
from .controllers import (
    update_ema,
    tail_ratio_lambda_controller,
    valley_quantile_controller,
)
from .residual_signals import (
    ResidualStats,
    compute_wrongness_E_t,
    compute_confidence_C_t,
)
from .rw_cegar import CNN_RW_CEGAR

__all__ = [
    "ScaleGrad",
    "gate",
    "update_ema",
    "tail_ratio_lambda_controller",
    "valley_quantile_controller",
    "ResidualStats",
    "compute_wrongness_E_t",
    "compute_confidence_C_t",
    "CNN_RW_CEGAR",
]
