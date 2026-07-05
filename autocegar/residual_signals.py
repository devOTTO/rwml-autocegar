"""Residual-domain signals for the residual-based CEGAR adaptation.

This module maps DeepAnT's forecast residual ``r = y - y_hat`` to the two factors
the generalized gate consumes:

* ``E_t`` (wrongness, replaces ``1 - p_y``): how far this prediction is from truth.
* ``C_t`` (confidence, replaces ``sigmoid(k(conf - tau))``): how confident the
  model is, derived from its *historical* residual level (an EMA).

It also maintains the streaming residual statistics the formulas need:
``residual_q95`` and ``residual_ema``, aggregated per-feature (DeepAnT is
multivariate) and globally.

============================ PLACEHOLDER NOTICE ============================
The exact E_t / C_t formulas, the normalization, and the constants ``tau`` and
``k`` are PLACEHOLDERS pending Luis's confidence/wrongness notebook. The two
``compute_*`` functions below and the worked-example test in
``tests/test_gate.py`` are the spots to update once that notebook arrives.
Everything else (streaming stats, gate plumbing, controllers) is final.
===========================================================================
"""
from collections import deque
from typing import Deque, Optional

import torch


def compute_wrongness_E_t(window_residual: torch.Tensor, residual_q95: float, max_value: float = 1.0) -> torch.Tensor:
    """E_t = clamp(residual / residual_q95, max=1.0).  [PLACEHOLDER formula]

    Args:
        window_residual: per-window residual magnitude, shape ``[B]``.
        residual_q95: 95th-percentile residual used for normalization.
        max_value: clamp ceiling (default 1.0).

    Returns:
        Wrongness ``E_t`` in ``[0, max_value]``, shape ``[B]``.
    """
    q95 = max(float(residual_q95), 1e-8)
    return (window_residual / q95).clamp(min=0.0, max=max_value)


def compute_confidence_C_t(residual_ema: float, tau: float, k: float) -> float:
    """C_t = sigmoid(k * (tau - residual_ema)).  [PLACEHOLDER formula]

    Note the sign flip relative to classification: a *low* historical residual
    means the model has been accurate, hence *high* confidence.

    Args:
        residual_ema: EMA of the model's residual level (historical accuracy).
        tau: residual threshold.
        k: sigmoid sharpness.

    Returns:
        Scalar confidence ``C_t`` in ``[0, 1]``.
    """
    return float(torch.sigmoid(torch.tensor(k * (tau - residual_ema))).item())


class ResidualStats:
    """Streaming residual statistics: per-feature + global ``q95`` and ``ema``.

    The q95 is estimated from a bounded rolling buffer of recent residuals (exact
    streaming quantiles would need P^2 / t-digest; a buffer is sufficient and
    transparent for this scaffolding). The EMA tracks the model's current residual
    level and is the input to ``C_t``.
    """

    def __init__(self, n_channels: int, ema_beta: float = 0.9, buffer_size: int = 5000):
        self.n_channels = int(n_channels)
        self.ema_beta = float(ema_beta)
        self.buffer_size = int(buffer_size)

        self._global_buffer: Deque[float] = deque(maxlen=self.buffer_size)
        self._channel_buffers = [deque(maxlen=self.buffer_size) for _ in range(self.n_channels)]

        self.global_ema: Optional[float] = None
        self.channel_ema = [None for _ in range(self.n_channels)]

    @staticmethod
    def window_residual(residual: torch.Tensor) -> torch.Tensor:
        """Per-window scalar residual magnitude (mean |r| over channels x horizon).

        Uses mean-absolute (L1) to stay consistent with DeepAnT's training loss.

        Args:
            residual: ``y - y_hat`` of shape ``[B, C, H]``.

        Returns:
            ``[B]`` per-window residual magnitude.
        """
        return residual.abs().mean(dim=[1, 2])

    def update(self, residual: torch.Tensor) -> torch.Tensor:
        """Update streaming stats from a batch of residuals.

        Args:
            residual: ``y - y_hat`` of shape ``[B, C, H]``.

        Returns:
            The per-window residual magnitude ``[B]`` (also used downstream for E_t).
        """
        residual = residual.detach()
        w = self.window_residual(residual)  # [B]

        # global buffer + EMA
        self._global_buffer.extend(w.flatten().cpu().tolist())
        batch_mean = float(w.mean().item())
        self.global_ema = self._ema(self.global_ema, batch_mean)

        # per-channel buffers + EMA  (mean |r| over batch x horizon -> [C])
        per_channel = residual.abs().mean(dim=[0, 2])  # [C]
        for c in range(min(self.n_channels, per_channel.numel())):
            val = float(per_channel[c].item())
            self._channel_buffers[c].append(val)
            self.channel_ema[c] = self._ema(self.channel_ema[c], val)

        return w

    def _ema(self, prev: Optional[float], value: float) -> float:
        if prev is None or self.ema_beta <= 0.0:
            return float(value)
        return float(self.ema_beta * prev + (1.0 - self.ema_beta) * value)

    @property
    def q95(self) -> float:
        """Global 95th-percentile residual from the rolling buffer."""
        return self.quantile(0.95)

    def quantile(self, q: float) -> float:
        """Global q-quantile residual from the rolling buffer.

        Used to map a controller-chosen quantile (``valley_quantile_controller``)
        back to a residual threshold ``tau`` for ``C_t``. Falls back to 1.0 when
        the buffer is empty.
        """
        if not self._global_buffer:
            return 1.0
        t = torch.tensor(list(self._global_buffer), dtype=torch.float32)
        return float(torch.quantile(t, float(min(max(q, 0.0), 1.0))).item())

    def median(self) -> float:
        """Global median residual from the rolling buffer (robust location)."""
        return self.quantile(0.5)

    def mad(self) -> float:
        """Median absolute deviation * 1.4826 (robust scale ~ std for normal data).

        Used by Proposal 1's robust z-score wrongness. Falls back to 1.0 on an
        empty buffer.
        """
        if not self._global_buffer:
            return 1.0
        t = torch.tensor(list(self._global_buffer), dtype=torch.float32)
        med = t.median()
        return float((t - med).abs().median().item()) * 1.4826

    def channel_q95(self) -> list:
        """Per-channel 95th-percentile residuals from the rolling buffers."""
        out = []
        for buf in self._channel_buffers:
            if not buf:
                out.append(1.0)
            else:
                t = torch.tensor(list(buf), dtype=torch.float32)
                out.append(float(torch.quantile(t, 0.95).item()))
        return out

    @property
    def ema(self) -> float:
        """Global residual EMA (model's historical accuracy signal for C_t)."""
        return float(self.global_ema) if self.global_ema is not None else 0.0
