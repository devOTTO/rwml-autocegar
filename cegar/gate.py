"""Generalized CEGAR gate.

This is the B2 abstraction: the original classification gate in
``autocegar/ecg_loss.py`` computed its two factors from logits/targets::

    wrong_gate = 1 - p_y                      # how wrong
    conf_gate  = sigmoid(k * (conf - tau))    # how confident
    gate       = wrong_gate * conf_gate
    scale      = 1 + lam * gate               # applied via ScaleGrad

Here we change *only the input signature*: the gate receives the two factors
directly as ``confidence`` and ``wrongness`` (each already in ``[0, 1]``), so any
domain (classification, residual-based AD, ...) can drive it. The
gate/scale/normalization math and the emitted ``stats`` are kept identical to the
original so the tau/lambda controllers in :mod:`cegar.controllers` consume them
unchanged.
"""
from typing import Dict, Optional, Tuple

import torch

_CONF_HIST_BINS = 50


def gate(
    confidence: torch.Tensor,
    wrongness: torch.Tensor,
    lam: float = 1.0,
    scale_normalize: bool = False,
    detach_gates: bool = True,
    conf_raw: Optional[torch.Tensor] = None,
    hist_bins: int = _CONF_HIST_BINS,
    hist_min: float = 0.0,
    hist_max: float = 1.0,
    minimal_stats: bool = False,
) -> Tuple[torch.Tensor, Dict]:
    """Compute the per-sample gradient ``scale`` and control statistics.

    Args:
        confidence: conf-gate analog in ``[0, 1]`` (was ``sigmoid(k(conf-tau))``).
        wrongness:  wrong-gate analog in ``[0, 1]`` (was ``1 - p_y``).
        lam: gate strength lambda.
        scale_normalize: if True, divide scale so ``mean(scale) == 1`` (used with
            auto-lambda to keep the global step size unchanged).
        detach_gates: detach the gate from the autograd graph (the gate is a
            weighting signal, not something to backprop through).
        conf_raw: optional raw (pre-sigmoid) confidence used only to build the
            histogram consumed by the valley-detection controller. Defaults to
            ``confidence`` when not provided.
        hist_bins / hist_min / hist_max: histogram binning for valley detection.
        minimal_stats: skip the logging-only statistics.

    Returns:
        ``(scale, stats)`` where ``scale`` is a 1-D tensor of shape ``[B]`` (reshape
        at the call site before passing to ``ScaleGrad``), and ``stats`` mirrors the
        dict emitted by the original ``ecg_loss``.
    """
    confidence = confidence.reshape(-1).float()
    wrongness = wrongness.reshape(-1).float()

    if detach_gates:
        confidence = confidence.detach()
        wrongness = wrongness.detach()

    # --- gate + scale (logic untouched from ecg_loss) ---
    gate_vals = wrongness * confidence
    scale = 1.0 + lam * gate_vals
    if scale_normalize:
        scale = scale / (scale.mean().detach().clamp_min(1e-8))

    # --- control stats (consumed by controllers; mirror ecg_loss) ---
    g_flat = gate_vals.detach().float().view(-1)
    gate_mean_val = g_flat.mean().item()
    if g_flat.numel() <= 1:
        gate_p95_val = g_flat.item() if g_flat.numel() == 1 else 0.0
        gate_p99_val = gate_p95_val
    else:
        gate_p95_val = torch.quantile(g_flat, 0.95).item()
        gate_p99_val = torch.quantile(g_flat, 0.99).item()
    active_frac_val = (confidence > 0.5).float().mean().item()

    c_for_hist = (conf_raw if conf_raw is not None else confidence).detach().float().view(-1)
    conf_hist_counts = torch.histc(c_for_hist, bins=hist_bins, min=hist_min, max=hist_max)

    stats: Dict = {
        "gate_mean": gate_mean_val,
        "gate_p95": gate_p95_val,
        "gate_p99": gate_p99_val,
        "conf_gate_active_frac": active_frac_val,
        "_conf_hist": conf_hist_counts.cpu().tolist(),
    }
    if scale_normalize:
        s_flat = scale.detach().float().view(-1)
        if s_flat.numel() <= 1:
            stats["scale_p99_after_norm"] = s_flat.item() if s_flat.numel() == 1 else 0.0
        else:
            stats["scale_p99_after_norm"] = torch.quantile(s_flat, 0.99).item()

    if not minimal_stats:
        gate_std_val = g_flat.std().item() if g_flat.numel() > 1 else 0.0
        stats.update({
            "gate_std": gate_std_val,
            "wrong_mean": wrongness.mean().item(),
            "conf_mean": confidence.mean().item(),
            "scale_mean": scale.mean().item(),
        })

    return scale, stats
