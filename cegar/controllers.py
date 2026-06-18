"""CEGAR controllers, extracted from ``autocegar/train.py`` as pure functions.

Two controllers are reused *unchanged in concept*; only the surrounding object /
wandb plumbing was stripped so they can be imported standalone:

* :func:`tail_ratio_lambda_controller` -- the ``auto_tr`` lambda rule
  (``autocegar/train.py`` ``LossFunction`` ~L3552-3576). Picks lambda so the tail
  amplification ``(1 + lam*g_p99) / (1 + lam*g_mean)`` approaches a target ratio.
* :func:`valley_quantile_controller` -- the ``auto_q_valley`` tau rule
  (``autocegar/train.py`` ``_ecg_on_epoch_end`` ~L2093-2161). Finds the valley of
  the confidence histogram each epoch and maps it to a quantile threshold.

Both consume the batch/epoch statistics emitted by :func:`cegar.gate.gate`
(``gate_mean``, ``gate_p99``, ``_conf_hist``) and are framework-agnostic.
"""
import math
from typing import Dict, List, Optional, Tuple


def update_ema(prev: Optional[float], value: float, beta: float) -> float:
    """Standard EMA update used throughout the CEGAR controllers."""
    if prev is None or beta <= 0.0:
        return float(value)
    return float(beta * prev + (1.0 - beta) * value)


def tail_ratio_lambda_controller(
    gate_mean_ema: Optional[float],
    gate_p99_ema: Optional[float],
    ratio_target: float = 3.0,
    lam_max: float = 1.5,
    prev_lam_smooth: Optional[float] = None,
    lam_ema: float = 0.9,
    invalid_decay: float = 0.95,
) -> Tuple[float, Dict]:
    """``auto_tr`` lambda rule (core, from ``train.py`` L3552-3576).

    Solving ``(1 + lam*p99) / (1 + lam*mean) = r`` for lam gives
    ``lam = (r - 1) / (p99 - r*mean)``. When the denominator is non-positive the
    target ratio is unreachable, so we decay the previous lambda instead.

    Args:
        gate_mean_ema: EMA of batch ``gate_mean``.
        gate_p99_ema: EMA of batch ``gate_p99``.
        ratio_target: desired tail amplification ratio ``r`` (default 3.0).
        lam_max: hard cap on lambda.
        prev_lam_smooth: previous smoothed lambda (for EMA + invalid decay).
        lam_ema: EMA smoothing factor for lambda.
        invalid_decay: multiplier applied to the previous lambda when the target
            ratio is currently unreachable.

    Returns:
        ``(lam_smooth, info)``.
    """
    r = float(ratio_target)
    if gate_mean_ema is None or gate_p99_ema is None:
        lam_target = 0.0
    else:
        denom = gate_p99_ema - r * gate_mean_ema
        if denom <= 1e-8:
            lam_target = (prev_lam_smooth * invalid_decay) if prev_lam_smooth is not None else 0.0
        else:
            lam_target = (r - 1.0) / denom
        lam_target = min(lam_max, max(0.0, lam_target))

    if prev_lam_smooth is None or lam_ema <= 0.0:
        lam_smooth = lam_target
    else:
        lam_smooth = lam_ema * prev_lam_smooth + (1.0 - lam_ema) * lam_target
    lam_smooth = min(lam_max, max(0.0, lam_smooth))

    info = {"lam_target": float(lam_target), "lam_smooth": float(lam_smooth), "ratio_target": r}
    return float(lam_smooth), info


def valley_quantile_controller(
    conf_hist: List[float],
    prev_q: float,
    ema_beta: float = 0.9,
    smooth: int = 3,
    q_min: float = 0.1,
    q_max: float = 0.99,
) -> Tuple[float, Dict]:
    """``auto_q_valley`` tau rule (from ``train.py`` L2093-2161).

    Gaussian-smooths the (accumulated) confidence histogram, finds the first
    local minimum (the "valley") above conf 0.1 and below 80% of the global max,
    maps that confidence value to a cumulative quantile, then EMA-smooths and
    clamps the result.

    Args:
        conf_hist: accumulated histogram counts (e.g. 50 bins over ``[0, 1]``).
        prev_q: current quantile threshold (EMA state).
        ema_beta: EMA smoothing factor for the quantile.
        smooth: Gaussian kernel half-width in bins.
        q_min / q_max: clamp range for the quantile.

    Returns:
        ``(q_new, info)``. When no valley is found, ``q_new == prev_q`` and
        ``info['valley_bin'] is None``.
    """
    info: Dict = {"valley_bin": None, "valley_conf": None, "q_raw": None}
    if conf_hist is None or len(conf_hist) < 3:
        return float(prev_q), info

    hist = list(conf_hist)
    n_bins = len(hist)
    smooth = max(1, int(smooth))
    total_counts = sum(hist) or 1.0

    # Gaussian smooth
    smoothed = [0.0] * n_bins
    for i in range(n_bins):
        w_sum = 0.0
        c_sum = 0.0
        for d in range(-smooth * 2, smooth * 2 + 1):
            j = i + d
            if 0 <= j < n_bins:
                w = math.exp(-0.5 * (d / max(smooth, 1)) ** 2)
                c_sum += hist[j] * w
                w_sum += w
        smoothed[i] = c_sum / (w_sum or 1.0)

    # Find first local minimum, skipping conf < 0.1, below 80% of global max.
    _bin_min = max(1, n_bins // 10)
    valley_bin = None
    for i in range(_bin_min, n_bins - 1):
        if smoothed[i] <= smoothed[i - 1] and smoothed[i] <= smoothed[i + 1]:
            if smoothed[i] < max(smoothed) * 0.8:
                valley_bin = i
                break

    if valley_bin is None:
        return float(prev_q), info

    valley_conf = (valley_bin + 0.5) / n_bins
    cum = 0.0
    q_valley = 0.5  # fallback
    for i, c in enumerate(hist):
        cum += c
        if cum / total_counts >= valley_conf:
            q_valley = (i + 0.5) / n_bins
            break

    q_new = ema_beta * float(prev_q) + (1.0 - ema_beta) * q_valley
    q_new = float(min(max(q_new, q_min), q_max))

    info.update({"valley_bin": int(valley_bin), "valley_conf": float(valley_conf), "q_raw": float(q_valley)})
    return q_new, info
