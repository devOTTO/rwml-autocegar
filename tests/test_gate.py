"""Unit tests for the CEGAR gate machinery.

The gate *mechanics* (ScaleGrad, scale formula, normalization, controllers) are
fully tested here -- they do not depend on Luis's notebook.

The exact residual E_t / C_t worked example
(value=2, conf=0.9, q95=10 -> sigmoid ~= 0.881 -> no-rewrite) is left as a
PLACEHOLDER (``test_worked_example_pending_notebook``): the qualitative behavior
is asserted now, and the exact numeric constants (tau, k, normalization) get
filled in once the notebook arrives.

Run with:  pytest tests/ -v
"""
import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from cegar import (  # noqa: E402
    ScaleGrad,
    gate,
    tail_ratio_lambda_controller,
    valley_quantile_controller,
    compute_wrongness_E_t,
    compute_confidence_C_t,
)


# ----------------------------------------------------------------- ScaleGrad
def test_scale_grad_forward_is_identity():
    x = torch.randn(4, 3, requires_grad=True)
    scale = torch.rand(4, 1)
    out = ScaleGrad.apply(x, scale)
    assert torch.allclose(out, x)


def test_scale_grad_backward_rescales_gradient():
    x = torch.ones(3, 2, requires_grad=True)
    scale = torch.tensor([[2.0], [0.5], [4.0]])
    out = ScaleGrad.apply(x, scale)
    out.sum().backward()
    # grad of sum wrt x is 1, rescaled per-row by scale
    assert torch.allclose(x.grad, scale.expand_as(x))


# ---------------------------------------------------------------------- gate
def test_gate_scale_formula():
    conf = torch.tensor([1.0, 1.0, 0.0])
    wrong = torch.tensor([1.0, 0.5, 1.0])
    lam = 2.0
    scale, _ = gate(conf, wrong, lam=lam, scale_normalize=False)
    # scale = 1 + lam * (wrong * conf)
    expected = 1.0 + lam * (wrong * conf)
    assert torch.allclose(scale, expected)


def test_gate_scale_normalize_mean_one():
    conf = torch.rand(64)
    wrong = torch.rand(64)
    scale, stats = gate(conf, wrong, lam=1.0, scale_normalize=True)
    assert abs(scale.mean().item() - 1.0) < 1e-5
    assert "scale_p99_after_norm" in stats


def test_gate_stats_keys_for_controllers():
    conf = torch.rand(128)
    wrong = torch.rand(128)
    _, stats = gate(conf, wrong, lam=0.5)
    for key in ("gate_mean", "gate_p95", "gate_p99", "conf_gate_active_frac", "_conf_hist"):
        assert key in stats
    assert len(stats["_conf_hist"]) == 50


# --------------------------------- qualitative CEGAR behavior (notebook-free)
def test_large_error_high_conf_rewrites():
    """Large error + high confidence -> large gate -> meaningful rewrite (scale > 1)."""
    conf = torch.tensor([0.95])   # high confidence
    wrong = torch.tensor([1.0])   # large (clamped) error
    scale, _ = gate(conf, wrong, lam=1.0, scale_normalize=False)
    assert scale.item() > 1.5


def test_small_error_high_conf_no_rewrite():
    """Small error + high confidence -> tiny gate -> effectively no rewrite (scale ~ 1)."""
    conf = torch.tensor([0.95])   # high confidence
    wrong = torch.tensor([0.05])  # small error
    scale, _ = gate(conf, wrong, lam=1.0, scale_normalize=False)
    assert scale.item() == pytest.approx(1.0, abs=0.1)


# ----------------------------------------------------------------- controllers
def test_tail_ratio_lambda_controller_solves_ratio():
    # With p99=0.9, mean=0.1, target r=3: lam = (r-1)/(p99 - r*mean) = 2 / 0.6 = 3.33 -> capped.
    lam, info = tail_ratio_lambda_controller(0.1, 0.9, ratio_target=3.0, lam_max=10.0,
                                             prev_lam_smooth=None, lam_ema=0.0)
    assert lam == pytest.approx(2.0 / 0.6, rel=1e-4)
    # achieved ratio check
    achieved = (1 + lam * 0.9) / (1 + lam * 0.1)
    assert achieved == pytest.approx(3.0, rel=1e-3)


def test_tail_ratio_lambda_controller_unreachable_decays():
    # denom <= 0 (p99 < r*mean): unreachable -> decay previous lambda
    lam, _ = tail_ratio_lambda_controller(0.5, 0.5, ratio_target=3.0, lam_max=10.0,
                                          prev_lam_smooth=1.0, lam_ema=0.0, invalid_decay=0.9)
    assert lam == pytest.approx(0.9)


def test_valley_controller_finds_bimodal_valley():
    # Bimodal histogram: peak low, valley middle, peak high.
    hist = [50] * 10 + [2] * 5 + [50] * 35
    q_new, info = valley_quantile_controller(hist, prev_q=0.6, ema_beta=0.0)
    assert info["valley_bin"] is not None
    assert 0.1 <= q_new <= 0.99


def test_valley_controller_no_valley_keeps_prev():
    hist = [10] * 50  # flat: no valley below 80% of max
    q_new, info = valley_quantile_controller(hist, prev_q=0.42, ema_beta=0.0)
    assert info["valley_bin"] is None
    assert q_new == pytest.approx(0.42)


# ------------------------------------------------- residual signal placeholders
def test_E_t_clamps_to_one():
    r = torch.tensor([5.0, 15.0, 2.0])
    E_t = compute_wrongness_E_t(r, residual_q95=10.0)
    assert torch.allclose(E_t, torch.tensor([0.5, 1.0, 0.2]))


def test_C_t_high_when_residual_low():
    # low historical residual -> high confidence
    c_low_resid = compute_confidence_C_t(residual_ema=0.0, tau=1.0, k=5.0)
    c_high_resid = compute_confidence_C_t(residual_ema=2.0, tau=1.0, k=5.0)
    assert c_low_resid > 0.9
    assert c_high_resid < 0.1


@pytest.mark.skip(reason="PENDING Luis's notebook: exact tau/k/normalization constants.")
def test_worked_example_pending_notebook():
    """Worked example from Luis: value=2, conf=0.9, q95=10 -> sigmoid ~= 0.881 -> no-rewrite.

    Fill in the exact E_t/C_t constants once the notebook arrives:
        E_t = compute_wrongness_E_t(value=2, residual_q95=10)  # -> 0.2
        C_t = compute_confidence_C_t(..., tau=?, k=?)          # -> ~= 0.881
        gate = E_t * C_t                                       # small -> no rewrite
        assert scale ~= 1  (no-rewrite)
    """
    raise NotImplementedError
