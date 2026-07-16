"""Proposal 4 - Dual-Gate Residual-and-Gradient RW-CEGAR (docx-faithful).

A point is a confident error if it has HIGH residual AND is gradient-correctable
(small input changes strongly reduce the forecasting loss). Built on the shared hooks
base (rw_cegar_hooks.py), though it only needs the standard per-window gate path.

    g_res  = sigma(k_r*(robust_z(resid) - tau_r))               # residual (as in P1)
    h_t    = ||d loss / d input||   (per window)                # input-gradient norm
    g_grad = sigma(k_h*(robust_z(h_t) - tau_h))                 # gradient-correctability
    g      = g_res * g_grad                                      # dual gate

The gate is computed in-batch (no epoch lag): an extra forward+backward w.r.t. the model
INPUT gives h_t. `benefit` variant uses the actual loss reduction
b_t = l(W) - l(W - eta_x*grad) instead of the raw gradient norm. Intervention is the
inherited gradient amplification (ScaleGrad, scale = 1 + lam*g); per-timestep gate_per_t
is recorded by the base for interpretability. Only `_compute_signals` is overridden.

**Stage-1: amplification only.** The docx's normalized, correctable-point write-back
(`C <- C - eta_C * g * grad/||grad||`) is NOT implemented; the amplified correction
gradient already concentrates on high-(residual*correctability) windows. `correction_init`
is 'neg_x' (RW-1-faithful) like P1/P2/P3/P5; warm-up runs plain RW-1 so no input jump.
Known minor: the input-gradient forward runs with the model in train mode, so dropout
(if any) adds a little noise to h_t.
"""
import numpy as np
import torch

from autocegar.rw_cegar_hooks import CNN_RW_CEGAR_HookBase

_EPS = 1e-8


def _robust_z(v):
    med = v.median()
    mad = (v - med).abs().median() + _EPS
    return (v - med) / mad


class CNN_RW_CEGAR_P4(CNN_RW_CEGAR_HookBase):
    """Proposal 4 = residual gate x input-gradient-correctability gate.

    New hyperparameters:
        k_r, tau_r     residual-gate sigmoid sharpness / robust-z threshold.
        k_h, tau_h     gradient-gate sigmoid sharpness / robust-z threshold.
        use_benefit    if True, gate on the estimated loss reduction b_t instead of ||grad||.
        eta_x          step size for the benefit estimate.
    """

    PROPOSAL = 4
    NAME = "P4-DualResidualGradient"

    def __init__(self, *args, k_r=1.0, tau_r=2.0, k_h=1.0, tau_h=0.0,
                 use_benefit=False, eta_x=0.1, **kwargs):
        kwargs.setdefault("warmup_epochs", 10)
        kwargs.setdefault("correction_init", "neg_x")    # RW-1-faithful (warm-up runs plain RW-1)
        super().__init__(*args, **kwargs)
        self.k_r = float(k_r)
        self.tau_r = float(tau_r)
        self.k_h = float(k_h)
        self.tau_h = float(tau_h)
        self.use_benefit = bool(use_benefit)
        self.eta_x = float(eta_x)

    def _per_window_rmse(self, inp, target):
        out = self.model(inp).view(-1, self.feats * self.pred_len)
        return torch.sqrt(torch.nn.functional.mse_loss(
            out, target, reduction="none").mean(dim=1) + _EPS)      # [B]

    def _compute_signals(self, window_resid, res_stats, model_input=None, target=None):
        # Both docx factors are computed IN-BATCH (no epoch lag, unlike P3/P5).
        #
        # -- factor 1, residual gate (identical in spirit to P1):
        #      g_res = sigma(k_r * (robust_z(resid) - tau_r)),  robust_z = (r - med)/MAD
        #    "is this window's forecast error unusually large vs the running residual
        #    distribution?" median/MAD make tau_r portable across datasets.
        g_res = torch.sigmoid(self.k_r * (_robust_z(window_resid.detach()) - self.tau_r))

        # -- factor 2, gradient-correctability (docx h_t = ||d loss / d input||):
        #    reuse the RW trick as a MEASUREMENT: clone the (corrected) input,
        #    switch gradients on, and ask "if this input could move, how strongly
        #    would the loss push it?". A large input-gradient norm means a small
        #    input edit would cut the loss a lot = the error is CORRECTABLE.
        #    Costs one extra forward+backward per batch.
        inp = model_input.detach().clone().requires_grad_(True)     # detached copy: no
        tgt = target.detach()                                       #   effect on training
        per_win = self._per_window_rmse(inp, tgt)                    # [B] l(W_t) per window
        gnorm = torch.autograd.grad(per_win.sum(), inp)[0]           # [B, feats, W] d l/d input
        h = gnorm.reshape(gnorm.shape[0], -1).norm(dim=1)            # [B] h_t = ||grad|| per window
        if self.use_benefit:
            # `benefit` variant (docx b_t): instead of the raw gradient norm, take one
            # actual gradient step on the input and measure the realized loss drop
            #   b_t = l(W) - l(W - eta_x * grad)
            with torch.no_grad():
                stepped = inp - self.eta_x * gnorm
                benefit = (per_win.detach() - self._per_window_rmse(stepped, tgt))  # [B]
            signal = benefit
        else:
            signal = h
        #    g_grad = sigma(k_h * (robust_z(h_t) - tau_h)); robust-z again so the raw
        #    gradient scale (which varies wildly across datasets) drops out.
        g_grad = torch.sigmoid(self.k_h * (_robust_z(signal.detach()) - self.tau_h))

        # dual gate: base multiplies these as E * C ->  g = g_res * g_grad
        # ("high residual AND gradient-correctable"), then scale = 1 + lam*g.
        return g_res.clamp(0.0, 1.0), g_grad.clamp(0.0, 1.0)
