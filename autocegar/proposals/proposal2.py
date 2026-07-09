"""Proposal 2 - Uncertainty-Aware Residual CEGAR.

================================ WHAT THIS IS ================================
The doc's cleanest conceptual analogue of classification CEGAR: a **confident
forecasting error = high residual AND low predictive uncertainty**. Same
robust-z residual wrongness as P1, but the CONFIDENCE term is the model's
*inverse predictive uncertainty* instead of a residual-tail signal.

Uncertainty is estimated by **MC-dropout**: run K stochastic forward passes
(dropout kept ON) on the same input and take the per-window prediction variance.
Low variance -> the model is confident -> high C_t.

Only `_compute_signals` is overridden (same reuse pattern as P1); it now also
receives `model_input` (= x + correction) so it can do the extra MC-dropout
forwards. Everything else — warm-up, gate, ScaleGrad, RW-1 correction, score,
interpretability — is inherited from `CNN_RW_CEGAR`.

    E_t (wrongness)  = sigmoid(k  * (robust_z(residual) - tau))       # high residual
    u_t              = mean_feat Var_{k=1..K} f_dropout(x+corr)       # MC-dropout var
    C_t (confidence) = sigmoid(k_u * (tau_u - robust_z(u_t)))         # LOW uncertainty
    gate  = E_t * C_t   (inherited: scale = 1 + lam*gate, then normalized)

Cost: K extra forward passes per batch (medium-high, as the doc notes). Risk:
DeepAnT/RW dropout uncertainty may be poorly calibrated — the fail-fast screen
tells us whether it helps in practice.
=============================================================================
"""
from collections import deque

import torch

from autocegar.rw_cegar import CNN_RW_CEGAR

_MAD_TO_STD = 1.4826


class CNN_RW_CEGAR_P2(CNN_RW_CEGAR):
    """Proposal 2 = robust-z residual wrongness x inverse-MC-dropout-uncertainty confidence.

    New hyperparameters beyond the base:
        mc_samples  number of MC-dropout forward passes for the uncertainty estimate.
        tau_u       robust-z threshold on uncertainty for the confidence sigmoid
                    (C_t = 0.5 when uncertainty sits at this robust-z level).
        k_u         confidence sigmoid sharpness (defaults to k when unset).
        unc_buffer  rolling buffer length for the robust uncertainty stats.
    Reused-with-P2-defaults: tau (residual robust-z threshold, 2.0), k (1.0),
        warmup_epochs (10), correction_init ('zero'), scale_normalize (True).
    """

    PROPOSAL = 2
    NAME = "P2-UncertaintyAware"

    def __init__(self, *args, mc_samples=5, tau_u=0.0, k_u=None,
                 unc_buffer=5000, **kwargs):
        kwargs.setdefault("tau", 2.0)
        kwargs.setdefault("k", 1.0)
        kwargs.setdefault("warmup_epochs", 10)
        kwargs.setdefault("correction_init", "zero")
        kwargs.setdefault("scale_normalize", True)
        super().__init__(*args, **kwargs)
        self.mc_samples = int(mc_samples)
        self.tau_u = float(tau_u)
        self.k_u = float(k_u) if k_u is not None else None
        self._unc_buf = deque(maxlen=int(unc_buffer))

    @staticmethod
    def _median_mad(buf):
        if not buf:
            return 0.0, 1.0
        t = torch.tensor(list(buf), dtype=torch.float32)
        med = t.median()
        return float(med), max(float((t - med).abs().median()) * _MAD_TO_STD, 1e-8)

    def _mc_uncertainty(self, model_input):
        """Per-window predictive uncertainty via K MC-dropout forward passes. [B]."""
        self.model.train(True)  # keep dropout active for the stochastic passes
        B = model_input.shape[0]
        with torch.no_grad():
            preds = torch.stack(
                [self.model(model_input).view(B, self.feats * self.pred_len)
                 for _ in range(self.mc_samples)], dim=0)     # [K, B, F]
        return preds.var(dim=0).mean(dim=1)                    # [B]

    def _compute_signals(self, window_resid, res_stats, model_input=None):
        # Wrongness: standardized residual (robust-z), same as P1.
        med = res_stats.median()
        mad = max(res_stats.mad(), 1e-8)
        E_t = torch.sigmoid(self.k * ((window_resid - med) / mad - self.tau))   # [B]

        # Confidence: inverse predictive uncertainty (low uncertainty -> confident).
        unc = self._mc_uncertainty(model_input)                # [B]
        self._unc_buf.extend(unc.detach().flatten().cpu().tolist())
        u_med, u_mad = self._median_mad(self._unc_buf)
        unc_z = (unc - u_med) / u_mad
        ku = self.k_u if self.k_u is not None else self.k
        C_t = torch.sigmoid(ku * (self.tau_u - unc_z))         # high when unc is low
        return E_t, C_t
