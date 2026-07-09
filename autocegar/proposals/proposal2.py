"""Proposal 2 - Uncertainty-Aware Residual CEGAR (docx-faithful).

================================ WHAT THIS IS ================================
The doc's cleanest conceptual analogue of classification CEGAR: a confident
forecasting error is high residual with LOW predictive uncertainty. Implemented
straight from the doc's equations (not a robust-z adaptation):

    Ŷ_t^m = f_θ^m(W_t),  m = 1..M          # M MC-dropout forward passes
    μ_t   = (1/M) Σ_m Ŷ_t^m                # MC mean prediction
    u_t   = (1/M) Σ_m ‖Ŷ_t^m − μ_t‖²       # predictive uncertainty (variance, summed over dims)
    e_t   = ‖Y_t − μ_t‖ / (u_t + ε)        # wrongness = residual (vs MC mean) STANDARDIZED by uncertainty
    g_err,t  = σ(k_e · (e_t − τ_e))
    g_conf,t = σ(k_c · (τ_u − u_t))         # confidence: raw u_t, high when uncertainty is low
    g_t = g_err,t · g_conf,t

Only `_compute_signals` is overridden; it receives `model_input` (= x+correction,
for the MC passes) and `target` (= corrected Y_t, for ‖Y_t − μ_t‖). Everything
else — warm-up, gate, ScaleGrad, RW-1 correction, score, interpretability — is
inherited from `CNN_RW_CEGAR`.

DIFFERENCE FROM P1: P1's wrongness is a robust-z of the residual and its
confidence is 1 / a residual-tail sigmoid. P2 puts uncertainty into BOTH the
wrongness (as the standardizing denominator) and the confidence, per the doc.

KNOWN DEVIATION (documented): the doc's write-back multiplies the correction
gradient by an extra conservative `g_safe,t` term to suppress writes on extreme
anomaly tails. That needs a separate correction-gradient path (same single-graph
coupling issue noted for P1's write-back), so it is NOT implemented — the RW
correction uses the inherited ScaleGrad path. `e_t`/`u_t` are un-normalized (per
the doc), so τ_e / τ_u are scale-dependent and may need tuning (MC-dropout
uncertainty can be poorly calibrated — a risk the doc itself flags).
=============================================================================
"""
import torch

from autocegar.rw_cegar import CNN_RW_CEGAR


class CNN_RW_CEGAR_P2(CNN_RW_CEGAR):
    """Proposal 2 = uncertainty-standardized residual wrongness x inverse-uncertainty confidence.

    New hyperparameters beyond the base:
        mc_samples  M — number of MC-dropout forward passes.
        tau_u       τ_u — uncertainty threshold in the confidence sigmoid
                    (g_conf = 0.5 when u_t == τ_u).
        k_u         k_c — confidence sigmoid sharpness (defaults to k when unset).
        unc_eps     ε — stabilizer in the wrongness denominator (u_t + ε).
    Reused-with-P2-defaults: tau (τ_e, residual/uncertainty-ratio threshold), k
        (k_e), warmup_epochs (10), correction_init ('zero'), scale_normalize (True).
    """

    PROPOSAL = 2
    NAME = "P2-UncertaintyAware"

    def __init__(self, *args, mc_samples=5, tau_u=0.0, k_u=None,
                 unc_eps=1e-6, **kwargs):
        kwargs.setdefault("tau", 2.0)
        kwargs.setdefault("k", 1.0)
        kwargs.setdefault("warmup_epochs", 10)
        kwargs.setdefault("correction_init", "zero")
        kwargs.setdefault("scale_normalize", True)
        super().__init__(*args, **kwargs)
        self.mc_samples = int(mc_samples)
        self.tau_u = float(tau_u)
        self.k_u = float(k_u) if k_u is not None else None
        self.unc_eps = float(unc_eps)

    def _mc_forward(self, model_input):
        """M MC-dropout passes. Returns stacked predictions ``[M, B, F]``."""
        self.model.train(True)  # keep dropout active for the stochastic passes
        B = model_input.shape[0]
        with torch.no_grad():
            return torch.stack(
                [self.model(model_input).view(B, self.feats * self.pred_len)
                 for _ in range(self.mc_samples)], dim=0)          # [M, B, F]

    def _compute_signals(self, window_resid, res_stats, model_input=None, target=None):
        preds = self._mc_forward(model_input)                       # [M, B, F]
        mu = preds.mean(dim=0)                                       # μ_t   [B, F]
        u_t = preds.var(dim=0, unbiased=False).sum(dim=1)           # u_t   [B]  (= (1/M)Σ‖·‖²)

        resid = torch.norm(target - mu, dim=1)                      # ‖Y_t − μ_t‖   [B]
        e_t = resid / (u_t + self.unc_eps)                          # standardized residual [B]

        ku = self.k_u if self.k_u is not None else self.k
        g_err = torch.sigmoid(self.k * (e_t - self.tau))           # σ(k_e (e_t − τ_e))
        g_conf = torch.sigmoid(ku * (self.tau_u - u_t))            # σ(k_c (τ_u − u_t)), raw u_t
        return g_err, g_conf
