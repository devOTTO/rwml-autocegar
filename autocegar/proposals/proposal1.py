"""Proposal 1 - Residual-Gated RW-CEGAR.

================================ WHAT THIS IS ================================
The FIRST (simplest, "fail-fast") candidate from
``rw_cegar_research_Proposal_1_to_Proposal_5.docx`` and the July-3 (week-8)
meeting with Luis. The easiest DeepAnT/RW-compatible CEGAR hybrid; the agreed
starting point before the pricier proposals (2 = MC-dropout / uncertainty,
3 = correction-consistency, 4 = dual-gate input-gradient, 5 = temporal
persistence).

IMPLEMENTATION NOTE — reuse over rewrite:
    Proposal 1 differs from the base ``CNN_RW_CEGAR`` (RW-1 + CEGAR gate) ONLY in
    the wrongness / confidence definitions. Everything else — warm-up curriculum,
    the ``gate()`` -> ``ScaleGrad`` mechanism, the auto-lambda / auto-tau
    controllers, the epoch-wise RW-1 correction step, the anomaly score
    (mean|correction|) — is inherited unchanged. So this file is a thin subclass
    that overrides ONLY ``_compute_signals``. Each later proposal follows the same
    pattern (override the smallest hook it needs).

Proposal-1 signals (Luis, week-8):
  * WRONGNESS  E_t = sigmoid(k * (robust_z - tau))
        robust_z = (r - median(r)) / MAD(r)   — a ROBUST z-score (median/MAD),
        NOT a plain z-score, so one fixed ``tau`` behaves the same across datasets
        with very different scale / spread (opportunity vs gecco vs creditcard).
  * CONFIDENCE C_t:
        - conf_mode='basic'    : C_t = 1                         (always confident)
        - conf_mode='quantile' : C_t = sigmoid(k*(r - q_thr)/MAD) (in the tail?)
  * The inherited gate then forms g = E_t * C_t, scale = 1 + lam*g (mean-normalized
    when scale_normalize=True) and applies it to the per-window forecasting-loss
    gradient via ScaleGrad (identity forward -> reported loss stays true RMSE).

Risk (doc "Risks"): most dangerous proposal conceptually — high residual is also
anomaly evidence, so amplifying it can teach the model to "correct away" real
anomalies. The opportunity/gecco/creditcard comparison is meant to expose this.
=============================================================================
"""
import torch

from autocegar.rw_cegar import CNN_RW_CEGAR


class CNN_RW_CEGAR_P1(CNN_RW_CEGAR):
    """Proposal 1 = ``CNN_RW_CEGAR`` with robust-z wrongness + basic/quantile confidence.

    Only new hyperparameters beyond the base class:
        conf_mode  'basic' (C_t = 1) | 'quantile' (residual-tail confidence).
        conf_q     tail quantile for conf_mode='quantile' (default 0.95).

    Reused-with-P1-defaults base hyperparameters:
        tau        robust-z threshold for the wrongness sigmoid (default 2.0).
        k          gate sigmoid sharpness / "augmenting factor" (default 1.0).
        warmup_epochs   forecaster-only warm-up (default 10; week-8 said 10-15).
        correction_init 'zero' (smooth transition after warm-up; default here).
    All other base knobs (lam, lam_mode/tau_mode controllers, scale_normalize,
    l1_weight, ...) keep their base defaults and can still be overridden.
    """

    PROPOSAL = 1
    NAME = "P1-ResidualGated"

    def __init__(self, *args, conf_mode="basic", conf_q=0.95, k_conf=None, **kwargs):
        # Proposal-1-appropriate defaults (caller/​runner can still override).
        kwargs.setdefault("tau", 2.0)             # robust-z units (tau_e), not residual-EMA
        kwargs.setdefault("k", 1.0)               # gentle sigmoid in z-space (k_e)
        kwargs.setdefault("warmup_epochs", 10)
        kwargs.setdefault("correction_init", "zero")
        kwargs.setdefault("scale_normalize", True)  # docx: s_t = (1+lam*g)/mean(1+lam*g)
        super().__init__(*args, **kwargs)
        if conf_mode not in ("basic", "quantile"):
            raise ValueError(f"conf_mode must be 'basic' or 'quantile', got {conf_mode}")
        self.conf_mode = conf_mode
        self.conf_q = float(conf_q)
        # k_conf (k_c) defaults to k (k_e) when not given -> single shared sharpness.
        self.k_conf = float(k_conf) if k_conf is not None else None

    def _compute_signals(self, window_resid, res_stats):
        """Robust-z wrongness x (basic|tail-quantile) confidence. Both ``[B]``.

        Exact docx Proposal-1 formulas:
            e_t      = (r - median(r)) / MAD(r)              # RobustZ
            g_err,t  = sigmoid(k_e * (e_t - tau_e))
            g_conf,t = 1                        (basic)
                     = sigmoid(k_c * (e_t - Q_qc(E)))        (selective)
        The 'selective' branch below uses (r - quantile_q(r))/MAD, which is
        algebraically identical to (e_t - Q_qc(E)) because the median cancels:
            e_t - Q_qc(E) = (r-med)/MAD - (Q_qc(r)-med)/MAD = (r - Q_qc(r))/MAD.
        """
        med = res_stats.median()
        mad = max(res_stats.mad(), 1e-8)
        kc = self.k_conf if self.k_conf is not None else self.k

        robust_z = (window_resid - med) / mad
        E_t = torch.sigmoid(self.k * (robust_z - self.tau))          # g_err,t  [B] in [0,1]

        if self.conf_mode == "basic":
            C_t = torch.ones_like(E_t)                               # g_conf,t = 1
        else:  # 'quantile' — is this residual in the distribution's tail?
            q_thr = res_stats.quantile(self.conf_q)                  # Q_qc in residual units
            C_t = torch.sigmoid(kc * (window_resid - q_thr) / mad)   # == sigmoid(kc*(e_t - Q_qc(E)))
        return E_t, C_t
