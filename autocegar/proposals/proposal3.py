"""Proposal 3 - RW-Correction-Consistency CEGAR (Stage 1).

================================ WHAT THIS IS ================================
The signal is the RW correction's OWN behaviour, not the residual (P1) or the
predictive uncertainty (P2). A point that is corrected large AND consistently over
epochs is a "confident RW anomaly candidate", and P3's key move is to STOP writing
there (preserve the correction as evidence) instead of reshaping it away, which is
what sinks P1/P2 (their gate finds anomalies and then "corrects them away", lowering
the |correction| score).

Stage 1 (this file) implements the persistence + preserve-write-back core:
    d_t   = mean_feat |C_t|                          # per-timestep correction magnitude
    pi_t  = alpha*pi_t + (1-alpha)*1(d_t > Q_q(d))   # persistence: consistently large?
    g_t   = pi_t                                     # confident (consistent) candidate
    write-back:  grad_C <- grad_C * (1 - gamma*g_t)  # SUPPRESS writes on those points

So a point that keeps landing in the top (1-corr_q) of correction magnitude, epoch
after epoch, gets its correction frozen (preserved) rather than further updated.

Isolation: the model-gradient CEGAR gate (P1/P2's ScaleGrad path) is turned OFF here
(`lam=0`, so the amplification scale is 1) so this run measures ONLY the effect of
the preserve-write-back. Everything else (warm-up, correction optimizer, epoch-wise
RW-1 step, score = mean|correction|, logging) is inherited from `CNN_RW_CEGAR`.

Only two things are overridden: `_writeback_scale` (the new base hook) applies the
(1-gamma*g) suppression; `fit` then exposes P3's persistence gate as `gate_per_t`
for the interpretability metrics (the inherited gate_per_t would be ~0 with lam=0).

Stage 2 (later): add direction stability v_t = cos(dC^(e), dC^(e-1)) into the gate
(g = g_corr * g_stab) and optionally re-enable the correction-consistency gate on the
model gradient. The base hook already receives the full correction each epoch, so the
epoch-to-epoch history needed for v_t can be stashed here without further base changes.
=============================================================================
"""
import numpy as np
import torch

from autocegar.rw_cegar import CNN_RW_CEGAR


class CNN_RW_CEGAR_P3(CNN_RW_CEGAR):
    """Proposal 3 = correction-magnitude x persistence gate, preserve-write-back.

    New hyperparameters beyond the base:
        gamma          write-back suppression strength; grad *= (1 - gamma*g),
                       gamma=1 fully freezes confident-anomaly corrections.
        corr_q         quantile of |correction| defining "large" this epoch (0.95).
        persist_alpha  EMA factor for the persistence indicator across epochs (0.9).
    Reused-with-P3-defaults: lam=0 (model-gradient gate off, Stage-1 isolation),
        warmup_epochs (10), correction_init ('zero').
    """

    PROPOSAL = 3
    NAME = "P3-CorrectionConsistency"

    def __init__(self, *args, gamma=0.9, corr_q=0.95, persist_alpha=0.9, **kwargs):
        # Stage 1 isolation: model-gradient gate OFF. Forced (not setdefault) because
        # run_proposal.py always passes --lam (default 1.0), which would otherwise
        # re-enable the base placeholder gate and confound the write-back-only test.
        kwargs["lam"] = 0.0
        kwargs.setdefault("warmup_epochs", 10)
        kwargs.setdefault("correction_init", "zero")
        super().__init__(*args, **kwargs)
        self.gamma = float(gamma)
        self.corr_q = float(corr_q)
        self.persist_alpha = float(persist_alpha)
        self._persist = None       # [T] persistence EMA, carried across epochs
        self._p3_gate = None       # [T] final-epoch gate, for interpretability

    def _writeback_scale(self, correction, grad, epoch):
        # per-timestep correction magnitude (same reduction as the |correction| score)
        d = correction.detach()[0].abs().mean(dim=0)                  # [T]
        thr = torch.quantile(d, self.corr_q)
        indicator = (d > thr).float()                                 # [T] in the top tail this epoch
        if self._persist is None or self._persist.shape != d.shape:
            self._persist = torch.zeros_like(d)
        self._persist = self.persist_alpha * self._persist + (1.0 - self.persist_alpha) * indicator
        g = self._persist.clamp(0.0, 1.0)                             # [T] consistently-large -> confident
        self._p3_gate = g.detach().cpu().numpy()
        # SUPPRESS writes on confident, consistent corrections -> preserve as evidence
        scale = (1.0 - self.gamma * g).clamp(0.0, 1.0)                # [T]
        return grad * scale.view(1, 1, -1)

    def fit(self, data, train_idx=None):
        scores = super().fit(data, train_idx)
        if self._p3_gate is not None:
            # report P3's persistence gate (the inherited gate_per_t is ~0 with lam=0)
            self.gate_per_t = self._p3_gate
        return scores
