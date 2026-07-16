"""Proposal 3 - RW-Correction-Consistency CEGAR (docx-faithful, full).

================================ WHAT THIS IS ================================
The signal is the RW correction's OWN behaviour, not the residual (P1) or the
predictive uncertainty (P2). A point that is corrected large AND in a consistent
direction over epochs is a "confident RW anomaly candidate". P3's move is to
PRESERVE the correction there (as anomaly evidence) instead of reshaping it away,
which is what sinks P1/P2 (their gate finds anomalies and then "corrects them away",
lowering the |correction| score).

This implements the docx Proposal-3 formulation directly (no staging):

    d_t^e      = mean_feat |C_t^e|                              # correction magnitude
    v_t^e      = cos( C_t^e − C_t^{e-1} , C_t^{e-1} − C_t^{e-2} )   # direction stability
    g_corr,t   = σ( k_d · (d_t − τ_d) / sd(d) ),  τ_d = Q_{corr_q}(d)
    g_stab,t   = σ( k_v · (v_t − τ_v) )
    g_t        = g_corr,t · g_stab,t                            # persistence-EMA smoothed

Both docx mechanisms are on:
  1. Gradient amplification — g_t drives the per-window model/correction gradient via
     the inherited ScaleGrad path (scale = 1 + λ·g_win), amplifying updates on windows
     overlapping stable-corrected points. g_win for epoch e uses epoch e−1's g_t (the
     current epoch's correction is not finalized until its epoch-end step).
  2. Preserve write-back — the epoch-wise RW correction step is suppressed on those
     points: grad_C ← grad_C · (1 − γ·g_t)  (docx "more safely" variant η_C(1−γg)).

Note the docx lists both mechanisms together; they pull in opposite directions
(amplify grows the correction, preserve freezes it), so their net effect is
empirical — this run measures exactly that. The `preserve_only` variant (λ=0) isolates
the write-back for an ablation.

Inherited unchanged from `CNN_RW_CEGAR`: warm-up, RW-1 epoch-wise correction step,
L1 penalty, forward loss RMSE(X+C, Y+C)+α‖C‖₁, score = mean|correction|, logging.
Only `_compute_signals` (amplification gate) and `_writeback_scale` (gate compute +
preserve write-back) are overridden; `fit` exposes the consistency gate as
`gate_per_t` for the interpretability metrics.
=============================================================================
"""
import numpy as np
import torch

# P3/P4/P5 share the hooks base (rw_cegar_hooks.py): P1/P2 base + _writeback_scale
# hook + per-window target-index stash. The P1/P2 base (rw_cegar.py) stays clean.
from autocegar.rw_cegar_hooks import CNN_RW_CEGAR_HookBase

_EPS = 1e-8


class CNN_RW_CEGAR_P3(CNN_RW_CEGAR_HookBase):
    """Proposal 3 = (correction-magnitude x direction-stability) gate, driving BOTH
    gradient amplification and a preserve (suppress) write-back.

    New hyperparameters beyond the base:
        gamma          write-back suppression strength; grad_C *= (1 - gamma*g),
                       gamma=1 fully freezes confident-anomaly corrections.
        corr_q         quantile of |correction| defining tau_d ("large") each epoch.
        k_d, k_v       sigmoid sharpness for the magnitude / stability gates.
        tau_v          direction-stability threshold (cos); 0 => aligned deltas pass.
        persist_alpha  EMA factor smoothing the gate across epochs (persistence).
        amplify        if False, force lam=0 (preserve-only ablation, no gradient amp).
    """

    PROPOSAL = 3
    NAME = "P3-CorrectionConsistency"

    def __init__(self, *args, gamma=0.9, corr_q=0.95, k_d=1.0, k_v=5.0, tau_v=0.0,
                 persist_alpha=0.9, amplify=True, **kwargs):
        if not amplify:
            # preserve-only ablation: isolate the write-back (no model-gradient amp)
            kwargs["lam"] = 0.0
        kwargs.setdefault("warmup_epochs", 10)
        kwargs.setdefault("correction_init", "neg_x")
        super().__init__(*args, **kwargs)
        self.gamma = float(gamma)
        self.corr_q = float(corr_q)
        self.k_d = float(k_d)
        self.k_v = float(k_v)
        self.tau_v = float(tau_v)
        self.persist_alpha = float(persist_alpha)
        self.amplify = bool(amplify)
        self._persist = None       # [T] gate EMA, carried across epochs
        self._C_prev = None        # [feats, T] correction at the previous epoch end
        self._dC_prev = None       # [feats, T] previous epoch-to-epoch delta
        self._g_prev_t = None      # [T] last computed gate (drives next-epoch amp + interp)

    # ── gradient amplification: map the previous-epoch consistency gate onto the
    #    per-window wrongness, so scale = 1 + lam*g_win up-weights stable-correction
    #    windows. confidence = 1 (the consistency IS the signal). ──
    def _compute_signals(self, window_resid, res_stats, model_input=None, target=None):
        # Amplification path: look up the epoch-(e-1) gate map built in
        # _writeback_scale (the correction is only final at epoch end).
        B = window_resid.shape[0]
        confidence = torch.ones_like(window_resid)            # C = 1 (consistency IS the signal)
        if not self.amplify or self._g_prev_t is None:
            # preserve-only ablation / first main epoch: gate 0 = plain RW-1
            return torch.zeros_like(window_resid), confidence
        yb = np.asarray(self._cur_yb_idx).reshape(B, -1)      # [B, pred_len] target timesteps
        gwin = self._g_prev_t[yb].mean(axis=1)                # [B] g_win = mean_t g_prev[t]
        E_t = torch.as_tensor(gwin, dtype=window_resid.dtype, device=window_resid.device)
        return E_t.clamp(0.0, 1.0), confidence                # base: scale = 1 + lam*E*C

    def _writeback_scale(self, correction, grad, epoch):
        # Epoch end: (A) build next epoch's gate map g_t, (B) preserve write-back.
        C = correction.detach()[0]                            # [feats, T]
        d = C.abs().mean(dim=0)                               # d_t = ||C_t|| (mean|.| = same summary as the score)
        tau_d = torch.quantile(d, self.corr_q)                # tau_d = Q_{corr_q}(d), auto threshold
        g_corr = torch.sigmoid(self.k_d * (d - tau_d) / (d.std() + _EPS))   # magnitude gate: sigma of k_d*(d - tau_d)/sd  [T]

        # v_t = cos(C^e - C^{e-1}, C^{e-1} - C^{e-2}); needs 2 deltas -> first 2 epochs v=0
        if self._C_prev is not None:
            dC = C - self._C_prev                             # [feats, T] this epoch's delta
        else:
            dC = torch.zeros_like(C)
        if self._dC_prev is not None:
            num = (dC * self._dC_prev).sum(dim=0)             # [T] dot over features
            den = dC.norm(dim=0) * self._dC_prev.norm(dim=0) + _EPS
            v = num / den                                     # [T] cos in [-1, 1]
        else:
            v = torch.zeros_like(d)
        g_stab = torch.sigmoid(self.k_v * (v - self.tau_v))   # stability gate: sigma of k_v*(v - tau_v)  [T]

        g = (g_corr * g_stab).clamp(0.0, 1.0)                 # docx gate: g_corr times g_stab
        # docx pi (persistence) folded in as an EMA of the gate across epochs
        if self._persist is None or self._persist.shape != g.shape:
            self._persist = torch.zeros_like(g)
        self._persist = self.persist_alpha * self._persist + (1.0 - self.persist_alpha) * g
        g_use = self._persist.clamp(0.0, 1.0)
        self._g_prev_t = g_use.detach().cpu().numpy()         # next-epoch amp gate + interp

        # roll history AFTER v used the previous deltas
        self._dC_prev = dC.clone()
        self._C_prev = C.clone()

        # preserve (docx eta_C(1-gamma*g)): grad = accumulated RMSE + L1, so this is
        # the one place the L1 shrink can be damped on confident timesteps
        scale = (1.0 - self.gamma * g_use).clamp(0.0, 1.0)    # [T]
        return grad * scale.view(1, 1, -1)

    def fit(self, data, train_idx=None):
        scores = super().fit(data, train_idx)
        if self._g_prev_t is not None:
            # report P3's consistency gate (the inherited per-window gate is the
            # amplification signal, which is ~0 in the preserve-only ablation)
            self.gate_per_t = self._g_prev_t
        return scores
