"""Proposal 5 - Temporal-Persistence Confident-Error CEGAR (docx-faithful).

Gate only errors that PERSIST over neighbouring time windows, not isolated residual
spikes. Built on the shared hooks base (rw_cegar_hooks.py).

    m_t   = 1(e_t > tau_e),   tau_e = Q_{q_e}(e)          # residual indicator
    p_t   = mean(m_{t-h .. t+h})                          # temporal persistence
    g     = sigma(k*(e_t-tau_e)/mad) * sigma(k_p*(p_t-tau_p))   # residual x persistence

Persistence needs time-ordered neighbours, but the batch loop is shuffled, so (like P3)
persistence is computed at the epoch end from this-epoch per-timestep residuals and used
NEXT epoch as the amplification gate via the ScaleGrad path (scale = 1 + lam*g_win).
Stage-1 = gradient amplification only; write-back is left unchanged (the amplified
correction gradient already concentrates on persistent segments).

Overrides `_compute_signals` (accumulate residuals + amplification gate) and
`_writeback_scale` (epoch-end persistence compute); `fit` exposes the persistence gate.
"""
import numpy as np
import torch

from autocegar.rw_cegar_hooks import CNN_RW_CEGAR_HookBase

_EPS = 1e-8


class CNN_RW_CEGAR_P5(CNN_RW_CEGAR_HookBase):
    """Proposal 5 = (residual x temporal-persistence) gate, gradient amplification.

    New hyperparameters:
        q_e            quantile of per-timestep residual defining tau_e (0.8).
        persist_h      half-width of the persistence window (p_t over +/-h) (5).
        k_p, tau_p     persistence-gate sigmoid sharpness / threshold.
        persist_alpha  EMA factor smoothing the gate across epochs.
    """

    PROPOSAL = 5
    NAME = "P5-TemporalPersistence"

    def __init__(self, *args, q_e=0.8, persist_h=5, k_p=5.0, tau_p=0.5,
                 persist_alpha=0.9, **kwargs):
        kwargs.setdefault("warmup_epochs", 10)
        kwargs.setdefault("correction_init", "neg_x")
        super().__init__(*args, **kwargs)
        self.q_e = float(q_e)
        self.persist_h = int(persist_h)
        self.k_p = float(k_p)
        self.tau_p = float(tau_p)
        self.persist_alpha = float(persist_alpha)
        self._resid_sum = None      # [T] this-epoch residual accumulation
        self._resid_cnt = None      # [T]
        self._persist = None        # [T] persistence-gate EMA across epochs
        self._g_prev_t = None       # [T] previous-epoch gate (amplification + interp)

    def _accumulate(self, window_resid, yb):
        # Bookkeeping for the docx per-timestep residual e_t: batches are shuffled and
        # window-level, so we scatter each window's residual onto the TIMESTEP it was
        # predicting (yb = target indices from the hooks base) and average at epoch end:
        #   e_t = resid_sum[t] / resid_cnt[t]
        wr = window_resid.detach().cpu().numpy()                    # [B] per-window |resid|
        flat = yb.reshape(-1)
        mx = int(flat.max()) + 1 if flat.size else 0
        if self._resid_sum is None or self._resid_sum.shape[0] < mx:
            # grow the [T] ledgers lazily to the largest target index seen so far
            ns = np.zeros(max(mx, 1)); nc = np.zeros(max(mx, 1))
            if self._resid_sum is not None:
                ns[:self._resid_sum.shape[0]] = self._resid_sum
                nc[:self._resid_cnt.shape[0]] = self._resid_cnt
            self._resid_sum, self._resid_cnt = ns, nc
        # scatter-add: ledger[t] += residual of the window targeting t (and count += 1)
        np.add.at(self._resid_sum, flat, np.repeat(wr, yb.shape[1]))
        np.add.at(self._resid_cnt, flat, 1.0)

    def _compute_signals(self, window_resid, res_stats, model_input=None, target=None):
        # AMPLIFICATION PATH (every batch). Two jobs, no math:
        #   1. deposit this batch's residuals into the per-timestep ledger (for the
        #      persistence computation at epoch end), and
        #   2. look up the PREVIOUS epoch's gate map for this batch's target timesteps.
        # (Same lag as P3: persistence needs the whole epoch's e_t, so epoch e
        # amplifies with the map built at the end of epoch e-1.)
        B = window_resid.shape[0]
        yb = np.asarray(self._cur_yb_idx).reshape(B, -1)            # [B, pred_len] target steps
        self._accumulate(window_resid, yb)
        confidence = torch.ones_like(window_resid)                  # C = 1: the residual-size
        #                                        and persistence factors are both already
        #                                        inside the epoch-end map (g = g_res*g_pers)
        if self._g_prev_t is None:
            # first main epoch: no map yet -> gate 0 -> plain RW-1 behaviour
            return torch.zeros_like(window_resid), confidence
        gp = self._g_prev_t
        idx = np.clip(yb, 0, len(gp) - 1)
        # window gate = mean of the map over this window's target timesteps -> [B]
        gwin = gp[idx].mean(axis=1)
        E_t = torch.as_tensor(gwin, dtype=window_resid.dtype, device=window_resid.device)
        return E_t.clamp(0.0, 1.0), confidence

    def _writeback_scale(self, correction, grad, epoch):
        # EPOCH-END HOOK. P5 uses it ONLY as a timing point to build next epoch's
        # persistence map from this epoch's ledger; the correction gradient itself is
        # returned untouched (Stage-1 = amplification only, no preserve write-back).
        if self._resid_cnt is not None:
            T = correction.shape[2]
            # docx e_t: per-timestep mean residual over this epoch (ledger sum/count)
            e = self._resid_sum / np.maximum(self._resid_cnt, 1.0)  # [T'] e_t
            if len(e) < T:
                e = np.pad(e, (0, T - len(e)))
            e = e[:T]
            # docx tau_e = Q_{q_e}(e): residual threshold from the (positive) quantile,
            # data-adaptive so "large error" means the same thing on every dataset
            pos = e[e > 0]
            tau_e = float(np.quantile(pos, self.q_e)) if pos.size else 0.0
            # docx m_t = 1(e_t > tau_e): binary "large error here" indicator
            m = (e > tau_e).astype(float)
            # docx p_t = mean(m_{t-h .. t+h}): an 11-tap moving average of the mask is
            # exactly "what fraction of my +/-h neighbourhood also exceeds tau_e".
            # Isolated spikes score p ~= 1/(2h+1); true block anomalies score p ~= 1.
            h = self.persist_h
            kern = np.ones(2 * h + 1) / (2 * h + 1)
            p = np.convolve(m, kern, mode="same")                   # [T] p_t
            # docx gate g = sigma(k*(e_t - tau_e)/mad) * sigma(k_p*(p_t - tau_p)):
            # residual-SIZE factor x persistence factor (std-normalized here)
            g_res = 1.0 / (1.0 + np.exp(-self.k_p * (e - tau_e) / (e.std() + _EPS)))
            g_pers = 1.0 / (1.0 + np.exp(-self.k_p * (p - self.tau_p)))
            g = np.clip(g_res * g_pers, 0.0, 1.0)
            # EMA across epochs (persist_alpha): the map only brightens where the
            # residual stays large-and-persistent over consecutive epochs
            if self._persist is None or self._persist.shape != g.shape:
                self._persist = np.zeros_like(g)
            self._persist = self.persist_alpha * self._persist + (1.0 - self.persist_alpha) * g
            self._g_prev_t = np.clip(self._persist, 0.0, 1.0)       # consumed next epoch
            self._resid_sum = self._resid_cnt = None                # fresh ledger next epoch
        return grad                                                 # correction update untouched

    def fit(self, data, train_idx=None):
        scores = super().fit(data, train_idx)
        if self._g_prev_t is not None:
            self.gate_per_t = self._g_prev_t
        return scores
