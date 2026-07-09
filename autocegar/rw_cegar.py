"""RW-1 + CEGAR — the AutoCEGAR trainer built on the *reproduction-successful*
RW-1 (paper Algorithm 2) rather than the old standalone DeepAnT track.

`CNN_RW_CEGAR` subclasses the reproduced `rw.cnn_rw.CNN_RW` (linear gradient,
weak L1, epoch-wise correction update — the config that reproduces the paper)
and steers the `correction` tensor toward likely-anomalous windows via a CEGAR
gate:

    gate       = clamp(E_t * C_t, 0, 1)          # wrong AND historically-confident
    scale      = 1 + lam * gate                  # per-window gradient weight
    (scale is applied to the per-window RMSE **gradient** via ScaleGrad, so the
     reported loss stays the true RMSE while the correction gradient of gated
     windows is up-weighted.)

Wiring implemented here (previously scaffolded but unused):
  * gate()      — `autocegar.gate.gate` computes the per-window scale + the
                  control statistics the λ/τ controllers consume.
  * ScaleGrad   — `autocegar.scale_grad.ScaleGrad` applies the gate to the
                  gradient only (identity forward), instead of multiplying the
                  loss value.
  * controllers — `tail_ratio_lambda_controller` (auto-λ) and
                  `valley_quantile_controller` (auto-τ) run at each epoch end and
                  carry their smoothed state across epochs *and across the
                  warm-up→main phase boundary* (phase continuity).
  * warm-up     — `warmup_epochs` trains the CNN forecaster alone (no correction,
                  no gate) so it learns the normal dynamics first; the residual
                  statistics keep updating during warm-up so E_t/C_t are already
                  warm when the gate switches on.

Defaults reproduce the previous behaviour (`warmup_epochs=0`, `lam_mode='fixed'`,
`tau_mode='fixed'`, `correction_init='neg_x'`).

CEGAR signal formulas (E_t, C_t) live in `autocegar.residual_signals` and are
still the placeholder forms pending Luis's notebook. In particular C_t is a
single scalar per batch under the placeholder, so `tau_mode='auto_q_valley'` is
wired but effectively a no-op (a degenerate confidence histogram has no valley)
until C_t becomes per-window; auto-λ does not depend on the C_t formula and is
fully functional. `k` has no controller in the codebase and stays a fixed
hyperparameter.
"""
import numpy as np
import torch
from torch import optim

from rw.cnn_rw import CNN_RW, _rmse  # reproduced RW-1 (Algorithm 2)
from autocegar.residual_signals import (
    ResidualStats, compute_wrongness_E_t, compute_confidence_C_t,
)
from autocegar.gate import gate as compute_gate
from autocegar.scale_grad import ScaleGrad
from autocegar.controllers import (
    update_ema, tail_ratio_lambda_controller, valley_quantile_controller,
)


class CNN_RW_CEGAR(CNN_RW):
    """RW-1 with a CEGAR gate steering the correction gradient.

    Extra hyperparameters beyond CNN_RW:
        lam         gate strength (0 => plain RW-1). Initial value; becomes the
                    controller state when lam_mode='auto_tr'.
        tau, k      C_t sigmoid threshold / sharpness.
        ema_beta    residual EMA smoothing for C_t.
        buffer_size rolling buffer for the q95 used by E_t.
        warmup_epochs      epochs of forecaster-only training before the gate /
                           correction switch on (curriculum). 0 = off.
        correction_init    'neg_x' (Algorithm-2 faithful, default) or 'zero'.
                           With warm-up, 'zero' gives a continuous transition
                           (the model warms up on x, then x+correction≈x); with
                           'neg_x' the input jumps once when correction turns on.
        lam_mode           'fixed' | 'auto_tr' (tail-ratio λ controller).
        tau_mode           'fixed' | 'auto_q_valley' (valley-detection τ controller).
        ratio_target, lam_max, lam_ctrl_ema   auto-λ controller knobs.
        tau_ctrl_ema, tau_q_init               auto-τ controller knobs.
        scale_normalize    normalize the gate scale so mean(scale)=1 (keeps the
                           global correction step size unchanged under auto-λ).
    """

    def __init__(self, *args, lam=1.0, tau=0.5, k=10.0,
                 ema_beta=0.9, buffer_size=5000,
                 warmup_epochs=0, correction_init='neg_x',
                 lam_mode='fixed', tau_mode='fixed',
                 ratio_target=3.0, lam_max=1.5, lam_ctrl_ema=0.9,
                 tau_q_init=0.5, tau_ctrl_ema=0.9,
                 scale_normalize=False, **kwargs):
        super().__init__(*args, **kwargs)
        self.lam = lam
        self.tau = tau
        self.k = k
        self.ema_beta = ema_beta
        self.buffer_size = buffer_size

        self.warmup_epochs = int(warmup_epochs)
        if correction_init not in ("neg_x", "zero"):
            raise ValueError(f"correction_init must be 'neg_x' or 'zero', got {correction_init}")
        self.correction_init = correction_init

        if lam_mode not in ("fixed", "auto_tr"):
            raise ValueError(f"lam_mode must be 'fixed' or 'auto_tr', got {lam_mode}")
        if tau_mode not in ("fixed", "auto_q_valley"):
            raise ValueError(f"tau_mode must be 'fixed' or 'auto_q_valley', got {tau_mode}")
        self.lam_mode = lam_mode
        self.tau_mode = tau_mode
        self.ratio_target = ratio_target
        self.lam_max = lam_max
        self.lam_ctrl_ema = lam_ctrl_ema
        self.tau_q_init = tau_q_init
        self.tau_ctrl_ema = tau_ctrl_ema
        self.scale_normalize = scale_normalize

        # controller state, carried across epochs AND the warm-up→main boundary
        self._lam_smooth = float(lam)
        self._tau_q = float(tau_q_init)
        self._gate_mean_ema = None
        self._gate_p99_ema = None

        # optional per-epoch callback: fn(metrics: dict) -> None. Set by the
        # caller (e.g. run_proposal.py) to stream metrics to wandb during
        # training. None => no-op, so the base behaviour is unchanged.
        self.on_epoch_end = None

    def _init_correction(self, ts):
        if self.correction_init == "zero":
            return torch.zeros_like(ts).detach().requires_grad_(True)
        return (-ts.clone()).detach().requires_grad_(True)

    def _compute_signals(self, window_resid, res_stats, model_input=None):
        """Wrongness E_t and confidence C_t (both ``[B]`` in ``[0, 1]``).

        THIS is the only per-proposal knob. The base class keeps the placeholder
        residual signals (q95-ratio wrongness + scalar historical-EMA confidence);
        each proposal subclass overrides ONLY this method to plug in its own
        wrong/confidence definitions, reusing warm-up / gate / ScaleGrad /
        controllers unchanged.

        Args:
            window_resid: per-window residual magnitude ``[B]``.
            res_stats:    the live :class:`ResidualStats` (q95, ema, quantile, ...).
            model_input:  the exact tensor fed to the model this batch
                          (``x + x_corr``, shape ``[B, feats, W]``). Proposals that
                          need extra forward passes (e.g. P2 MC-dropout uncertainty)
                          use it; P1 / the base ignore it.

        Returns:
            ``(E_t, confidence)`` each a tensor of shape ``[B]``.
        """
        E_t = compute_wrongness_E_t(window_resid, res_stats.q95)      # [B]
        C_t = compute_confidence_C_t(res_stats.ema, self.tau, self.k)  # scalar
        confidence = torch.full_like(E_t, float(C_t))                 # broadcast -> [B]
        return E_t, confidence

    def fit(self, data, train_idx=None):
        print(f"Training CNN_RW_CEGAR (RW-1 + CEGAR gate) | warmup={self.warmup_epochs} "
              f"| lam_mode={self.lam_mode} | tau_mode={self.tau_mode} "
              f"| correction_init={self.correction_init}...")
        ts = self._normalize(data)
        ts = torch.from_numpy(ts).float().permute(1, 0).unsqueeze(0).to(self.device).contiguous()

        correction = self._init_correction(ts)
        total_length = ts.shape[2]
        X_indices, Y_indices = self.create_sliding_window(
            total_length, self.window_size, self.pred_len, shuffle=True)

        self.model_optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
        self.correction_optimizer = optim.RMSprop(
            [correction], lr=self.correction_rate, alpha=0.99, eps=1e-8)

        res_stats = ResidualStats(self.feats, ema_beta=self.ema_beta,
                                  buffer_size=self.buffer_size)

        # Interpretability: per-timestep gate activation, recorded in the FINAL
        # epoch, so we can later check whether the gate fired AT anomalies (vs the
        # ground-truth labels) and whether correction concentrated there.
        gate_accum = np.zeros(total_length, dtype=np.float64)
        gate_count = np.zeros(total_length, dtype=np.float64)

        for epoch in range(1, self.epochs + 1):
            warm = epoch <= self.warmup_epochs
            self.model.train(mode=True)
            avg_loss, n_batches, gate_sum = 0.0, 0, 0.0
            if not warm:
                self.correction_optimizer.zero_grad()

            # per-epoch controller accumulators
            conf_hist_accum = None

            for i in range(0, X_indices.shape[0], self.batch_size):
                xb_idx = X_indices[i:i + self.batch_size]
                yb_idx = Y_indices[i:i + self.batch_size]

                x = ts[0, :, xb_idx].permute(1, 0, 2)
                target = ts[0, :, yb_idx].permute(1, 0, 2)

                self.model_optimizer.zero_grad()

                if warm:
                    # ── warm-up: forecaster only, no correction / no gate ──
                    output = self.model(x).view(-1, self.feats * self.pred_len)
                    target_full = target.reshape(-1, self.feats * self.pred_len)
                    B = output.shape[0]
                    residual = (target_full - output).detach().view(B, self.feats, self.pred_len)
                    res_stats.update(residual)  # keep signals warm for phase continuity
                    loss = _rmse(output, target_full)
                    loss.backward()
                    self.model_optimizer.step()
                    avg_loss += loss.item()
                    n_batches += 1
                    continue

                # ── main phase: RW-1 correction + CEGAR gate ──────────────
                x_corr = correction[0, :, xb_idx].permute(1, 0, 2)
                target_corr = correction[0, :, yb_idx].permute(1, 0, 2)

                output = self.model(x + x_corr).view(-1, self.feats * self.pred_len)
                target_full = (target + target_corr).reshape(-1, self.feats * self.pred_len)

                B = output.shape[0]
                residual = (target_full - output).detach().view(B, self.feats, self.pred_len)
                window_resid = res_stats.update(residual)                 # [B]
                E_t, confidence = self._compute_signals(window_resid, res_stats, x + x_corr)  # [B], [B]

                # canonical gate() -> per-window scale + control stats
                scale, stats = compute_gate(
                    confidence=confidence, wrongness=E_t, lam=self.lam,
                    scale_normalize=self.scale_normalize, detach_gates=True,
                    minimal_stats=True)
                g_win = (E_t * confidence).clamp(0.0, 1.0).detach()  # [B] per-window gate
                gate_sum += float(g_win.mean().item())
                if epoch == self.epochs:  # record per-timestep gate in the final epoch
                    tgt = np.asarray(yb_idx).reshape(-1)
                    np.add.at(gate_accum, tgt, np.repeat(g_win.cpu().numpy(), self.pred_len))
                    np.add.at(gate_count, tgt, 1.0)

                # ── gate applied to the per-window RMSE GRADIENT (ScaleGrad),
                #    not the loss value: reported loss stays the true RMSE. ──
                per_window_rmse = torch.sqrt(
                    torch.nn.functional.mse_loss(output, target_full, reduction="none").mean(dim=1))
                loss = ScaleGrad.apply(per_window_rmse, scale).mean()
                loss.backward()  # correction grad of gated windows is up-weighted

                self.model_optimizer.step()
                avg_loss += loss.item()
                n_batches += 1

                # accumulate controller inputs
                self._gate_mean_ema = update_ema(self._gate_mean_ema, stats["gate_mean"], self.lam_ctrl_ema)
                self._gate_p99_ema = update_ema(self._gate_p99_ema, stats["gate_p99"], self.lam_ctrl_ema)
                ch = stats.get("_conf_hist")
                if ch is not None:
                    if conf_hist_accum is None:
                        conf_hist_accum = list(ch)
                    else:
                        conf_hist_accum = [a + b for a, b in zip(conf_hist_accum, ch)]

            if not warm:
                # ── L1 outlier penalty + epoch-wise correction step (RW-1) ──
                l1_loss = self.l1_weight * torch.norm(correction, p=1)
                l1_loss.backward()
                if correction.grad is not None:
                    correction.grad = self._grad_activation(correction.grad)
                    self.correction_optimizer.step()

                # ── controllers (epoch end); state carried to next epoch ──
                if self.lam_mode == "auto_tr":
                    self._lam_smooth, _ = tail_ratio_lambda_controller(
                        self._gate_mean_ema, self._gate_p99_ema,
                        ratio_target=self.ratio_target, lam_max=self.lam_max,
                        prev_lam_smooth=self._lam_smooth, lam_ema=self.lam_ctrl_ema)
                    self.lam = self._lam_smooth
                if self.tau_mode == "auto_q_valley" and conf_hist_accum is not None:
                    self._tau_q, info = valley_quantile_controller(
                        conf_hist_accum, prev_q=self._tau_q, ema_beta=self.tau_ctrl_ema)
                    if info.get("valley_bin") is not None:
                        # map the chosen quantile back to a residual threshold
                        self.tau = res_stats.quantile(self._tau_q)
            else:
                l1_loss = torch.tensor(0.0)

            avg_loss /= max(n_batches, 1)
            phase = "warmup" if warm else "main"
            metrics = {
                "epoch": epoch,
                "phase": phase,
                "loss": float(avg_loss),
                "l1": float(l1_loss),
                "gate_mean": float(gate_sum / max(n_batches, 1)),
                "lam": float(self.lam),
                "tau": float(self.tau),
                "q95": float(res_stats.q95),
            }
            if self.on_epoch_end is not None:
                self.on_epoch_end(metrics)  # e.g. stream to wandb every epoch
            if epoch == 1 or epoch % 20 == 0 or epoch == self.epochs or epoch == self.warmup_epochs:
                print(f"Epoch [{epoch}/{self.epochs}] ({phase}) | Loss: {avg_loss:.4f} "
                      f"| L1: {float(l1_loss):.4f} | gate: {gate_sum/max(n_batches,1):.4f} "
                      f"| lam: {self.lam:.3f} | tau: {self.tau:.3f} | q95: {res_stats.q95:.4f}")

        scores = np.abs(correction.detach().cpu().numpy()[0, :, :]).mean(axis=0)

        # interpretability outputs (aligned per timestep):
        #   correction_per_t = how much each point was changed (== score)
        #   gate_per_t       = mean gate activation on windows targeting that point
        self.correction_per_t = scores
        self.gate_per_t = gate_accum / np.maximum(gate_count, 1.0)
        # full correction tensor [feats, T] for original-vs-corrected example plots
        self.correction_full = correction.detach().cpu().numpy()[0]
        return scores
