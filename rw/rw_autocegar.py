"""RW-AutoCEGAR trainer — RW-2 + CEGAR gate on the correction update.

Core idea
---------
In vanilla RW-2 every window gets the same correction gradient magnitude.
Here we scale the per-window prediction loss by the CEGAR gate before
backprop, so the correction is updated *more aggressively* for windows
that the model finds anomalous::

    gate   = E_t × C_t
    loss_w = RMSE_per_window × (1 + λ·gate) + α·||correction||_1
    loss_w.backward()  → correction.grad already has gate-scaled signal

Signal mapping (residual domain):
    E_t (wrongness) = clamp(|residual| / residual_q95, max=1.0)
         ↕ replaces 1 - p_y
    C_t (confidence) = sigmoid(k·(τ - residual_ema))
         ↕ replaces sigmoid(k·(conf - τ))

    gate = E_t × C_t
      → high when: prediction error is large (E_t↑) AND
                   model has historically been accurate (C_t↑, low ema)
      → this is exactly "confident model, suddenly wrong = anomaly"

Everything above the gate computation is unchanged RW-2 (same algorithm,
same loss formula, same Savitzky-Golay smoothing, same anomaly score).

NOTE: E_t / C_t formula constants (τ, k) are PLACEHOLDER values pending
Luis's notebook (same as ``cegar/residual_signals.py``).
"""
import numpy as np
import torch
import torch.nn.functional as F
from torch.optim import Adam, RMSprop
from torch.utils.data import DataLoader, TensorDataset

from deepant.helper import retrieve_save_path
from cegar.residual_signals import ResidualStats, compute_wrongness_E_t, compute_confidence_C_t
from .rw2 import RW2Trainer


class RWCegarTrainer(RW2Trainer):
    """RW-2 with CEGAR gate applied to the per-window correction gradient.

    Extra args vs ``RW2Trainer``:
        lam: gate strength λ.
        k: sigmoid sharpness for C_t (PLACEHOLDER).
        tau: residual threshold for C_t (PLACEHOLDER).
        residual_ema_beta: EMA factor for the residual_ema stat.
        residual_buffer_size: rolling buffer size for q95 estimation.
    """

    def __init__(self, deepant_cfg, in_channels: int,
                 alpha: float = 1e-2,
                 correction_lr: float = 1e-2,
                 rmsprop_alpha: float = 0.9,
                 savgol_window: int = 11,
                 savgol_polyorder: int = 3,
                 # CEGAR params ──────────────────────────────────────────
                 lam: float = 1.0,
                 k: float = 1.0,        # PLACEHOLDER – update from notebook
                 tau: float = 0.5,      # PLACEHOLDER – update from notebook
                 residual_ema_beta: float = 0.9,
                 residual_buffer_size: int = 5000,
                 # ────────────────────────────────────────────────────────
                 wandb_run=None,
                 num_workers: int = 0):
        super().__init__(deepant_cfg, in_channels, alpha, correction_lr,
                         rmsprop_alpha, savgol_window, savgol_polyorder,
                         wandb_run, num_workers)
        self.lam = lam
        self.k = k
        self.tau = tau
        self.res_stats = ResidualStats(in_channels, ema_beta=residual_ema_beta,
                                       buffer_size=residual_buffer_size)

    # override train to inject CEGAR gate
    def train(self, X: np.ndarray, save_path: str):
        """Train with RW-2 + CEGAR-guided correction update."""
        model_save = retrieve_save_path(save_path, "model.pt")

        X_t = torch.from_numpy(X).float().to(self.device)

        self.correction = torch.nn.Parameter(-X_t.clone())
        self.res_stats = ResidualStats(self.in_channels,
                                       ema_beta=self.res_stats.ema_beta,
                                       buffer_size=self.res_stats.buffer_size)

        opt_model = Adam(self.model.parameters(), lr=self.dcfg.lr)
        opt_corr = RMSprop([self.correction], lr=self.correction_lr,
                           alpha=self.rmsprop_alpha, eps=1e-8)

        best_loss = float("inf")

        for epoch in range(self.dcfg.epochs):
            self.model.train()

            X_corr = X_t + self.correction
            xs, ys = self._make_windows(X_corr)

            loader = DataLoader(
                TensorDataset(torch.arange(len(xs))),
                batch_size=self.dcfg.batch_size,
                shuffle=True,
            )

            opt_model.zero_grad()
            opt_corr.zero_grad()

            epoch_loss = 0.0
            n_batches = 0
            gate_mean_epoch = 0.0

            for (idx,) in loader:
                x_b = xs[idx]   # [B, C, W]
                y_b = ys[idx]   # [B, C, P]

                y_hat = self.model(x_b)

                # ── per-window residual (no grad to model output needed here)
                residual = (y_b - y_hat).detach()          # [B, C, P]
                window_resid = self.res_stats.update(residual)   # [B]

                # ── E_t, C_t  (PLACEHOLDER formulas)
                E_t = compute_wrongness_E_t(window_resid, self.res_stats.q95)  # [B]
                C_t_scalar = compute_confidence_C_t(
                    self.res_stats.ema, self.tau, self.k)
                C_t = torch.full_like(E_t, C_t_scalar)           # [B]

                gate = (E_t * C_t).clamp(0.0, 1.0)               # [B]
                gate_scale = 1.0 + self.lam * gate                # [B]
                gate_mean_epoch += gate.mean().item()

                # ── scaled per-window prediction loss
                per_window_rmse = torch.sqrt(
                    F.mse_loss(y_hat, y_b, reduction="none").mean(dim=[1, 2])
                )                                                  # [B]
                pred_loss = (per_window_rmse * gate_scale.detach()).mean()

                reg_loss = self.alpha * self.correction.abs().mean()
                loss = pred_loss + reg_loss
                loss.backward(retain_graph=True)
                epoch_loss += loss.item()
                n_batches += 1

            opt_model.step()
            opt_corr.step()

            # ── Savitzky-Golay smoothing (same as RW-2) ──────────────────
            self._smooth_correction()

            avg_loss = epoch_loss / max(n_batches, 1)
            corr_mag = self.correction.detach().abs().mean().item()
            avg_gate = gate_mean_epoch / max(n_batches, 1)

            if avg_loss < best_loss:
                best_loss = avg_loss
                torch.save(self.model.state_dict(), model_save)

            if epoch % max(1, self.dcfg.epochs // 20) == 0:
                print(f"[RW-CEGAR] epoch {epoch:>4}/{self.dcfg.epochs}  "
                      f"loss={avg_loss:.5f}  |corr|={corr_mag:.5f}  "
                      f"gate={avg_gate:.4f}  q95={self.res_stats.q95:.4f}")

            self._step_wandb({
                "RWCegar/loss": avg_loss,
                "RWCegar/correction_mag": corr_mag,
                "RWCegar/gate_mean": avg_gate,
                "RWCegar/residual_q95": self.res_stats.q95,
                "RWCegar/residual_ema": self.res_stats.ema,
            }, epoch)

        print(f"[RW-CEGAR] done. Model → {model_save}")
