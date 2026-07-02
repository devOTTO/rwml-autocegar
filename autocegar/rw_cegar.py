"""RW-1 + CEGAR — the AutoCEGAR trainer built on the *reproduction-successful*
RW-1 (paper Algorithm 2) rather than the old standalone DeepAnT track.

`CNN_RW_CEGAR` subclasses the reproduced `rw.cnn_rw.CNN_RW` (linear gradient,
weak L1, epoch-wise correction update — the config that reproduces the paper)
and injects the CEGAR gate into the per-window predictive loss:

    gate       = clamp(E_t * C_t, 0, 1)          # wrong AND historically-confident
    gate_scale = 1 + lam * gate
    pred_loss  = mean( per_window_RMSE * gate_scale )

So windows the model gets wrong *while it has been accurate overall* (i.e.
likely true anomalies, not model noise) are up-weighted, steering the
`correction` tensor toward them. Everything else (init = -X, L1 sparsity,
linear-activated RMSprop correction step, |correction| anomaly score) is
inherited unchanged from the reproduced RW-1.

CEGAR signal formulas (E_t, C_t) live in `autocegar.residual_signals` and are
still the placeholder forms pending Luis's notebook.
"""
import numpy as np
import torch
from torch import optim

from rw.cnn_rw import CNN_RW, _rmse  # reproduced RW-1 (Algorithm 2)
from autocegar.residual_signals import (
    ResidualStats, compute_wrongness_E_t, compute_confidence_C_t,
)


class CNN_RW_CEGAR(CNN_RW):
    """RW-1 with a CEGAR gate on the predictive loss.

    Extra hyperparameters beyond CNN_RW:
        lam         gate strength (0 => plain RW-1).
        tau, k      C_t sigmoid threshold / sharpness.
        ema_beta    residual EMA smoothing for C_t.
        buffer_size rolling buffer for the q95 used by E_t.
    """

    def __init__(self, *args, lam=1.0, tau=0.5, k=10.0,
                 ema_beta=0.9, buffer_size=5000, **kwargs):
        super().__init__(*args, **kwargs)
        self.lam = lam
        self.tau = tau
        self.k = k
        self.ema_beta = ema_beta
        self.buffer_size = buffer_size

    def fit(self, data, train_idx=None):
        print("Training CNN_RW_CEGAR (RW-1 + CEGAR gate)...")
        ts = self._normalize(data)
        ts = torch.from_numpy(ts).float().permute(1, 0).unsqueeze(0).to(self.device).contiguous()

        correction = (-ts.clone()).detach().requires_grad_(True)
        total_length = ts.shape[2]
        X_indices, Y_indices = self.create_sliding_window(
            total_length, self.window_size, self.pred_len, shuffle=True)

        self.model_optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
        self.correction_optimizer = optim.RMSprop(
            [correction], lr=self.correction_rate, alpha=0.99, eps=1e-8)

        res_stats = ResidualStats(self.feats, ema_beta=self.ema_beta,
                                  buffer_size=self.buffer_size)

        for epoch in range(1, self.epochs + 1):
            self.model.train(mode=True)
            avg_loss, n_batches, gate_sum = 0.0, 0, 0.0
            self.correction_optimizer.zero_grad()

            for i in range(0, X_indices.shape[0], self.batch_size):
                xb_idx = X_indices[i:i + self.batch_size]
                yb_idx = Y_indices[i:i + self.batch_size]

                x = ts[0, :, xb_idx].permute(1, 0, 2)
                target = ts[0, :, yb_idx].permute(1, 0, 2)
                x_corr = correction[0, :, xb_idx].permute(1, 0, 2)
                target_corr = correction[0, :, yb_idx].permute(1, 0, 2)

                self.model_optimizer.zero_grad()

                output = self.model(x + x_corr).view(-1, self.feats * self.pred_len)
                target_full = (target + target_corr).reshape(-1, self.feats * self.pred_len)

                # ── CEGAR gate from the per-window residual ──────────────
                B = output.shape[0]
                residual = (target_full - output).detach().view(B, self.feats, self.pred_len)
                window_resid = res_stats.update(residual)                 # [B]
                E_t = compute_wrongness_E_t(window_resid, res_stats.q95)   # [B]
                C_t = compute_confidence_C_t(res_stats.ema, self.tau, self.k)
                gate = (E_t * C_t).clamp(0.0, 1.0)                         # [B]
                gate_scale = (1.0 + self.lam * gate).detach()             # [B]
                gate_sum += float(gate.mean().item())

                # ── gated per-window RMSE (RW-1's RMSE, up-weighted) ─────
                per_window_rmse = torch.sqrt(
                    torch.nn.functional.mse_loss(output, target_full, reduction="none").mean(dim=1))
                loss = (per_window_rmse * gate_scale).mean()
                loss.backward()  # accumulates gradient on `correction`

                self.model_optimizer.step()
                avg_loss += loss.item()
                n_batches += 1

            # ── L1 outlier penalty + epoch-wise correction step (RW-1) ───
            l1_loss = self.l1_weight * torch.norm(correction, p=1)
            l1_loss.backward()
            if correction.grad is not None:
                correction.grad = self._grad_activation(correction.grad)
                self.correction_optimizer.step()

            avg_loss /= max(n_batches, 1)
            if epoch == 1 or epoch % 20 == 0 or epoch == self.epochs:
                print(f"Epoch [{epoch}/{self.epochs}] | Loss: {avg_loss:.4f} "
                      f"| L1: {l1_loss.item():.4f} | gate: {gate_sum/max(n_batches,1):.4f} "
                      f"| q95: {res_stats.q95:.4f}")

        scores = np.abs(correction.detach().cpu().numpy()[0, :, :]).mean(axis=0)
        return scores
