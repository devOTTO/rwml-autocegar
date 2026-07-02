#!/usr/bin/env python3
"""Diagnostic: why do RW / RW-1 score-INVERT on GHL (AUC-ROC << 0.5)?

Runs the algorithm-faithful RW-1 (CNN_RW) and/or RW (CNN_uns) fit loop on a
single GHL file, but instrumented per-epoch to observe where the inversion
comes from. Logs, for every epoch:

  * loss                    — predictive RMSE
  * |corr| mean             — overall mean magnitude of the correction
  * |corr| anom / normal    — mean |corr| in anomaly vs normal timesteps
  * sep = anom/normal ratio — >1 means correct direction, <1 means INVERTED
  * auc_roc                 — AUC-ROC of the current score vs labels
  * grad_zero_frac          — fraction of correction gradient that is exactly
                              0 after the activation (freeze indicator)

This reuses the shipped CNNModel but reimplements the fit loop inline so the
production model files are NOT modified. Config matches the real EXP-E run
(window=50, epochs=200, batch=256, lr=8e-4, correction_rate=0.1).

Run via the SLURM wrapper submit_diag_ghl.sh (needs a GPU).
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd
import torch
from torch import optim
from sklearn.metrics import roc_auc_score, average_precision_score

sys.path.insert(0, "/ocean/projects/cis260190p/yhwang2/TSB-AD")
from TSB_AD.models.CNN_RW import CNNModel, _rmse  # noqa: E402

DATA_DIR = "/ocean/projects/cis260190p/yhwang2/data/TSB-AD-M/TSB-AD-M/"


def normalize(ts):
    mean, std = np.mean(ts, axis=0), np.std(ts, axis=0)
    std = np.where(std == 0, 1e-8, std)
    return (ts - mean) / std


def sliding(total_length, W, P, shuffle=True):
    n = total_length - W - P + 1
    Xi = np.array([np.arange(i, i + W) for i in range(n)])
    Yi = np.array([np.arange(i + W, i + W + P) for i in range(n)])
    if shuffle:
        perm = np.random.permutation(n)
        Xi, Yi = Xi[perm], Yi[perm]
    return Xi, Yi


def grad_act(grad, activation):
    if activation == "linear":
        return grad
    if activation == "relu":
        return torch.relu(grad)
    if activation == "sigmoid":
        return torch.sigmoid(grad)
    raise ValueError(activation)


def run(method, filename, epochs, use_wandb, activation=None, l1_weight=1.0):
    # RW-1 => separate correction tensor + L1 (Algorithm 2, no grad activation)
    # RW   => correct the input in place, linear grad, no L1
    is_rw1 = method == "RW-1"
    if activation is None:
        # faithful defaults: Algorithm 2 has NO activation => linear
        activation = "linear"

    W, P, bs, lr, cr = 50, 1, 256, 8e-4, 0.1
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    df = pd.read_csv(os.path.join(DATA_DIR, filename)).dropna()
    data = df.iloc[:, 0:-1].values.astype(float)
    label = df["Label"].astype(int).to_numpy()
    feats = data.shape[1]
    print(f"[{method}] {filename}: data{data.shape}, "
          f"anomalies={label.sum()} ({100*label.mean():.2f}%)", flush=True)

    ts = normalize(data)
    ts = torch.from_numpy(ts).float().permute(1, 0).unsqueeze(0).to(dev).contiguous()
    L = ts.shape[2]

    model = CNNModel(n_features=feats, num_channel=[32, 32, 40],
                     predict_time_steps=P, device=dev).to(dev)

    if is_rw1:
        correction = (-ts.clone()).detach().requires_grad_(True)
        var = correction
    else:
        X = ts.clone().detach().requires_grad_(True)
        X_orig = ts.clone().detach()
        var = X

    Xi, Yi = sliding(L, W, P, shuffle=True)
    m_opt = optim.Adam(model.parameters(), lr=lr)
    c_opt = optim.RMSprop([var], lr=cr, alpha=0.99, eps=1e-8)

    run = None
    if use_wandb:
        import wandb
        run = wandb.init(
            entity=os.environ.get("WANDB_ENTITY", "yoonmeeh-cmu"),
            project=os.environ.get("WANDB_PROJECT", "rwml-autocegar"),
            group="diag-ghl-freeze",
            name=f"diag-{method}-{activation}-l1w{l1_weight}-{filename.split('_')[0]}",
            job_type="diagnostic",
            tags=["diagnostic", method, activation, f"l1w={l1_weight}"],
            config=dict(method=method, filename=filename, epochs=epochs,
                        window=W, batch=bs, lr=lr, correction_rate=cr,
                        activation=activation, l1_weight=l1_weight),
            reinit=True,
        )

    rows = []
    anom = label.astype(bool)
    for epoch in range(1, epochs + 1):
        model.train(True)
        c_opt.zero_grad()
        avg_loss, nb = 0.0, 0
        for i in range(0, Xi.shape[0], bs):
            xb, yb = Xi[i:i + bs], Yi[i:i + bs]
            if is_rw1:
                x = ts[0, :, xb].permute(1, 0, 2)
                tgt = ts[0, :, yb].permute(1, 0, 2)
                xc = correction[0, :, xb].permute(1, 0, 2)
                tc = correction[0, :, yb].permute(1, 0, 2)
                m_opt.zero_grad()
                out = model(x + xc).view(-1, feats * P)
                tgt_full = (tgt + tc).reshape(-1, feats * P)
            else:
                x = X[0, :, xb].permute(1, 0, 2)
                tgt = X[0, :, yb].permute(1, 0, 2)
                m_opt.zero_grad()
                out = model(x).view(-1, feats * P)
                tgt_full = tgt.reshape(-1, feats * P)
            loss = _rmse(out, tgt_full)
            loss.backward()
            m_opt.step()
            avg_loss += loss.item()
            nb += 1

        if is_rw1:
            l1 = l1_weight * torch.norm(correction, p=1)
            l1.backward()

        grad_zero_frac = float("nan")
        if var.grad is not None:
            g = grad_act(var.grad, activation)
            grad_zero_frac = float((g == 0).float().mean().item())
            var.grad = g
            c_opt.step()

        # current score
        if is_rw1:
            score = np.abs(correction.detach().cpu().numpy()[0]).mean(axis=0)
        else:
            score = np.abs((X.detach() - X_orig).cpu().numpy()[0]).mean(axis=0)

        cm = float(score.mean())
        a_mean = float(score[anom].mean()) if anom.any() else float("nan")
        n_mean = float(score[~anom].mean()) if (~anom).any() else float("nan")
        sep = a_mean / n_mean if n_mean else float("nan")
        try:
            auc = float(roc_auc_score(label, score))
            aucpr = float(average_precision_score(label, score))
        except Exception:
            auc = aucpr = float("nan")
        avg_loss /= max(nb, 1)

        rows.append(dict(epoch=epoch, loss=avg_loss, corr_mean=cm,
                         corr_anom=a_mean, corr_normal=n_mean, sep=sep,
                         auc_roc=auc, auc_pr=aucpr,
                         grad_zero_frac=grad_zero_frac))
        if run:
            run.log(rows[-1], step=epoch)
        if epoch == 1 or epoch % 10 == 0 or epoch == epochs:
            print(f"  ep{epoch:3d} loss={avg_loss:.4f} |corr|={cm:.4f} "
                  f"anom={a_mean:.4f} norm={n_mean:.4f} sep={sep:.3f} "
                  f"auc={auc:.3f} gz={grad_zero_frac:.3f}", flush=True)

    out_csv = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        f"diag_{method}_{activation}_l1w{l1_weight}_{filename.split('.')[0]}.csv")
    pd.DataFrame(rows).to_csv(out_csv, index=False)
    print(f"[{method}] wrote {out_csv}", flush=True)
    if run:
        best_pr = max(r["auc_pr"] for r in rows)
        run.summary.update(dict(final_auc=rows[-1]["auc_roc"],
                                final_auc_pr=rows[-1]["auc_pr"],
                                best_auc_pr=best_pr,
                                final_sep=rows[-1]["sep"]))
        run.finish()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", choices=["RW", "RW-1"], default="RW-1")
    ap.add_argument("--filename",
                    default="032_GHL_id_1_Sensor_tr_50000_1st_65001.csv")
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--activation", choices=["linear", "relu", "sigmoid"],
                    default=None, help="override grad activation")
    ap.add_argument("--l1_weight", type=float, default=1.0,
                    help="scale on the L1 outlier penalty (RW-1 only)")
    ap.add_argument("--wandb", action="store_true")
    a = ap.parse_args()
    run(a.method, a.filename, a.epochs, a.wandb, a.activation, a.l1_weight)
