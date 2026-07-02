from typing import Dict
import torchinfo
import numpy as np
import torch
from torch import nn, optim

from tsb_common.utility import get_activation_by_name
from tsb_common.torch_utility import get_gpu


class AdaptiveConcatPool1d(nn.Module):
    def __init__(self):
        super().__init__()
        self.ap = torch.nn.AdaptiveAvgPool1d(1)
        self.mp = torch.nn.AdaptiveAvgPool1d(1)

    def forward(self, x):
        return torch.cat([self.ap(x), self.mp(x)], 1)


class CNNModel(nn.Module):
    def __init__(self,
                 n_features,
                 num_channel=[32, 32, 40],
                 kernel_size=3,
                 stride=1,
                 predict_time_steps=1,
                 dropout_rate=0.25,
                 hidden_activation='relu',
                 device='cpu'):

        super(CNNModel, self).__init__()

        self.n_features = n_features
        self.dropout_rate = dropout_rate
        self.hidden_activation = hidden_activation
        self.kernel_size = kernel_size
        self.stride = stride
        self.predict_time_steps = predict_time_steps
        self.num_channel = num_channel
        self.device = device

        self.activation = get_activation_by_name(hidden_activation)

        self.conv_layers = nn.Sequential()
        prev_channels = self.n_features

        for idx, out_channels in enumerate(self.num_channel[:-1]):
            self.conv_layers.add_module(
                "conv" + str(idx),
                torch.nn.Conv1d(prev_channels, self.num_channel[idx + 1],
                self.kernel_size, self.stride))
            self.conv_layers.add_module(self.hidden_activation + str(idx),
                                    self.activation)
            self.conv_layers.add_module("pool" + str(idx), nn.MaxPool1d(kernel_size=2))
            prev_channels = out_channels

        self.fc = nn.Sequential(
            AdaptiveConcatPool1d(),
            torch.nn.Flatten(),
            torch.nn.Linear(2*self.num_channel[-1], self.num_channel[-1]),
            torch.nn.ReLU(),
            torch.nn.Dropout(dropout_rate),
            torch.nn.Linear(self.num_channel[-1], self.n_features)
        )

    def forward(self, x):
        b, c, l = x.shape
        x = self.conv_layers(x)

        outputs = torch.zeros(self.predict_time_steps, b, self.n_features).to(self.device)
        for t in range(self.predict_time_steps):
            decoder_input = self.fc(x)
            outputs[t] = torch.squeeze(decoder_input, dim=-2)

        return outputs


def _rmse(pred, target):
    return torch.sqrt(nn.functional.mse_loss(pred, target))


class CNN_RW():
    """RW-1 (paper Section 4.2 / Algorithm 2): extends RW with a separate
    trainable correction tensor (initialized as -X) added to the input
    before it is fed to the model. The model learns to predict the
    corrected series; the correction is regularized with an L1 (outlier)
    penalty to encourage sparsity.

    Update strategy is Epoch-Wise: gradients w.r.t. the correction tensor
    accumulate over a full epoch (predictive RMSE term per batch, L1 term
    once per epoch) and are applied once at the epoch's end via RMSprop.

    Gradient activation: Algorithm 2 (paper Section 4.2, p.32) applies NO
    activation to the correction gradient — the raw combined gradient
    (RMSE + L1) goes straight into RMSprop. We therefore default to
    'linear' (identity) to reproduce the pseudocode faithfully. The earlier
    'relu' default (chosen from Table 6.5's "Gradients Function = ReLU"
    best-HP row) is NOT in the algorithm and interacts badly with the L1
    term: since correction is initialized to -X (mostly negative), the L1
    gradient sign(correction) is ~-1 everywhere, and relu() zeroes the
    combined gradient, freezing the correction (grad_zero_frac -> 1.0) and
    inverting the score on datasets like Genesis/GECCO/GHL. See the exp_e
    README / diag_ghl_freeze diagnostic. 'relu'/'sigmoid' remain available
    for ablation.

    L1 weight (l1_weight): the outlier penalty is l1_weight * ||correction||_1.
    Algorithm 2 writes it as ||correction||_1 (implicit weight 1.0), but
    torch.norm sums over ALL elements, making the penalty far too strong: its
    sign(±1) gradient steadily crushes `correction` to 0 over the 200 epochs,
    reintroducing every point (including anomalies) so the anomaly signal
    collapses and the score inverts. An l1_weight of ~1e-3 keeps the sparsity
    pressure without collapse and reproduces the paper's per-dataset AUC-PR on
    the previously-failing sets (e.g. GECCO 0.61 vs paper 0.62, Genesis 0.030
    vs 0.032). See the L1-weight sweep in the exp_e README.
    """

    def __init__(self,
                 window_size=50,
                 pred_len=1,
                 batch_size=256,
                 epochs=200,
                 lr=0.0008,
                 correction_rate=0.1,
                 activation='linear',
                 l1_weight=0.001,
                 feats=1,
                 num_channel=[32, 32, 40],
                 validation_size=0.2):
        super().__init__()
        self.__anomaly_score = None

        self.cuda = True
        self.device = get_gpu(self.cuda)

        self.window_size = window_size
        self.pred_len = pred_len
        self.batch_size = batch_size
        self.epochs = epochs

        self.feats = feats
        self.num_channel = num_channel
        self.lr = lr
        self.correction_rate = correction_rate
        self.activation = activation
        self.l1_weight = l1_weight

        self.model = CNNModel(n_features=feats, num_channel=num_channel, predict_time_steps=self.pred_len, device=self.device).to(self.device)

        self.model_optimizer = None
        self.correction_optimizer = None
        self.save_path = None

    def _normalize(self, ts):
        mean, std = np.mean(ts, axis=0), np.std(ts, axis=0)
        std = np.where(std == 0, 1e-8, std)
        return (ts - mean) / std

    def _grad_activation(self, grad):
        if self.activation == 'linear':
            return grad
        elif self.activation == 'relu':
            return torch.relu(grad)
        elif self.activation == 'sigmoid':
            return torch.sigmoid(grad)
        raise ValueError(f"Unknown RW activation function: {self.activation}")

    def create_sliding_window(self, total_length, window, pred_w, shuffle=True):
        W, P = window, pred_w
        num_samples = total_length - W - P + 1

        X_indices = np.array([np.arange(i, i + W) for i in range(num_samples)])
        Y_indices = np.array([np.arange(i + W, i + W + P) for i in range(num_samples)])

        if shuffle:
            perm = np.random.permutation(num_samples)
            X_indices = X_indices[perm]
            Y_indices = Y_indices[perm]
        return X_indices, Y_indices

    def fit(self, data, train_idx=None):
        print("Training CNN_RW (RW-1, Algorithm 2) model...")
        ts = self._normalize(data)
        ts = torch.from_numpy(ts).float().permute(1, 0).unsqueeze(0).to(self.device).contiguous()  # 1, feat, LEN

        correction = (-ts.clone()).detach().requires_grad_(True)

        total_length = ts.shape[2]
        X_indices, Y_indices = self.create_sliding_window(total_length, self.window_size, self.pred_len, shuffle=True)

        self.model_optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
        self.correction_optimizer = optim.RMSprop([correction], lr=self.correction_rate, alpha=0.99, eps=1e-8)

        for epoch in range(1, self.epochs + 1):
            self.model.train(mode=True)
            avg_loss = 0
            n_batches = 0

            self.correction_optimizer.zero_grad()

            for i in range(0, X_indices.shape[0], self.batch_size):
                xb_idx = X_indices[i:i + self.batch_size]
                yb_idx = Y_indices[i:i + self.batch_size]

                x = ts[0, :, xb_idx].permute(1, 0, 2)        # B, feat, W
                target = ts[0, :, yb_idx].permute(1, 0, 2)   # B, feat, P

                x_corr = correction[0, :, xb_idx].permute(1, 0, 2)
                target_corr = correction[0, :, yb_idx].permute(1, 0, 2)

                self.model_optimizer.zero_grad()

                output = self.model(x + x_corr)
                output = output.view(-1, self.feats * self.pred_len)
                target_full = (target + target_corr).reshape(-1, self.feats * self.pred_len)

                loss = _rmse(output, target_full)
                loss.backward()  # accumulates gradient on `correction` across the whole epoch

                self.model_optimizer.step()

                avg_loss += loss.item()
                n_batches += 1

            # Outlier loss: L1 sparsity penalty on the full correction tensor,
            # added once per epoch on top of the accumulated predictive gradient.
            l1_loss = self.l1_weight * torch.norm(correction, p=1)
            l1_loss.backward()

            if correction.grad is not None:
                correction.grad = self._grad_activation(correction.grad)
                self.correction_optimizer.step()

            avg_loss /= max(n_batches, 1)
            print(f"Epoch [{epoch}/{self.epochs}] | Loss: {avg_loss:.4f} | L1: {l1_loss.item():.4f}")

        scores = np.abs(correction.detach().cpu().numpy()[0, :, :]).mean(axis=0)

        return scores

    def anomaly_score(self) -> np.ndarray:
        return self.__anomaly_score

    def param_statistic(self, save_file):
        model_stats = torchinfo.summary(self.model, (self.batch_size, self.window_size), verbose=0)
        with open(save_file, 'w') as f:
            f.write(str(model_stats))
