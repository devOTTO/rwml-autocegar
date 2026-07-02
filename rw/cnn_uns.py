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


class CNN_uns():
    """RW method (paper Section 4.1 / Algorithm 1): the RW-LSTM training
    strategy applied directly to a DeepAnt-style CNN forecaster, with the
    LSTM replaced by CNNModel. Backpropagated gradients are used to
    directly correct the input data (no separate correction tensor).

    Update strategy is Epoch-Wise: gradients w.r.t. the input accumulate
    over a full epoch and are applied once at the epoch's end via RMSprop,
    matching the best configuration reported in the paper (Table 6.5).
    """

    def __init__(self,
                 window_size=50,
                 pred_len=1,
                 batch_size=256,
                 epochs=200,
                 lr=0.0008,
                 correction_rate=0.1,
                 activation='linear',
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

        self.model = CNNModel(n_features=feats, num_channel=num_channel, predict_time_steps=self.pred_len, device=self.device).to(self.device)

        self.model_optimizer = None
        self.input_optimizer = None
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

    def fit(self, data):
        print("Training CNN_uns (RW, Algorithm 1) model...")
        ts = self._normalize(data)
        ts = torch.from_numpy(ts).float().permute(1, 0).unsqueeze(0).to(self.device).contiguous()  # 1, feat, LEN

        X = ts.clone().detach().requires_grad_(True)
        X_original = ts.clone().detach()

        total_length = X.shape[2]
        X_indices, Y_indices = self.create_sliding_window(total_length, self.window_size, self.pred_len, shuffle=True)

        self.model_optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
        self.input_optimizer = optim.RMSprop([X], lr=self.correction_rate, alpha=0.99, eps=1e-8)

        for epoch in range(1, self.epochs + 1):
            self.model.train(mode=True)
            avg_loss = 0
            n_batches = 0

            self.input_optimizer.zero_grad()

            for i in range(0, X_indices.shape[0], self.batch_size):
                xb_idx = X_indices[i:i + self.batch_size]
                yb_idx = Y_indices[i:i + self.batch_size]

                x_batch = X[0, :, xb_idx].permute(1, 0, 2)       # B, feat, W
                target_batch = X[0, :, yb_idx].permute(1, 0, 2)  # B, feat, P

                self.model_optimizer.zero_grad()

                output = self.model(x_batch)
                output = output.view(-1, self.feats * self.pred_len)
                target = target_batch.reshape(-1, self.feats * self.pred_len)

                loss = _rmse(output, target)
                loss.backward()  # accumulates gradient on X across the whole epoch

                self.model_optimizer.step()

                avg_loss += loss.item()
                n_batches += 1

            # Epoch-Wise correction: apply the accumulated, activated gradient once
            if X.grad is not None:
                X.grad = self._grad_activation(X.grad)
                self.input_optimizer.step()

            avg_loss /= max(n_batches, 1)
            print(f"Epoch [{epoch}/{self.epochs}] | Loss: {avg_loss:.4f}")

        correction = (X.detach() - X_original).cpu().numpy()[0, :, :]
        scores = np.abs(correction).mean(axis=0)

        return scores

    def anomaly_score(self) -> np.ndarray:
        return self.__anomaly_score

    def param_statistic(self, save_file):
        model_stats = torchinfo.summary(self.model, (self.batch_size, self.window_size), verbose=0)
        with open(save_file, 'w') as f:
            f.write(str(model_stats))
