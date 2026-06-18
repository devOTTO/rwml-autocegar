"""Per-sample gradient rescaling.

Copied verbatim from ``autocegar/ecg_loss.py`` (lines 5-14). The forward pass is
the identity, so the loss value is unchanged; only the backward gradient flowing
into ``x`` is multiplied by ``scale``. This is the mechanism the original plan
referred to as the "RW write / gradient-application step" -- it is gradient
reweighting, not a rewrite of inputs or targets.
"""
import torch


class ScaleGrad(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, scale):
        ctx.save_for_backward(scale)
        return x

    @staticmethod
    def backward(ctx, grad_output):
        scale, = ctx.saved_tensors
        return grad_output * scale, None
