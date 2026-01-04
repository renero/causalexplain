#
# Implementations of loss functions for DNNs.
# Reject using the one coming from CDT, because it is not working properly.
# Reject using the one at https://github.com/ZongxianLee/MMD_Loss.Pytorch
# Accepted using https://github.com/KevinMusgrave/pytorch-adapt
# Also valid: https://github.com/yiftachbeer/mmd_loss_pytorch/
#
# I keep the last one for simplicity. Results are the same.
#
import torch
import torch.nn as nn


class RBF(nn.Module):

    def __init__(self, n_kernels=5, mul_factor=2.0, bandwidth=None):
        super().__init__()
        self.bandwidth_multipliers = mul_factor ** (
            torch.arange(n_kernels, dtype=torch.float32) - n_kernels // 2)
        self.bandwidth = bandwidth

    def get_bandwidth(self, L2_distances):
        """
        Get the bandwidth of the RBF kernel.
        """
        if self.bandwidth is None:
            n_samples = L2_distances.shape[0]
            return L2_distances.data.sum() / (n_samples ** 2 - n_samples)

        return self.bandwidth

    def forward(self, X):
        L2_distances = torch.cdist(X, X) ** 2
        return torch.exp(-L2_distances[None, ...] / (self.get_bandwidth(L2_distances) * self.bandwidth_multipliers)[:, None, None]).sum(dim=0)


class MMDLoss(nn.Module):

    def __init__(self, kernel=RBF()):
        super().__init__()
        self.kernel = kernel

    def forward(self, X, Y):
        K = self.kernel(torch.vstack([X, Y]))

        X_size = X.shape[0]
        XX = K[:X_size, :X_size].mean()
        XY = K[:X_size, X_size:].mean()
        YY = K[X_size:, X_size:].mean()
        return XX - 2 * XY + YY
