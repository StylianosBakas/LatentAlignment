"""
* Copyright (C) Cogitat, Ltd.
* Creative Commons Attribution-NonCommercial 4.0 International (CC BY-NC 4.0)
"""
import torch
import torch.nn as nn
from torch import linalg


class LatentAlignment2d(nn.Module):
    def __init__(self, n_channels, affine=True):
        super(LatentAlignment2d, self).__init__()
        self.n_channels = n_channels
        self.affine = affine

        if self.affine:
            self.weight = nn.Parameter(torch.ones(n_channels))
            self.bias = nn.Parameter(torch.zeros(n_channels))
        else:
            self.weight = None
            self.bias = None

    def forward(self, x, sbj_trials):
        """
        Args:
            x: (batch * sbj_trials, channels, spatial, time)
            sbj_trials: number of trials per subject
        """
        _, channels, spatial, time = x.shape

        # Get sbj_trials out of the batch
        x = x.reshape(-1, sbj_trials, channels, spatial, time)
        batch = x.shape[0]

        # Standardize across all subject trials
        x = (x - x.mean(dim=[-4, -2, -1], keepdim=True)) / torch.sqrt(x.var(dim=[-4, -2, -1], keepdim=True) + 1e-8)

        # Apply trainable weight and bias if affine
        if self.affine:
            x = x * self.weight.reshape(-1, 1, 1) + self.bias.reshape(-1, 1, 1)

        # Move sbj_trials back in the batch
        x = x.reshape(batch * sbj_trials, channels, spatial, time)

        return x


class AdaptiveBatchNorm2d(nn.Module):
    """
    Adaptive batchnorm implementation based on
    Li, Y., Wang, N., Shi, J., Hou, X. and Liu, J., 2018.
    Adaptive batch normalization for practical domain adaptation. Pattern Recognition, 80, pp.109-117
    https://www.sciencedirect.com/science/article/abs/pii/S003132031830092X
    """

    def __init__(self, n_channels, affine=True):
        super(AdaptiveBatchNorm2d, self).__init__()
        self.n_channels = n_channels
        self.affine = affine
        self.bn = nn.BatchNorm2d(n_channels, affine=affine)

    def forward(self, x, sbj_trials):
        """
        Args:
            x: (batch * sbj_trials, channels, spatial, time)
            sbj_trials: number of trials per subject
        """
        _, channels, spatial, time = x.shape

        # Standard batchnorm during training
        if self.training:
            x = self.bn(x)
        # Standardize across all subject trials during inference
        else:
            # Get sbj_trials out of the batch
            x = x.reshape(-1, sbj_trials, channels, spatial, time)
            # Standardize across all subject trials
            x = (x - x.mean(dim=[-4, -2, -1], keepdim=True)) / torch.sqrt(x.var(dim=[-4, -2, -1], keepdim=True) + 1e-8)
            # Move sbj_trials back in the batch
            x = x.reshape(-1, channels, spatial, time)
            # Apply trainable weight and bias if affine
            if self.affine:
                x = self.bn.weight.data.reshape(-1, 1, 1) * x + self.bn.bias.data.reshape(-1, 1, 1)

        return x


class EuclideanAlignment(nn.Module):
    """
    Euclidean alignment implementation based on
    He, H. and Wu, D., 2019.
    Transfer learning for brain–computer interfaces: A Euclidean space data alignment approach.
    IEEE Transactions on Biomedical Engineering, 67(2), pp.399-410.
    https://ieeexplore.ieee.org/abstract/document/8701679
    """

    def __init__(self):
        super(EuclideanAlignment, self).__init__()

    def forward(self, x, sbj_trials):
        """
        Args:
            x: (batch * sbj_trials, spatial, time)
            sbj_trials: number of trials per subject
        """
        _, spatial, time = x.shape

        # Get sbj_trials out of the batch
        x = x.reshape(-1, sbj_trials, spatial, time)

        # Recenter per electrode and rescale per trial
        x = (x - x.mean(dim=(-1), keepdim=True)) / x.std(dim=(-2, -1), keepdim=True)
        # Calculate and condition the covariance matrix
        cov = torch.matmul(x, x.transpose(-2, -1)) / (x.shape[-1] - 1)
        cond = torch.eye(cov.shape[-1], device=cov.device) * 1e-4
        cov = cov + cond
        # Whiten based on the mean covariance per subject
        cov = cov.mean(dim=1, keepdim=True)
        cov = linalg.inv(linalg.cholesky(cov)).float()
        x = torch.matmul(cov, x)
        # Move sbj_trials back in the batch
        x = x.reshape(-1, spatial, time)

        return x
