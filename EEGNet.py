import torch
import torch.nn as nn
from torch.nn import functional as F
from alignment_techniques import LatentAlignment2d, AdaptiveBatchNorm2d, EuclideanAlignment


def max_norm_(module, c=4., return_module=False):
    """Applies a max-norm constraint on the weight of the passed module.
    Clamps the norm of the weight vector to the specified value if it exceeds the limit.
    The constraint is applied in-place on the module.

    Use like this in the forward pass:
        def forward(self, x):
            x = max_norm_(self.layer1, c=4., return_module=True)(x)
    Args:
         module: A nn.Module instance, e.g. nn.Conv1d or nn.Linear
         c: The maximum constraint on the weight.
         return_module: Specify whether the module should be returned for convenience.
    """
    norms = module.weight.data.norm(dim=None, keepdim=True)
    desired = torch.clamp(norms, 0., c)
    module.weight.data = module.weight.data * (desired / (norms + 1e-6))
    if return_module:
        return module


class EEGNet(nn.Module):
    """
    EEGNet implementation based on
    Lawhern, V.J., Solon, A.J., Waytowich, N.R., Gordon, S.M., Hung, C.P. and Lance, B.J., 2018.
    EEGNet: a compact convolutional neural network for EEG-based brain–computer interfaces.
    Journal of neural engineering, 15(5), p.056013.
    https://iopscience.iop.org/article/10.1088/1741-2552/aace8c/meta.
    Assumes sampling frequency of 160 for kernel size choices. Uses odd kernel sizes for valid paddings.
    """

    def __init__(self, in_shape, n_out, alignment='None', dropout=0.25):
        super(EEGNet, self).__init__()
        self.in_shape = in_shape
        self.n_out = n_out
        self.alignment = alignment

        n_filters = 8
        n_spatial = 2
        self.n_filters = n_filters
        self.n_spatial = n_spatial

        if alignment == 'euclidean':
            self.euclidean = EuclideanAlignment()
        # Input bn
        if alignment == 'latent':
            self.latent_align0 = LatentAlignment2d(in_shape[0], affine=False)
        elif alignment == 'adaptive':
            self.abn0 = AdaptiveBatchNorm2d(in_shape[0], affine=False)
        else:
            self.bn0 = nn.BatchNorm2d(in_shape[0], affine=False)

        # Temporal filters: kernel 1/2 of sampling freq + 1
        self.conv1 = nn.Conv2d(1, n_filters,
                               kernel_size=(1, 81), padding=(0, 40),
                               bias=False, groups=1)
        if alignment == 'latent':
            self.latent_align1 = LatentAlignment2d(n_filters)
        elif alignment == 'adaptive':
            self.abn1 = AdaptiveBatchNorm2d(n_filters)
        else:
            self.bn1 = nn.BatchNorm2d(n_filters)

        # Spatial filters
        self.conv2 = nn.Conv2d(n_filters, n_spatial * n_filters,
                               kernel_size=(in_shape[0], 1),
                               bias=False, groups=n_filters)
        if alignment == 'latent':
            self.latent_align2 = LatentAlignment2d(n_spatial * n_filters)
        elif alignment == 'adaptive':
            self.abn2 = AdaptiveBatchNorm2d(n_spatial * n_filters)
        else:
            self.bn2 = nn.BatchNorm2d(n_spatial * n_filters)
        self.pool1 = nn.AvgPool2d(kernel_size=(1, 4))
        self.drop1 = nn.Dropout(dropout)

        # Separable convolution: kernel 1/8 of sampling freq + 1
        self.conv3 = nn.Conv2d(n_spatial * n_filters, n_spatial * n_filters,
                               kernel_size=(1, 21), padding=(0, 10),
                               bias=False, groups=n_spatial * n_filters)
        self.conv4 = nn.Conv2d(n_filters * n_spatial, n_filters * n_spatial,
                               kernel_size=(1, 1),
                               bias=False, groups=1)
        if alignment == 'latent':
            self.latent_align3 = LatentAlignment2d(self.conv4.out_channels)
        elif alignment == 'adaptive':
            self.abn3 = AdaptiveBatchNorm2d(self.conv4.out_channels)
        else:
            self.bn3 = nn.BatchNorm2d(self.conv4.out_channels)
        self.pool2 = nn.AvgPool2d(kernel_size=(1, 8))
        self.drop2 = nn.Dropout(dropout)

        # Classifier
        self.n_features = int(n_spatial * n_filters * (in_shape[1] // 4 // 8))
        self.fc_out = nn.Linear(self.n_features, n_out)

    def forward(self, x, sbj_trials):
        """
        Args:
             x: (batch * sbj_trials, spatial, time)
             sbj_trials: number of trials per subject
        """
        _, spatial, time = x.shape

        # Euclidean alignment
        if self.alignment == 'euclidean':
            x = self.euclidean(x, sbj_trials)

        # Input batchnorm
        x = x.reshape(-1, spatial, 1, time)
        if self.alignment == 'latent':
            x = self.latent_align0(x, sbj_trials)
        elif self.alignment == 'adaptive':
            x = self.abn0(x, sbj_trials)
        else:
            x = self.bn0(x)
        x = x.reshape(-1, spatial, time)

        # Add artificial image channel dimension for Conv2d
        x = x.unsqueeze(1)

        # Temporal filters
        x = self.conv1(x)
        if self.alignment == 'latent':
            x = self.latent_align1(x, sbj_trials)
        elif self.alignment == 'adaptive':
            x = self.abn1(x, sbj_trials)
        else:
            x = self.bn1(x)

        # Spatial filters
        x = max_norm_(self.conv2, c=1., return_module=True)(x)
        if self.alignment == 'latent':
            x = self.latent_align2(x, sbj_trials)
        elif self.alignment == 'adaptive':
            x = self.abn2(x, sbj_trials)
        else:
            x = self.bn2(x)
        x = F.elu(x)
        x = self.pool1(x)
        x = self.drop1(x)

        # Block 2
        x = self.conv3(x)
        x = self.conv4(x)
        if self.alignment == 'latent':
            x = self.latent_align3(x, sbj_trials)
        elif self.alignment == 'adaptive':
            x = self.abn3(x, sbj_trials)
        else:
            x = self.bn3(x)
        x = F.elu(x)
        x = self.pool2(x)
        x = self.drop2(x)

        # Classifier
        x = x.reshape(-1, self.n_features)
        x = max_norm_(self.fc_out, c=0.25, return_module=True)(x)

        return x
