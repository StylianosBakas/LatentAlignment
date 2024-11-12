import torch.nn as nn
from torch.nn import functional as F
from alignment_techniques import LatentAlignment2d, AdaptiveBatchNorm2d, EuclideanAlignment


class DeepSleep(nn.Module):
    """
    DeepSleep implementation  based on
    Chambon, S., Galtier, M.N., Arnal, P.J., Wainrib, G. and Gramfort, A., 2018.
    A deep learning architecture for temporal sleep stage classification using multivariate and multimodal time series.
    IEEE Transactions on Neural Systems and Rehabilitation Engineering, 26(4), pp.758-769.
    https://ieeexplore.ieee.org/document/8307462.
    """

    def __init__(self, in_shape, n_out, alignment='None', dropout=0.25):
        super(DeepSleep, self).__init__()
        self.in_shape = in_shape
        self.n_out = n_out
        self.alignment = alignment

        n_filters = 2
        n_spatial = 8
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

        # Spatial filters
        self.conv1 = nn.Conv2d(1, n_spatial,
                               kernel_size=(in_shape[0], 1),
                               bias=True)
        if alignment == 'latent':
            self.latent_align1 = LatentAlignment2d(self.conv1.out_channels)
        elif alignment == 'adaptive':
            self.abn1 = AdaptiveBatchNorm2d(self.conv1.out_channels)
        else:
            self.bn1 = nn.BatchNorm2d(self.conv1.out_channels)

        # Block 1
        self.conv2 = nn.Conv2d(n_spatial, n_spatial * n_filters,
                               kernel_size=(1, 51), padding=(0, 25),
                               bias=True, groups=1)
        if alignment == 'latent':
            self.latent_align2 = LatentAlignment2d(self.conv2.out_channels)
        elif alignment == 'adaptive':
            self.abn2 = AdaptiveBatchNorm2d(self.conv2.out_channels)
        else:
            self.bn2 = nn.BatchNorm2d(self.conv2.out_channels)
        self.pool1 = nn.MaxPool2d(kernel_size=(1, 16))

        # Block 2
        self.conv3 = nn.Conv2d(self.conv2.out_channels, self.conv2.out_channels,
                               kernel_size=(1, 51), padding=(0, 25),
                               bias=True, groups=1)
        if alignment == 'latent':
            self.latent_align3 = LatentAlignment2d(self.conv2.out_channels)
        elif alignment == 'adaptive':
            self.abn3 = AdaptiveBatchNorm2d(self.conv2.out_channels)
        else:
            self.bn3 = nn.BatchNorm2d(self.conv2.out_channels)
        self.pool2 = nn.MaxPool2d(kernel_size=(1, 16))
        self.drop1 = nn.Dropout(dropout)

        # Classifier
        self.n_features = int(self.conv3.out_channels) * (in_shape[-1] // 16 // 16)
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

        # Spatial
        x = self.conv1(x)  # (batch * sbj, spatial, 1, time)
        if self.alignment == 'latent':
            x = self.latent_align1(x, sbj_trials, growing_context=growing_context)
        elif self.alignment == 'adaptive':
            x = self.abn1(x, sbj_trials)
        else:
            x = self.bn1(x)

        # Block 1
        x = self.conv2(x)  # (batch * sbj, filters, spatial, time)
        if self.alignment == 'latent':
            x = self.latent_align2(x, sbj_trials)
        elif self.alignment == 'adaptive':
            x = self.abn2(x, sbj_trials)
        else:
            x = self.bn2(x)
        x = F.relu(x)
        x = self.pool1(x)

        # Block 2
        x = self.conv3(x)
        if self.alignment == 'latent':
            x = self.latent_align3(x, sbj_trials)
        elif self.alignment == 'adaptive':
            x = self.abn3(x, sbj_trials)
        else:
            x = self.bn3(x)
        x = self.pool2(x)
        x = self.drop1(x)

        # Classifier
        x = x.reshape(-1, self.n_features)
        x = F.relu(x)
        x = self.fc_out(x)

        return x
