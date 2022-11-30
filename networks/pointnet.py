import torch
import torch.nn as nn
import torch.nn.functional as F


class MiniPointNetfeat(nn.Module):
    def __init__(self, nn_channels=(3, 64, 128, 256)):
        super(MiniPointNetfeat, self).__init__()
        self.nn_channels = nn_channels
        assert len(nn_channels) == 4
        self.conv1 = torch.nn.Conv1d(nn_channels[0], nn_channels[1], 1)
        self.conv2 = torch.nn.Conv1d(nn_channels[1], nn_channels[2], 1)
        self.conv3 = torch.nn.Conv1d(nn_channels[2], nn_channels[3], 1)
        self.bn1 = nn.BatchNorm1d(nn_channels[1])
        self.bn2 = nn.BatchNorm1d(nn_channels[2])
        self.bn3 = nn.BatchNorm1d(nn_channels[3])

    def forward(self, x):
        """

        :param x: (B, C, N) input points
        :return: global feature (B, C') or dense feature (B, C', N)
        """
        n_pts = x.size()[2]
        x = F.relu(self.bn1(self.conv1(x)))

        pointfeat = x
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.bn3(self.conv3(x))
        x = torch.max(x, 2, keepdim=True)[0]  # (B, C', 1)
        x = x.view(-1, self.nn_channels[-1])  # (B, C')
        global_feat = x
        x = x.view(-1, self.nn_channels[-1], 1).repeat(1, 1, n_pts)
        return torch.cat([x, pointfeat], dim=1), global_feat  # (B, C', N), (B, C')
