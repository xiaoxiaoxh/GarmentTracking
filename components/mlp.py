from torch import nn
import pytorch_lightning as pl

class PointBatchNorm1D(nn.BatchNorm1d):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    def forward(self, x):
        return super().forward(x.view(-1, x.shape[-1])).view(x.shape)


def MLP(channels, batch_norm=True, last_layer=False, drop_out=False):
    layers = list()
    for i in range(1, len(channels)):
        if i == len(channels) - 1 and last_layer:
            module_layers = [
                nn.Linear(channels[i - 1], channels[i])]
        else:
            module_layers = [
                nn.Linear(channels[i - 1], channels[i])]
            module_layers.append(nn.ReLU())
            if batch_norm:
                module_layers.append(
                    PointBatchNorm1D(channels[i]))
            if drop_out:
                module_layers.append(nn.Dropout(0.5))
        module = nn.Sequential(*module_layers)
        layers.append(module)
    return nn.Sequential(*layers)


class MLP_V2(pl.LightningModule):
    def __init__(self, channels, batch_norm=True, transpose_input=False):
        super(MLP_V2, self).__init__()
        layers = []
        norm_type = 'batch' if batch_norm else None
        for i in range(1, len(channels)):
            if i == len(channels) - 1:
                layers.append(EquivariantLayer(channels[i - 1], channels[i],
                                               activation=None, normalization=None))
            else:
                layers.append(EquivariantLayer(channels[i - 1], channels[i], normalization=norm_type))
        self.layers = nn.ModuleList(layers)
        self.transpose_input = transpose_input

    def forward(self, x):
        expand_dim = False
        if self.transpose_input:
            if len(x.shape) == 2:
                x = x.unsqueeze(1)  # (B, 1, C)
                expand_dim = True
            x = x.transpose(-1, -2)  # (B, C, 1) or (B, C, M)
        for layer in self.layers:
            x = layer(x)
        if self.transpose_input:
            if expand_dim:
                x = x.squeeze(-1)  # (B, C')
            else:
                x = x.transpose(-1, -2)  # (B, M, C')
        return x


class EquivariantLayer(nn.Module):
    def __init__(self, num_in_channels, num_out_channels, activation='relu', normalization=None, momentum=0.1,
                 num_groups=16):
        super(EquivariantLayer, self).__init__()

        self.num_in_channels = num_in_channels
        self.num_out_channels = num_out_channels
        self.activation = activation
        self.normalization = normalization

        self.conv = nn.Conv1d(self.num_in_channels, self.num_out_channels, kernel_size=1, stride=1, padding=0)

        if 'batch' == self.normalization:
            self.norm = nn.BatchNorm1d(self.num_out_channels, momentum=momentum, affine=True)
        elif 'instance' == self.normalization:
            self.norm = nn.InstanceNorm1d(self.num_out_channels, momentum=momentum, affine=True)
        elif 'group' == self.normalization:
            self.norm = nn.GroupNorm(num_groups=num_groups, num_channels=self.num_out_channels)

        if 'relu' == self.activation:
            self.act = nn.ReLU()
        elif 'leakyrelu' == self.activation:
            self.act = nn.LeakyReLU(0.01)

        self.weight_init()

    def weight_init(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                if self.activation == 'relu' or self.activation == 'leakyrelu':
                    nn.init.kaiming_normal_(m.weight, nonlinearity=self.activation)
                else:
                    m.weight.data.normal_(0, std=0.01)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm1d) or isinstance(m, nn.InstanceNorm1d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, x):
        y = self.conv(x)

        if self.normalization == 'batch':
            y = self.norm(y)
        elif self.normalization is not None:
            y = self.norm(y)

        if self.activation is not None:
            y = self.act(y)

        return y