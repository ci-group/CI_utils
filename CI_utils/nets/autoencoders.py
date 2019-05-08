import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


class ConvAE(nn.Module):

    def __init__(self, kd, ks, n_layers, dim_hidden, bn_eps, stride):

        super(ConvAE, self).__init__()

        # (kernel_size - 1) * dilation_size

        # W: Necessary to define new objects for different learning parameters or can the structures be reused?

        self.conv1 = nn.Conv1d(1, 2 * kd, ks, stride=4, padding=ks // 4)
        self.bn1 = nn.BatchNorm1d(2 * kd, eps=bn_eps)

        self.conv_enc1 = nn.Conv1d(kd * 2, kd, ks, stride=2, padding=(ks // 2)-1)
        self.bn_enc1 = nn.BatchNorm1d(kd, eps=bn_eps)

        self.conv_enc = nn.ModuleList(
            [nn.Conv1d(kd, kd, ks, stride=stride, padding=(ks // 2)-1) for i in range(n_layers-1)])
        self.bn_enc = nn.ModuleList([nn.BatchNorm1d(kd, eps=bn_eps) for i in range(n_layers-1)])

        self.conv_code = nn.Conv1d(kd, dim_hidden, 1, stride=1)
        self.bn_code = nn.BatchNorm1d(dim_hidden, eps=bn_eps)

        self.conv_dec1 = nn.ConvTranspose1d(dim_hidden, kd, ks, stride=2, padding=(ks // 2)-1)
        self.bn_dec1 = nn.BatchNorm1d(kd, eps=bn_eps)

        self.conv_dec = nn.ModuleList(
            [nn.ConvTranspose1d(kd, kd, ks, stride=2, padding=(ks // 2)-1) for i in range(n_layers - 1)])
        self.bn_dec = nn.ModuleList([nn.BatchNorm1d(kd, eps=bn_eps) for i in range(n_layers-1)])

        self.conv_dec_last = nn.ConvTranspose1d(kd, kd * 2, ks, stride=4, padding=ks // 4)
        self.bn_dec_last = nn.BatchNorm1d(kd * 2, eps=bn_eps)

        self.conv_rec = nn.ConvTranspose1d(kd * 2, 1, ks + 1, stride=1,padding=ks // 2)


    def forward(self, x):

        x = self.bn1(F.leaky_relu(self.conv1(x)))
        x = self.bn_enc1(F.leaky_relu(self.conv_enc1(x)))

        for lyr_i in range(len(self.conv_enc)):
            x = self.bn_enc[lyr_i](F.leaky_relu(self.conv_enc[lyr_i](x)))

        x = self.bn_code(self.conv_code(x))

        x = self.bn_dec1(F.leaky_relu(self.conv_dec1(x)))

        for lyr_i in range(len(self.conv_dec)):
            x = self.bn_dec[lyr_i](F.leaky_relu(self.conv_dec[lyr_i](x)))

        x = self.bn_dec_last(F.leaky_relu(self.conv_dec_last(x)))
        x = torch.sigmoid(self.conv_rec(x))

        return x