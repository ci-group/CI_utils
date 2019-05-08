import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from DR.CI_utils.CI_utils.nets.autoencoders import ConvAE

class ConvAE_Classif(ConvAE):

    def __init__(self, n_classes, kd, ks, n_layers, dim_hidden, bn_eps, stride):
        super(ConvAE_Classif, self).__init__(kd, ks, n_layers, dim_hidden, bn_eps, stride)

        self.code_comp = nn.Conv1d(dim_hidden, 1, 1, stride=1)

        self.bn_classif = nn.BatchNorm1d(1, eps=bn_eps)

        self.classif = nn.Sequential(
            nn.Linear(1 * 128,  128 * 2),
            nn.ReLU(True),
            nn.Linear(128 * 2, 2)
        )

    def forward(self, x):

        x = self.bn1(F.leaky_relu(self.conv1(x)))
        x = self.bn_enc1(F.leaky_relu(self.conv_enc1(x)))

        for lyr_i in range(len(self.conv_enc)):
            x = self.bn_enc[lyr_i](F.leaky_relu(self.conv_enc[lyr_i](x)))

        x = self.bn_code(self.conv_code(x))

        x_recon = self.bn_dec1(F.leaky_relu(self.conv_dec1(x)))

        for lyr_i in range(len(self.conv_dec)):
            x_recon = self.bn_dec[lyr_i](F.leaky_relu(self.conv_dec[lyr_i](x_recon)))

        x_recon = self.bn_dec_last(F.leaky_relu(self.conv_dec_last(x_recon)))
        x_recon = torch.sigmoid(self.conv_rec(x_recon))

        x_classif = self.bn_classif(F.relu(self.code_comp(x)))

        x_classif = x_classif.view(-1, 1 * 128)

        x_classif = self.classif(x_classif)

        return x_recon, F.softmax(x_classif)