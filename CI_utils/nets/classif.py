import torch.nn as nn
import torch.nn.functional as F


class VGG_Small(nn.Module):
    """
    This class implements a small version of the popular VGG network,
    which won the ImageNet 2014 challenge.
    """

    def __init__(self, n_channels, n_classes):
        """
        Initializes ConvNet object.

        :param n_channels: number of input channels
        :param n_classes: umber of classes of the classification problem
        """

        super(VGG_Small, self).__init__()

        # Internal Params
        kernel_size = 3
        padding = 1

        self.conv1 = nn.Conv2d(n_channels, 64, kernel_size, padding=padding)
        self.bn1 = nn.BatchNorm2d(64)
        self.max1 = nn.MaxPool2d(3, stride=2, padding=padding)

        self.conv2 = nn.Conv2d(64, 128, kernel_size, padding=padding)
        self.bn2 = nn.BatchNorm2d(128)
        self.max2 = nn.MaxPool2d(3, stride=2, padding=padding)

        self.conv3_a = nn.Conv2d(128, 256, kernel_size, padding=padding)
        self.bn3_a = nn.BatchNorm2d(256)
        self.conv3_b = nn.Conv2d(256, 256, kernel_size, padding=padding)
        self.bn3_b = nn.BatchNorm2d(256)
        self.max3 = nn.MaxPool2d(3, stride=2, padding=padding)

        self.conv4_a = nn.Conv2d(256, 512, kernel_size, padding=padding)
        self.bn4_a = nn.BatchNorm2d(512)
        self.conv4_b = nn.Conv2d(512, 512, kernel_size, padding=padding)
        self.bn4_b = nn.BatchNorm2d(512)
        self.max4 = nn.MaxPool2d(3, stride=2, padding=padding)

        self.conv5_a = nn.Conv2d(512, 512, kernel_size, padding=padding)
        self.bn5_a = nn.BatchNorm2d(512)
        self.conv5_b = nn.Conv2d(512, 512, kernel_size, padding=padding)
        self.bn5_b = nn.BatchNorm2d(512)
        self.max5 = nn.MaxPool2d(3, stride=2, padding=padding)

        self.avg = nn.AvgPool2d(1, stride=1, padding=0)

        self.linear = nn.Linear(512, n_classes)


    def forward(self, x):
        """
        Performs forward pass of the input. Here an input tensor x is
        transformed through several layer transformations.

        :param x: input to the network
        :return out: outputs of the network
        """

        x = self.feature_extractor(x)

        x = self.linear(x)

        return x

    def feature_extractor(self, x):
        """
        Performs forward pass of the input. Here an input tensor x is
        transformed through several layer transformations.

        :param x: input to the network
        :return out: outputs of the network
        """

        x = self.max1(F.relu(self.bn1(self.conv1(x))))
        x = self.max2(F.relu(self.bn2(self.conv2(x))))

        x = F.relu(self.bn3_a(self.conv3_a(x)))
        x = self.max3(F.relu(self.bn3_b(self.conv3_b(x))))

        x = F.relu(self.bn4_a(self.conv4_a(x)))
        x = self.max4(F.relu(self.bn4_b(self.conv4_b(x))))

        x = F.relu(self.bn5_a(self.conv5_a(x)))
        x = self.max5(F.relu(self.bn5_b(self.conv5_b(x))))

        x = self.avg(x)

        x = x.view(-1, 512)

        return x

