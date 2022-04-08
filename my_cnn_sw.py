import torch
from torch import  nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision import transforms
from torch import nn, optim
import numpy as np

class DownScaling(nn.Module):
    """
    downscaling block
    """

    def __init__(self, ch_in, ch_out):
        """
        :param ch_in:
        :param ch_out:
        """
        super(DownScaling, self).__init__()

        self.conv1 = nn.Conv2d(ch_in, ch_out, kernel_size=5, stride=2, padding=2, padding_mode='reflect')
        self.conv2 = nn.Conv2d(ch_out, ch_out, kernel_size=3, stride=1, padding=1, padding_mode='reflect')

    def forward(self, x):
        """
        :param x: [b, ch, h, w]
        :return:
        """
        out1 = self.conv1(x)
        out2 = self.conv2(out1)

        out = out1 + out2

        return out

class Bottleneck(nn.Module):
    """
    bottleneck block
    """

    def __init__(self, ch_in, ch_out):
        """
        :param ch_in:
        :param ch_out:
        """
        super(Bottleneck, self).__init__()

        self.conv1 = nn.Conv2d(ch_in, ch_out, kernel_size=3, stride=1, padding=1, padding_mode='reflect')
        self.conv2 = nn.Conv2d(ch_out, ch_out, kernel_size=3, stride=1, padding=1, padding_mode='reflect')

    def forward(self, x):
        """
        :param x: [b, ch, h, w]
        :return:
        """
        out1 = self.conv1(x)
        out2 = self.conv2(out1)

        out = out1 + out2

        return out

class Upscaling(nn.Module):
    """
    upscaling block
    """
    def __init__(self, ch_out):
        """
        :param ch_out:
        """
        super(Upscaling, self).__init__()

        self.conv1 = nn.Conv2d(ch_out, ch_out, kernel_size=3, stride=1, padding=1, padding_mode='reflect')
        self.conv2 = nn.Conv2d(ch_out, ch_out, kernel_size=3, stride=1, padding=1, padding_mode='reflect')

    def forward(self, x):
        """
        :param x: [b, ch, h, w]
        :return:
        """
        x = F.pixel_shuffle(x,2)
        out1 = self.conv1(x)
        out2 = self.conv2(out1)

        out = out1 + out2

        return out

class WaterBoundaryCNN(nn.Module):
    def __init__(self, input_num_c = 6, minchannel = 4, num_class = 1):
        super(WaterBoundaryCNN, self).__init__()

        self.conv_first = nn.Conv2d(input_num_c, minchannel, kernel_size=1, stride=1, padding=0)
        self.conv_last = nn.Conv2d(minchannel, num_class, kernel_size=1, stride=1, padding=0)
        self.softmax_layer = nn.Softmax(dim = 1)
        # self.sigmoid_layer = nn.Sigmoid()

        self.ds_1 = DownScaling(minchannel, minchannel*4)
        self.ds_2 = DownScaling(minchannel*4, minchannel*16)
        self.ds_3 = DownScaling(minchannel*16, minchannel*64)
        self.ds_4 = DownScaling(minchannel*64, minchannel*256)

        self.bn = Bottleneck(minchannel*256, minchannel*256)

        self.us_1 = Upscaling(minchannel*64)
        self.us_2 = Upscaling(minchannel*16)
        self.us_3 = Upscaling(minchannel*4)
        self.us_4 = Upscaling(minchannel)

    def forward(self, x):
        """
        :param x: [b, 6, h, w]
        :return:
        """
        batchsz = x.size(0)
        x1 = self.conv_first(x)
        x2 = self.ds_1(x1)
        x3 = self.ds_2(x2)
        x4 = self.ds_3(x3)
        x5 = self.ds_4(x4)

        x = self.bn(x5)

        x = self.us_1(x + x5)
        x = self.us_2(x + x4)
        x = self.us_3(x + x3)
        x = self.us_4(x + x2)
        x = self.conv_last(x + x1)

        x = self.softmax_layer(x)

        return x

def main():

    tmp = torch.randn(15, 6, 256, 256)

    net = WaterBoundaryCNN(num_class = 3)
    out = net(tmp)
    print('out:', out.shape)
    print(torch.sum(out[1,:,3,5]))

if __name__ == '__main__':
    main()