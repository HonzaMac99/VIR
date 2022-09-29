import os.path

import torch
import torch.nn as nn


# CNN block without MaxPool
def Conv_Block(in_channels, out_channels, kernel=(3, 3), padding=1):
    layers = nn.Sequential(nn.Conv2d(in_channels, out_channels, kernel_size=kernel, padding=padding, bias=False),
                           nn.BatchNorm2d(out_channels),
                           nn.LeakyReLU(0.1),
                           )
    return layers


# CNN block with MaxPool
def Conv_Block_MP(in_channels, out_channels, kernel=(3, 3), padding=1):
    layers = nn.Sequential(nn.Conv2d(in_channels, out_channels, kernel_size=kernel, padding=padding, bias=False),
                           nn.BatchNorm2d(out_channels),
                           nn.LeakyReLU(0.1),
                           nn.MaxPool2d(2)
                           )
    return layers


class YoloTiny(nn.Module):
    def __init__(self):
        super(YoloTiny, self).__init__()

        self.conv_block1 = Conv_Block_MP(3, 16, (3, 3), padding=1)
        self.conv_block2 = Conv_Block_MP(16, 32, (3, 3), padding=1)
        self.conv_block3 = Conv_Block_MP(32, 64, (3, 3), padding=1)
        self.conv_block4 = Conv_Block_MP(64, 128, (3, 3), padding=1)
        self.conv_block5 = Conv_Block_MP(128, 256, (3, 3), padding=1)
        self.conv_block6 = Conv_Block_MP(256, 512, (3, 3), padding=1)
        self.conv_block7 = Conv_Block(512, 1024, (3, 3), padding=1)
        self.conv_block8 = Conv_Block(1024, 1024, (3, 3), padding=1)
        self.conv_block9 = Conv_Block(1024, 5, (3, 3), padding=1)

        # flatten 10x10x5 --> 1x500

        self.lin1 = nn.Sequential(nn.Linear(500, 1024),
                                  nn.LeakyReLU(0.1))
        self.lin2 = nn.Sequential(nn.Linear(1024, 500),
                                  nn.Sigmoid())

        # self.weight_init()

    def forward(self, x):

        # x = x / 255 * 2 - 1

        x = self.conv_block1(x)
        x = self.conv_block2(x)
        x = self.conv_block3(x)
        x = self.conv_block4(x)
        x = self.conv_block5(x)
        x = self.conv_block6(x)
        x = self.conv_block7(x)
        x = self.conv_block8(x)
        x = self.conv_block9(x)
        x = self.lin1(x.view(-1, self.lin1[0].in_features))
        x = self.lin2(x.view(-1, self.lin2[0].in_features))

        x = torch.reshape(x, (-1, 10, 10, 5))
        # print("this is the size after reshape:", x.size())

        return x

    def weight_init(self):
        for lay in self.modules():
            if type(lay) in [torch.nn.Conv2d, torch.nn.Linear]:
                torch.nn.init.xavier_uniform_(lay.weight)

    def save_model(self, ep_num=0):
        directory = os.path.abspath(os.path.dirname(__file__))
        if ep_num == 0:
            torch.save(self.state_dict(), directory + '/weights.pth')
            print('saved weights - default')
            print(directory + '/weights.pth')
        else:
            torch.save(self.state_dict(), directory + '/weights_%d.pth' % ep_num)
            print('saved weights')
            print(directory + '/weights_%d.pth' % ep_num)

