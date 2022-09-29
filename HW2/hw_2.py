import torch
import torch.nn as nn
import os
import torch.nn.functional as F



def Conv_Block_1(in_channels, out_channels, kernel=(3,3), padding=1):
    layers = nn.Sequential(nn.Conv2d(in_channels, out_channels, kernel_size=kernel, padding=padding),
                           nn.BatchNorm2d(out_channels),
                           nn.ReLU(),
                           nn.MaxPool2d(2)
                           )
    return layers


def Conv_Block_2(in_channels, out_channels, kernel=(3,3), padding=1):
    layers = nn.Sequential(nn.Conv2d(in_channels, out_channels, kernel_size=kernel, padding=padding),
                           nn.BatchNorm2d(out_channels),
                           nn.ReLU(),
                           nn.Conv2d(out_channels, out_channels, kernel_size=kernel, padding=padding),
                           nn.BatchNorm2d(out_channels),
                           nn.ReLU(),
                           nn.MaxPool2d(2)
                           )
    return layers


def Conv_Block_3(in_channels, out_channels, kernel=(3,3), padding=1):
    layers = nn.Sequential(nn.Conv2d(in_channels, out_channels, kernel_size=kernel, padding=padding),
                           nn.BatchNorm2d(out_channels),
                           nn.ReLU(),
                           nn.Conv2d(out_channels, out_channels, kernel_size=kernel, padding=padding),
                           nn.BatchNorm2d(out_channels),
                           nn.ReLU(),
                           nn.Conv2d(out_channels, out_channels, kernel_size=kernel, padding=padding),
                           nn.BatchNorm2d(out_channels),
                           nn.ReLU(),
                           nn.MaxPool2d(2)
                           )
    return layers


class Model(torch.nn.Module):  # zacnu s malym modelem s vice parametry, jinak hrozi nepropagace gradientu
    def __init__(self, nbr_classes=10):
        super ().__init__ ()

        conv_out = 32
        kernel_out = 5

        conv_base = 32
        # conv_base = 64

        # this part of code is based on the corona-VIR_Net

        self.conv_block1 = Conv_Block_1 (3, conv_base)  # , kernel=(5,5), padding=2
        self.conv_block2 = Conv_Block_1 (conv_base, 2 * conv_base)
        self.conv_block3 = Conv_Block_1 (2 * conv_base, 4 * conv_base)
        self.conv_block4 = Conv_Block_1 (4 * conv_base, 8 * conv_base)
        self.conv_block5 = Conv_Block_1 (8 * conv_base, 8 * conv_base)
        self.conv_block6 = Conv_Block_1 (8 * conv_base, 16 * conv_base)
        self.conv_block7 = Conv_Block_1 (16 * conv_base, 16 * conv_base)  # kernel=(5,5), padding=2
        # self.conv_block8 = Conv_Block_1(16*conv_base, nbr_classes)

        self.lin1 = nn.Linear (16 * conv_base, nbr_classes)

        self.lin2a = nn.Linear (16 * conv_base, 32 * conv_base)
        self.lin2b = nn.Linear (32 * conv_base, nbr_classes)

        self.lin3a = nn.Linear (16 * conv_base, 64 * conv_base)
        self.lin3b = nn.Linear (64 * conv_base, 32 * conv_base)
        self.lin3c = nn.Linear (32 * conv_base, nbr_classes)

        # self.max_ = nn.MaxPool2d(kernel_size = 7, stride = 7, padding = 7)

        self.weight_init ()

    def forward(self, x):
        # print("begin forward")
        # print(x.shape)

        x = x / 255 * 2 - 1

        x = self.conv_block1 (x)
        x = self.conv_block2 (x)
        x = self.conv_block3 (x)
        x = self.conv_block4 (x)
        x = self.conv_block5 (x)
        x = self.conv_block6 (x)
        x = self.conv_block7 (x)

        # print(x.shape)

        x = x.view (-1, self.lin3a.in_features)

        x = F.relu (self.lin_2a (x))
        x = F.relu (self.lin_2b (x))

        # x = F.relu(self.lin1(x))

        # x = F.relu(self.lin3a(x))
        # x = F.relu(self.lin3b(x))
        # x = F.relu(self.lin3c(x))

        x = torch.softmax (x, dim=1)

        return x

    def weight_init(self):
        for lay in self.modules ():
            if type (lay) in [torch.nn.Conv2d, torch.nn.Linear]:
                torch.nn.init.xavier_uniform_ (lay.weight)

def load_model():
    # This is the function to be filled. Your returned model needs to be an instance of subclass of torch.nn.Module
    # Model needs to be accepting tensors of shape [B, 3, 128, 128], where B is batch_size, which are in a range of [0-1] and type float32
    # It should be possible to pass in cuda tensors (in that case, model.cuda() will be called first).
    # The model will return scores (or probabilities) for each of the 10 classes, i.e a tensor of shape [B, 10]
    # The resulting tensor should have same device and dtype as incoming tensor

    directory = os.path.abspath(os.path.dirname(__file__))

    # The model should be trained in advance and in this function, you should instantiate model and load the weights into it:
    model = Model()
    model.load_state_dict(torch.load(directory + '/weights.pts', map_location='cpu'))

    # For more info on storing and loading weights, see https://pytorch.org/tutorials/beginner/saving_loading_models.html
    return model
