import torch
import torch.nn as nn
import os

# Implement some type of Unet, you can use the parameters, that are on the courseware.
# You are allowed to use functional implementation of from some github repo. However, it
# is not advised to do so, since the only extension here are the transposed convolutions and skip connections,
# which you should try to do on your own. We recommend to build at least one functional Down and Up block with
# transposed convolution and skip connection. You can build the rest of the architecture
# with inspiration of some already implemented version


def Get_Up_Conv(in_channels, out_channels, kernel=(3,3), padding=1):
    layers = nn.Sequential(nn.Conv2d(in_channels, out_channels, kernel_size=kernel, padding=padding),
                           nn.BatchNorm2d(out_channels),
                           nn.ReLU(),
                           )

    return layers

def Get_Up_Conv_2(in_channels, out_channels, kernel=(3,3), padding=1):
    layers = nn.Sequential(nn.Conv2d(in_channels, in_channels, kernel_size=kernel, padding=padding),
                           nn.BatchNorm2d(in_channels),
                           nn.ReLU(),
                           nn.Conv2d(in_channels, out_channels, kernel_size=kernel, padding=padding),
                           nn.BatchNorm2d(out_channels),
                           nn.ReLU(),
                           )

    return layers


def Get_Down_Conv(in_channels, out_channels, kernel=(3,3), padding=1):
    layers = nn.Sequential(nn.Conv2d(in_channels, in_channels, kernel_size=kernel, padding=padding),
                           nn.BatchNorm2d(in_channels),
                           nn.ReLU(),
                           )
    return layers

def Get_Down_Conv_2(in_channels, out_channels, kernel=(3,3), padding=1):
    layers = nn.Sequential(nn.Conv2d(in_channels, out_channels, kernel_size=kernel, padding=padding),
                           nn.BatchNorm2d(out_channels),
                           nn.ReLU(),
                           nn.Conv2d(out_channels, out_channels, kernel_size=kernel, padding=padding),
                           nn.BatchNorm2d(out_channels),
                           nn.ReLU(),
                           )
    return layers

class Your_Unet(nn.Module):
    def __init__(self, n_channels=3, n_classes=3, bias=False):
        super(Your_Unet, self).__init__()
        self.name = 'Your_Unet'
        self.n_channels = n_channels
        self.n_classes = n_classes
        # If using batchnorm, you should remove the bias from layers
        self.bias = bias

        # For upsampling, use Transpose Convolution
        # If you battle the dimensions on output, you can pass argument "output_padding=[x,y]" in order to match the dims
        # See docum for details: https://pytorch.org/docs/stable/generated/torch.nn.ConvTranspose2d.html

        self.up_conv = nn.ConvTranspose2d(in_channels=3, out_channels=64, kernel_size=(2, 2), output_padding=(0, 1))
        self.max_pool = nn.MaxPool2d(2)

        self.conv_block1 = Get_Up_Conv(4, 32)
        self.conv_block2 = Get_Up_Conv(32, 64)
        # max_pool
        self.conv_block3 = Get_Up_Conv_2(64, 128)
        # max_pool
        self.conv_block4 = Get_Up_Conv_2(128, 256)
        # max_pool
        self.conv_block5 = Get_Up_Conv_2(256, 512)
        # up_conv
        self.conv_block6 = Get_Down_Conv_2(256+512, 256)
        # up_conv
        self.conv_block7 = Get_Down_Conv_2(128+256, 128)
        # up_conv
        self.conv_block8 = Get_Down_Conv_2(64+128, 64)
        self.conv9 = nn.Conv2d(64, 3, kernel_size=(3, 3), padding=1)

        # Do not forget, if the softmax layer is in Focal Loss implementation or in the model!!!
        # It depends on if you use Focal Loss or not

        # Weight normalization
        self.weight_init()

    def forward(self, x):
        # For skip connection, you can use torch.cat to concatenate the channels
        x = torch.cat((x[:,:1], x[:,1:]), dim=1)
        print(f"Shape of output after concatenation", x.shape)

        # data jsou normalizovana z inicializace

        # dimension down
        x = self.conv_block1(x)
        x_skip1 = self.conv_block2(x)
        x = self.max_pool(x_skip1)
        x_skip2 = self.conv_block3(x)
        x = self.max_pool(x_skip2)
        x_skip3 = self.conv_block4(x)
        x = self.max_pool(x_skip3)
        x = self.conv_block5(x)

        # dimension up
        x = self.up_conv(x)
        x_merge3 = torch.cat((x_skip3[:, :1], x[:, 1:]), dim=1)
        x = self.conv_block6(x_merge3)
        x = self.up_conv(x)
        x_merge2 = torch.cat((x_skip2[:, :1], x[:, 1:]), dim=1)
        x = self.conv_block7(x_merge2)
        x = self.up_conv(x)
        x_merge1 = torch.cat((x_skip1[:, :1], x[:, 1:]), dim=1)
        x = self.conv_block8(x_merge1)
        x = self.conv9(x)

        x = torch.softmax(x, dim=1)

        return x

    def weight_init(self):
        ''' Xavier initialization '''
        for lay in self.modules():
            if type(lay) in [nn.Conv2d, nn.ConvTranspose2d]:
                torch.nn.init.xavier_uniform_(lay.weight)


def load_model(train=True):
    ''' Same way as in HW2 '''

    directory = os.path.abspath(os.path.dirname(__file__))

    # The model should be trained in advance and in this function, you should instantiate model and load the weights into it:

    if train:
        return Your_Unet()

    # Do not forget to set the right path for weights, if it is different

    else:
        model = Your_Unet()
        model.load_state_dict(torch.load(directory + '/weights.pth', map_location='cpu'))
        return model
