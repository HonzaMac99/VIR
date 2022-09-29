import torch
import torch.nn as nn
import math


class YoloLoss(nn.Module):
    def __init__(self):
        super(YoloLoss, self).__init__()
        # how much we should pay loss for no object (noobj) and the box coordinates (coord)
        self.lambda_noobj = 0.5
        self.lambda_coord = 5

    def forward(self, predictions, target):
        """
        LOSS function which you have to implement
        :param predictions: Prediction tensor of size [BATCH_SIZE, HEIGHT, WIDTH, 5]
        :param target: Labels tensor of size [BATCH_SIZE, HEIGHT, WIDTH, 5]
        :return: Computed loss [optionaly other loss segments to plot]
        """

        batch_size = predictions.size()[0]
        h = 10
        w = 10


        box_loss = 0
        object_loss = 0
        no_object_loss = 0
        for i in range(batch_size):
            for j in range(h):
                for k in range(w):
                    c_p = predictions[i, j, k, 0]
                    c_t = target[i, j, k, 0]
                    c_d = (c_t - c_p) ** 2

                    if c_t == 1:
                        [x_p, y_p, h_p, w_p] = predictions[i, j, k][1:]
                        [x_t, y_t, h_t, w_t] = target[i, j, k][1:]
                        x_d = (x_t - x_p) ** 2
                        y_d = (y_t - y_p) ** 2
                        #h_d = (math.sqrt(h_t) - math.sqrt(h_p)) ** 2
                        #w_d = (math.sqrt(w_t) - math.sqrt(w_p)) ** 2
                        h_d = (h_t - h_p) ** 2
                        w_d = (w_t - w_p) ** 2
                        box_loss += self.lambda_coord * (x_d + y_d + h_d + w_d)
                        object_loss += c_d
                    else:
                        no_object_loss += self.lambda_noobj * c_d

        '''
        box_loss /= batch_size
        object_loss /= batch_size
        no_object_loss /= batch_size
        '''

        loss = box_loss + object_loss + no_object_loss

        return loss, box_loss, object_loss, no_object_loss

