#!/usr/bin/env python3

import argparse
import pickle

import numpy as np
import torch
import torch.utils.data as tdata

import hw_2

CLASSES = {
    0: 'bird',
    1: 'lizard',
    2: 'snake',
    3: 'spider',
    4: 'dog',
    5: 'cat',
    6: 'butterfly',
    7: 'monkey',
    8: 'fish',
    9: 'fruit',
}


class Dataset(tdata.Dataset):
    def __init__(self, pkl_name):
        self.pkl_name = pkl_name
        with open(self.pkl_name, 'rb') as f:
            loaded_data = pickle.load(f)
        self.labels = loaded_data['labels']
        self.data = loaded_data['data']

    def __getitem__(self, i):
        return {
            'labels': self.labels[i].astype(
                'i8'
            ),  # torch wants labels to be of type LongTensor, in order to compute losses
            'data': self.data[i].astype('f4').transpose((2, 0, 1)),
            # First retype to float32 (default dtype for torch)
            # then permute axes (torch expects data in CHW order) # Scale input data in your model's forward pass!!!
        }

    def __len__(self):
        return self.labels.shape[0]


def get_prediction_order(prediction, label):
    # prediction has shape [B, 10] (where B is batch size, 10 is number of classes)
    # label has shape [B]

    # both are torch tensors, prediction represents either score or probability of each class.
    # probability is torch.softmax(score, dim=1)

    # either way, the higher the value for each class, the more probable it is according to your model
    # therefore we can sort it according to given probability - and check on which place is the correct label.

    # ideally you want it to be at first place, but for example ImageNet is also evaluated on top-5 error
    # take 5 most confident predictions and only if your label is not in those best predictions, count it as error

    # Since ImageNet dataset has 1000 classes, if your predictions were random, top-5 error should be around 99.5 %

    prediction = prediction.detach()  # detach from computational graph (no grad)
    label = label.detach()

    prediction_sorted = torch.argsort(prediction, 1, True)
    finder = (
            label[:, None] == prediction_sorted
    )  # None as an index creates new dimension of size 1, so that broadcasting works as expected
    order = torch.nonzero(finder)[:, 1]  # returns a tensor of indices, where finder is True.

    return order


def create_confusion_matrix(num_classes, prediction, label):
    prediction = prediction.detach()
    label = label.detach()
    prediction = torch.argmax(prediction, 1)
    cm = torch.zeros(
        (num_classes, num_classes), dtype=torch.long, device=label.device
    )  # empty confusion matrix
    indices = torch.stack((label, prediction))  # stack labels and predictions
    new_indices, counts = torch.unique(
        indices, return_counts=True, dim=1
    )  # Find, how many cases are for each combination of (pred, label)
    cm[new_indices[0], new_indices[1]] += counts

    return cm


def print_stats(conf_matrix, orders):
    num_classes = conf_matrix.shape[0]
    print('Confusion matrix:')
    print(conf_matrix)
    print('\n---\n')
    print('Precision and recalls:')
    for c in range(num_classes):
        precision = conf_matrix[c, c] / conf_matrix[:, c].sum()
        recall = conf_matrix[c, c] / conf_matrix[c].sum()
        f1 = (2 * precision * recall) / (precision + recall)
        print(
            'Class {cls:10s} ({c}):\tPrecision: {prec:0.5f}\tRecall: {rec:0.5f}\tF1: {f1:0.5f}'.format(
                cls=CLASSES[c], c=c, prec=precision, rec=recall, f1=f1
            )
        )

    print('\n---\n')
    print('Top-n accuracy and error:')
    order_len = len(orders)
    for n in range(num_classes):
        topn = (orders <= n).sum()
        acc = topn / order_len
        err = 1 - acc
        print(
            'Top-{n}:\tAccuracy: {acc:0.5f}\tError: {err:0.5f}'.format(n=(n + 1), acc=acc, err=err)
        )


def evaluate(num_classes, dataset_file, batch_size=32, model=None):
    if model is None:
        model = hw_1.load_model()  # load model, your hw
    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    model = model.to(device)
    model = model.eval()  # switch to eval mode, so that some special layers behave nicely
    dataset = Dataset(dataset_file)
    loader = tdata.DataLoader(dataset, batch_size=batch_size)

    confusion_matrix = torch.zeros(
        (num_classes, num_classes), dtype=torch.long, device=device
    )  # empty confusion matrix
    orders = []

    with torch.no_grad():  # disable gradient computation
        for i, batch in enumerate(loader):
            data = batch['data'].to(device)
            labels = batch['labels'].to(device)

            prediction = model(data)
            confusion_matrix += create_confusion_matrix(num_classes, prediction, labels)
            order = get_prediction_order(prediction, labels).cpu().numpy()
            orders.append(order)
            print('Processed {i:02d}th batch'.format(i=(i + 1)))

    print('\n---\n')
    orders = np.concatenate(orders, 0)
    confusion_matrix = confusion_matrix.cpu().numpy()

    print_stats(confusion_matrix, orders)
    return (orders == 0).mean()  # Return top-1 accuracy


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Evaluation demo for HW01')
    parser.add_argument('dataset', type=str)
    parser.add_argument('--batch_size', '-bs', default=32, type=int)
    parser.add_argument('--num_classes', '-nc', default=10, type=int)

    args = parser.parse_args()
    evaluate(args.num_classes, args.dataset, args.batch_size)
