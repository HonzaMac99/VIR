import torch
import torch.nn.functional as F
import numpy as np

import scipy.sparse
from load_model import load_model
import load_data
from load_data import Lidar_Dataset

def weight_init(file=None):
    # Calculation of class weights, if you want to add it.
    # You should keep it in <0;1> for consistency with weights and loss values
    # Hint: you can apply inversion of the values first

    # unique, count = np.unique(labels, return_counts=True)
    weights = np.array((1, 1, 1))
    return weights

def loss_function(weight=None, gamma=2):
    if weight is None:
        weight = np.array((1, 1, 1))

    def focal_loss(prediction, labels):
        # Apply class-weights
        tweight = torch.from_numpy(weight).to(prediction.device).float()
        prediction_prob = F.softmax(prediction, dim=1)

        # If nan are present in loss (exploding gradients usually), you can clip them
        prediction_prob = prediction_prob.clamp(1e-7, 1. - 1e-7)

        # cross entropy part
        # cross_entropy(reduction='none') applies cross entropy element-wise, returning the same 2d map
        result = F.cross_entropy(prediction, labels, reduction='none', weight=tweight)[:, None, ...]

        # focal part
        # You can use tensor.gather() to snap the predictions with highest probibility
        loss_weight = (1 - prediction_prob.gather(1, labels[:, None, ...])) ** gamma

        # final form
        loss = loss_weight * result

        return torch.mean(loss)

    return focal_loss

def optimizer(model, l_rate=0.01):
    return torch.optim.SGD(model.parameters(), l_rate, weight_decay=0.0003)

def confusion_matrix(prediction, label, num_classes=3):
    ''' Creates confusion matrix as in HW2 '''
    predictions = torch.flatten(torch.argmax(prediction, 0)).detach().cpu().numpy()
    labels = torch.flatten(label).detach().cpu().numpy()

    tmp_cm = scipy.sparse.coo_matrix(
        (np.ones(np.prod(labels.shape), 'u8'), (labels, predictions)),
        shape=(num_classes, num_classes)
    ).toarray()

    return tmp_cm

def metric(cm):
    ''' Calculates Intersection-over-Union from the accumulated confusion matrix '''
    IOU = np.zeros(len(cm))

    for x in range(len(IOU)):
        IOU[x] = cm[x, x] / (sum(cm[:, x]) + sum(cm[x, :]) - cm[x, x]) # IOU = TP / (FP + FN + TP)

    return IOU


def train_model(net, dataloader, device='cpu', epochs=10):
    # Device
    if torch.cuda.is_available():
        device = torch.device(2)
    # Training utilities
    net.train()
    optim = optimizer(net)
    criterion = loss_function()

    # Iterate training
    for e in range(epochs):
        conf_matrix = np.zeros((3, 3))
        # Get data ===========================
        for i, batch in enumerate(dataloader):
            data = batch['bev']
            labels = batch['label']

            if torch.cuda.is_available():
                data = data.cuda()
                labels = labels.cuda()

            # Forward pass
            y_ = net(data)

            # Calculate loss
            trn_loss = criterion(y_, labels)

            # Backprop
            optim.zero_grad()
            trn_loss.backward()

            # Update
            optim.step()

            # Calculate confusion matrix
            for j in range(len(data)):
                conf_matrix += confusion_matrix(y_[j], labels[j])

            IOU = metric(conf_matrix)

            print("Epoch: {}/{} \t iter: {} \t loss: {:.5f} \t IOU metric: {}"
                  .format(e + 1, epochs, i, trn_loss, np.round(100 * IOU, decimals=3)), end='\t')

def eval_model(net, dataloader, device='cpu', epochs=10):
    # Device
    if torch.cuda.is_available():
        device = torch.device(2)
    # validation
    net.eval()
    optim = optimizer(net)
    criterion = loss_function()

    # Iterate training
    for e in range(epochs):
        conf_matrix = np.zeros((3, 3))
        # Get data ===========================
        for i, batch in enumerate(dataloader):
            data = batch['bev']
            labels = batch['label']

            if torch.cuda.is_available():
                data = data.cuda()
                labels = labels.cuda()

            # Forward pass
            y_ = net(data)

            # Calculate loss
            trn_loss = criterion(y_, labels)

            # Calculate confusion matrix
            for j in range(len(data)):
                conf_matrix += confusion_matrix(y_[j], labels[j])

            IOU = metric(conf_matrix)

            print("Eval epoch: {}/{} \t iter: {} \t loss: {:.5f} \t IOU metric: {}"
                  .format(e + 1, epochs, i, trn_loss, np.round(100 * IOU, decimals=3)), end='\t')


def main():

    import glob
    ''' Read the paths to the dataset class '''
    trn_paths = sorted(glob.glob('/local/temporary/hw3/trn/*.npy'))
    val_paths = sorted(glob.glob('/local/temporary/hw3/val/*.npy'))

    # trn_paths = sorted(glob.glob('data/trn/*.npy'))
    # val_paths = sorted(glob.glob('data/val/*.npy'))

    ''' Init the dataset and dataloader'''
    trn_dataset = Lidar_Dataset(pcl_paths=trn_paths)
    val_dataset = Lidar_Dataset(pcl_paths=val_paths)

    ''' Think if you need to change num_workers for faster data generation '''
    trn_dataloader = torch.utils.data.DataLoader(trn_dataset, batch_size=24, shuffle=True, num_workers=8)
    val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=12, shuffle=False, num_workers=8)


    ''' Load class-weights if necessary '''
    clz_weights = weight_init()

    ''' Init Model '''
    net = load_model()

    if torch.cuda.is_available():
        net = net.cuda()

    ''' Train your model '''
    train_model(net, trn_dataloader, epochs=100)

    ''' Eval your model '''
    eval_model(net, val_dataloader, epochs=1)

    ''' Save best weights '''
    torch.save(net.state_dict(), "weights.pth")
    print("Done.")

if __name__ == "__main__":
    main()
