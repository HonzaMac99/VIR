import torch
import matplotlib.pyplot as plt

import datasets as data    # This is our data sript template

''' Mnist '''
train, validation, test = data.Download_MNIST()
trn_dataset = data.Classification_Dataset(preloaded_data=train[0], preloaded_labels=train[1])
trn_loader = torch.utils.data.DataLoader(trn_dataset, batch_size=64, shuffle=True)

''' Cifar10 '''
# train_cifar10, test_cifar10 = data.Download_CIFAR10()
# cifar_trn_loader = torch.utils.data.DataLoader(train_cifar10, batch_size=8, shuffle=True)
# cifar_test_loader = torch.utils.data.DataLoader(test_cifar10, batch_size=8, shuffle=True)

''' Set seed to fix number for repeatability '''
torch.manual_seed(1)
''' Hyperparameters '''
n = 784   # number of neurons
lr = 0.01 # try 2, 0.5, 0.01 ... this usually needs to be tune, tried
nbr_cls = 10

''' Weights initialization '''
weights = torch.randn(n, nbr_cls) / torch.sqrt(torch.tensor(n))
weights.requires_grad_()
bias = torch.zeros(10, requires_grad=True)


''' Visualize data '''
def plot_weights(weights, bias, data_sample, epoch=0):
        fig, axs = plt.subplots(2, nbr_cls)

        plt.title(f"Training Epoch: {epoch}")
        results = torch.zeros((784, nbr_cls))
        for i in range(nbr_cls):
                results[:, i] = weights[:, i] * data_sample + bias[i]
                axs[0][i].imshow(weights[:, i].detach().numpy().reshape((28, 28)))
                axs[1][i].imshow(results[:, i].detach().numpy().reshape((28, 28)))
        plt.show()
        plt.close()



def model(weights, bias, x):
        ''' Take weights of neural network and output predictions'''
        preds = x @ weights + bias
        return preds

def softmax(out):
        ''' Implement softmax layer - this should set sum of output layers to 1'''
        logits = out.exp() / out.exp().sum(1).unsqueeze(-1)     # unsqueeze add dimension for broadcast, [batch_size] to [batch_size, 1]
        return logits

def loss_function(logits, labels):
        ''' negative log likelihood - https://pytorch.org/docs/stable/generated/torch.nn.NLLLoss.html '''
        loss = - logits[range(logits.shape[0]), labels].log().mean()
        return loss

accuracy_list = []

for epoch in range(100):
        ''' Store predictions and labels for Accuracy calculation'''
        pred_list = []
        label_list = []

        ''' Iterate over trn_dataloader to sample data and labels '''
        for batch in trn_loader:
                x = batch['data']
                y = batch['label']

                output = model(weights, bias, x)

                logits = softmax(output)
                loss = loss_function(logits, y)

                loss.backward()

                with torch.no_grad():
                        weights -= lr * weights.grad
                        bias -= lr * bias.grad

                        # '''zero the gradients, otherwise accumulating them resulting in non-sense'''
                        weights.grad.zero_()
                        bias.grad.zero_()

                pred_list.append(torch.argmax(logits, dim=1))
                label_list.append(y)

        pred_list = torch.cat(pred_list)
        label_list = torch.cat(label_list)

        accuracy = (pred_list == label_list).float().mean()

        print(f"Epoch {epoch} \t Accuracy: {accuracy * 100:.3f} %")

        accuracy_list.append(accuracy * 100)

plt.plot(accuracy_list)
plt.title('Accuracy per epoch')
plt.xlabel('Epoch')
plt.ylabel('Accuracy [%]')
plt.show()


