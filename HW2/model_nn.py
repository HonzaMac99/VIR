import torch
import torch.nn as nn
import matplotlib.pyplot as plt

import datasets as data     # This is our data script template

''' Mnist '''
train, validation, test = data.Download_MNIST()

trn_dataset = data.Classification_Dataset(preloaded_data=train[0], preloaded_labels=train[1])
trn_loader = torch.utils.data.DataLoader(trn_dataset, batch_size=64, shuffle=True)

val_dataset = data.Classification_Dataset(preloaded_data=validation[0], preloaded_labels=validation[1])
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=64, shuffle=False)

test_dataset = data.Classification_Dataset(preloaded_data=test[0], preloaded_labels=test[1])
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=64, shuffle=False)

''' Cifar10 '''
# train_cifar10, test_cifar10 = data.Download_CIFAR10()
# cifar_trn_loader = torch.utils.data.DataLoader(train_cifar10, batch_size=8, shuffle=True)
# cifar_test_loader = torch.utils.data.DataLoader(test_cifar10, batch_size=8, shuffle=True)

''' Set seed to fix number for repeatability '''
torch.manual_seed(1)
''' Hyperparameters '''
n = 784   # number of neurons
nbr_cls = 10


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


''' Linear Model '''
linear = nn.Linear(n, nbr_cls)
softmax = torch.nn.Softmax(dim=1)
linear_model = nn.Sequential(linear, softmax)

''' Conv Model'''
class Conv_model(nn.Module):
        def __init__(self, in_channels=3, nbr_classes=10):
                super().__init__()

                self.Conv1 = nn.Conv2d(in_channels=in_channels, out_channels=8, kernel_size=(3, 3), stride=2, padding=0)
                self.Conv2 = nn.Conv2d(in_channels=8, out_channels=24, kernel_size=(3, 3), stride=2, padding=0)
                self.Conv3 = nn.Conv2d(24, 4, (3, 3), stride=1, padding=0)

                self.linear = nn.Linear(4 * 4 * 4, nbr_classes)
                self.softmax = nn.Softmax(dim=1)

        def forward(self, x):
                ''' reshape by batch_size '''
                x = x.view(len(x), 1, 28, 28)

                x = self.Conv1(x)
                x = self.Conv2(x)
                x = self.Conv3(x)
                ''' reshape for Fully connected layer'''
                x = x.view(len(x), 4 * 4 * 4)

                x = self.linear(x)
                ''' Apply Softmax '''
                logits = self.softmax(x)

                return logits


model = Conv_model()
''' Loss '''
loss_function = nn.CrossEntropyLoss()
''' Optimizer'''
optimizer = torch.optim.Adam(model.parameters(), lr=0.001) # Switch to Adam, faster convergence

accuracy_list = []

for epoch in range(10):
        ''' Store predictions and labels for Accuracy calculation'''
        pred_list = []
        label_list = []

        ''' Iterate over trn_dataloader to sample data and labels '''
        for batch in trn_loader:
                x = batch['data']
                y = batch['label']

                logits = model(x)
                loss = loss_function(logits, y)

                loss.backward()

                optimizer.step()
                optimizer.zero_grad()


        for batch in val_loader:
                with torch.no_grad():
                        x = batch['data']
                        y = batch['label']

                        logits = model(x)
                        loss = loss_function(logits, y)

                        pred_list.append(torch.argmax(logits, dim=1))
                        label_list.append(y)

        pred_list = torch.cat(pred_list)
        label_list = torch.cat(label_list)

        accuracy = (pred_list == label_list).float().mean()

        print(f"Epoch {epoch} \t Validation Accuracy: {accuracy * 100:.3f} %")

        accuracy_list.append(accuracy * 100)

plt.plot(accuracy_list)
plt.title('Accuracy per epoch')
plt.xlabel('Epoch')
plt.ylabel('Accuracy [%]')
plt.show()
