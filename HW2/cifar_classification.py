import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import pickle

''' Set seed to fix number for repeatability '''
torch.manual_seed(1)

''' Cifar10 '''
def load_pickle_file(path):
        with open(path, 'rb') as f:
                data = pickle.load(f)
        return data

class Cifar10_Dataset(torch.utils.data.Dataset):
        def __init__(self, path):
                self.file = load_pickle_file(path)
                self.data = self.file[0]
                self.labels = self.file[1]

        def __getitem__(self, item):
                batch = {'data' : torch.tensor(self.data[item], dtype=torch.float).permute(2,0,1) / 255 * 2 - 1 , # H,W, CH ---> CH, H, W
                         'labels' : torch.tensor(self.labels[item], dtype=torch.long)}

                return batch

        def __len__(self):
                return len(self.data)


def Get_Dataloader(batch_size=64, shuffle=True):
        loaders = []
        for data in ['trn', 'val', 'tst']:
                dataset = Cifar10_Dataset(f'cifar_data/{data}.pkl')
                loaders.append(torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=shuffle))
        return loaders


''' Model '''
def Conv_Block(in_channels, out_channels, kernel=(3,3), padding=1):
    layers = nn.Sequential(nn.Conv2d(in_channels, out_channels, kernel_size=kernel, padding=padding),
                           nn.ReLU(),
                           nn.MaxPool2d(2)
                           )
    return layers

class Simple_Conv_Model(torch.nn.Module):
    def __init__(self, nbr_classes=10):
        super().__init__()

        self.conv_block1 = Conv_Block(3, 32, (5,5), padding=1)
        self.conv_block2 = Conv_Block(32, 64, (5,5), padding=1)

        self.conv3 = nn.Conv2d(64, 64, (5,5), padding=1)
        self.conv4 = nn.Conv2d(64, 64, (5,5), padding=1)


        self.lin1 = nn.Linear(256, nbr_classes)

        self.weight_init()

    def forward(self, x):
        x = self.conv_block1(x)
        x = self.conv_block2(x)
        x = self.conv3(x)
        x = self.conv4(x)

        x = self.lin1(x.view(-1, self.lin1.in_features))

        x = torch.softmax(x, dim=1)

        return x

    def weight_init(self):
        for lay in self.modules():
            if type(lay) in [torch.nn.Conv2d, torch.nn.Linear]:
                torch.nn.init.xavier_uniform_(lay.weight)

def run_epoch(epoch_num, dataloader, model, optimizer, criterion):
        if torch.cuda.is_available():
            device = torch.device(0)
        else:
            device = 'cpu'
        running_loss = 0
        acc_list = []
        model = model.to(device)
        model = model.train()
        if optimizer is None:
                model = model.eval()

        for idx, batch in enumerate(dataloader):
                x = batch['data'].to(device)
                y = batch['labels'].to(device)

                # print(f"maximal value of pixel in batch: {x.max()}")
                prediction = model(x)
                loss = criterion(prediction, y)

                if optimizer is not None:

                        loss.backward()
                        optimizer.step()
                        optimizer.zero_grad()

                running_loss += loss.item()
                acc_list.append(torch.argmax(prediction, dim=1) == y)


                # print(f"Epoch: {epoch_num} \t Iter: {idx + 1} / {int(len(dataloader.dataset) / dataloader.batch_size) + 1}"
                #       f" \t loss: {running_loss / (idx + 1) :.3f}")
                # break # Break to overfitt on one example

        overall_accuracy = torch.cat(acc_list).float().mean() * 100

        return overall_accuracy, running_loss / len(dataloader)

''' Hyperparameters '''
nbr_cls = 10
epoch_num = 10
CLS = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

''' Modules '''
trn_loader, val_loader, test_loader = Get_Dataloader(512, shuffle=False)
model = Simple_Conv_Model(nbr_classes=nbr_cls)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = torch.nn.CrossEntropyLoss()

trn_results = []
val_results = []
trn_loss_list = []
val_loss_list = []

for e in range(epoch_num):
        ''' Training Epoch '''
        trn_acc, trn_loss = run_epoch(epoch_num=e, dataloader=trn_loader, model=model, optimizer=optimizer, criterion=criterion)
        trn_results.append(trn_acc), trn_loss_list.append(trn_loss)

        print(f'Epoch: {e} \t Trn Acc: {trn_acc :.3f} \t Trn Loss: {trn_loss :.3f}', end='\t')

        ''' Validation Epoch '''
        val_acc, val_loss = run_epoch(epoch_num=e, dataloader=val_loader, model=model, optimizer=None, criterion=criterion)
        val_results.append(val_acc), val_loss_list.append(val_loss)

        print(f'Val Acc: {val_acc :.3f} \t Val Loss: {val_loss :.3f}')

''' Testing the Model'''
test_acc, test_loss = run_epoch(epoch_num=e, dataloader=test_loader, model=model, optimizer=None, criterion=criterion)

print(f'Test Acc: {test_acc :.3f} \t Test Loss: {test_loss :.3f}')
