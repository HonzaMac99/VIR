import torch
import torch.nn as nn
import os

import matplotlib.pyplot as plt
import pickle
import torch.nn.functional as F


GPU_NO = 0

''' Set seed to fix number for repeatability '''
torch.manual_seed(1)

''' hw_2 '''
def load_pickle_file(path):
    with open(path, 'rb') as f:
        data = pickle.load(f)
    return data

class Dataset(torch.utils.data.Dataset):
    def __init__(self, path):
        self.file = load_pickle_file(path)
        self.data = self.file['data']
        self.labels = self.file['labels']

    def __getitem__(self, item):
        batch = {'data' : torch.tensor(self.data[item], dtype=torch.float).permute(2,0,1), # H,W, CH ---> CH, H, W
                 'labels' : torch.tensor(self.labels[item], dtype=torch.long)}

        return batch

    def __len__(self):
        return len(self.data)


def Get_Dataloader(batch_size=64, shuffle=True):
    loaders = []
    for data in ['trn', 'val']:
        dataset = Dataset(f'/local/temporary/vir/hw2/{data}.pkl')
        loaders.append(torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=shuffle))
    return loaders


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


class Model(nn.Module):
    '''Vgg16'''
    def __init__(self, nbr_classes=10):
        super().__init__()

        self.conv_block1 = Conv_Block_2(3, 64, (7,7), padding=3)
        self.conv_block2 = Conv_Block_2(64, 128, (3,3), padding=1)

        self.conv_block3 = Conv_Block_3(128, 256, (3,3), padding=1)
        self.conv_block4 = Conv_Block_3(256, 512, (3,3), padding=1)

        # self.conv_block5 = Conv_Block_3(512, 512, (3,3), padding=1)

        self.lin1 = nn.Linear(512*8*8, nbr_classes)

        self.weight_init()

    def forward(self, x):
        
        x = x / 255*2 -1

        x = self.conv_block1(x)
        x = self.conv_block2(x)
        x = self.conv_block3(x)
        x = self.conv_block4(x)
        #x = self.conv_block5(x)
        
        x = self.lin1(x.view(-1, self.lin1.in_features))

       
        return x

    def weight_init(self):
        for lay in self.modules():
            if type(lay) in [torch.nn.Conv2d, torch.nn.Linear]:
                torch.nn.init.xavier_uniform_(lay.weight)



class Model2(nn.Module):
    '''This is my super cool, but super dumb module'''
    def __init__(self, nbr_classes=10):
        super().__init__()

        self.conv_block1 = Conv_Block_1(3, 32, (5,5), padding=2)
        self.conv_block2 = Conv_Block_1(32, 64, (3,3), padding=1)
        self.conv3 = nn.Conv2d(64, 64, (3,3), padding=1)            
        self.conv4 = nn.Conv2d(64, 64, (3,3), padding=1) 

        self.lin1 = nn.Linear(2048*4, nbr_classes)       

        self.weight_init()

    def forward(self, x):    

        x = self.conv_block1(x)                                                                             
        x = self.conv_block2(x)                                                                                 
        x = self.conv3(x)                                                                          
        x = self.conv4(x)                                                                                                                                                         
        x = self.lin1(x.view(-1, self.lin1.in_features))

        
        return x

    def weight_init(self):
        for lay in self.modules():
            if type(lay) in [torch.nn.Conv2d, torch.nn.Linear]:
                torch.nn.init.xavier_uniform_(lay.weight)


class Model3(torch.nn.Module):       # zacnu s malym modelem s vice parametry, jinak hrozi nepropagace gradientu
    def __init__(self, nbr_classes=10):
        super().__init__()

        conv_out = 32
        kernel_out = 5

        conv_base = 32
        #conv_base = 64

        # this part of code is based on the corona-VIR_Net

        self.conv_block1 = Conv_Block_1(3, conv_base) # , kernel=(5,5), padding=2
        self.conv_block2 = Conv_Block_1(conv_base, 2*conv_base)
        self.conv_block3 = Conv_Block_1(2*conv_base, 4*conv_base)
        self.conv_block4 = Conv_Block_1(4*conv_base, 8*conv_base)
        self.conv_block5 = Conv_Block_1(8*conv_base, 8*conv_base)
        self.conv_block6 = Conv_Block_1(8*conv_base, 16*conv_base)
        self.conv_block7 = Conv_Block_1(16*conv_base, 16*conv_base) #  kernel=(5,5), padding=2
        #self.conv_block8 = Conv_Block_1(16*conv_base, nbr_classes)

        self.lin1 = nn.Linear(16 * conv_base, nbr_classes)

        self.lin2a = nn.Linear(16*conv_base, 32*conv_base)
        self.lin2b = nn.Linear(32*conv_base, nbr_classes)

        self.lin3a = nn.Linear(16*conv_base, 64*conv_base)
        self.lin3b = nn.Linear(64*conv_base, 32*conv_base)
        self.lin3c = nn.Linear(32*conv_base, nbr_classes)
                                                   
        #self.max_ = nn.MaxPool2d(kernel_size = 7, stride = 7, padding = 7)

        self.weight_init()
             
    def forward(self, x):
        #print("begin forward")
        #print(x.shape)
                                      
        x = x / 255 * 2 - 1
                                                      
        x = self.conv_block1(x)
        x = self.conv_block2(x)
        x = self.conv_block3(x)
        x = self.conv_block4(x)
        x = self.conv_block5(x)
        x = self.conv_block6(x)
        x = self.conv_block7(x)
                                                                                                               
        #print(x.shape)


        x = x.view(-1, self.lin3a.in_features)
                                                                                                                                                   
        x = F.relu(self.lin_2a(x))
        x = F.relu(self.lin_2b(x))
                                                                                                                                                                             
        #x = F.relu(self.lin1(x))
                                                                                                                                     
        #x = F.relu(self.lin3a(x))
        #x = F.relu(self.lin3b(x))
        #x = F.relu(self.lin3c(x))
                                                                                                                                                                                                               
        x = torch.softmax(x, dim=1)     
                                                                                                                                                                                                             
        return x

    def weight_init(self):
        for lay in self.modules():
            if type(lay) in [torch.nn.Conv2d, torch.nn.Linear]:
                torch.nn.init.xavier_uniform_(lay.weight)


def load_model():
    # This is the function to be filled. Your returned model needs to be an instance of subclass of torch.nn.Module
    # Model needs to be accepting tensors of shape [B, 3, 128, 128], where B is batch_size, which are in a range of [0-1] and type float32
    # It should be possible to pass in cuda tensors (in that case, model.cuda() will be called first).
    # The model will return scores (or probabilities) for each of the 10 classes, i.e a tensor of shape [B, 10]
    # The resulting tensor should have same device and dtype as incoming tensor

    directory = os.path.abspath(os.path.dirname(__file__))

    # The model should be trained in advance and in this function, you should instantiate model and load the weights into it:
    model = Model3()
    model.load_state_dict(torch.load(directory + '/weights.pts', map_location='cpu'))

    return model


def run_epoch(epoch_num, dataloader, model, optimizer, criterion):
    if torch.cuda.is_available():
        device = torch.device(GPU_NO)
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
epoch_num = 50
learning_rate = 0.001
my_batch_size = 30
weight_decay = 1e-6

CLS = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

''' Modules '''
trn_loader, val_loader = Get_Dataloader(my_batch_size, shuffle=True) 
model = Model(nbr_classes=nbr_cls)
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
criterion = torch.nn.CrossEntropyLoss()

trn_results = []
val_results = []
trn_loss_list = []
val_loss_list = []

val_acc_max = 0
val_acc = 0
training = True 

for e in range(epoch_num):
    ''' Training Epoch '''
    trn_acc, trn_loss = run_epoch(epoch_num=e, dataloader=trn_loader, model=model, optimizer=optimizer, criterion=criterion)
    trn_results.append(trn_acc), trn_loss_list.append(trn_loss)

    print(f'Epoch: {e} \t Trn Acc: {trn_acc :.3f} \t Trn Loss: {trn_loss :.3f}', end='\t')

    ''' Validation Epoch '''
    val_acc, val_loss = run_epoch(epoch_num=e, dataloader=val_loader, model=model, optimizer=None, criterion=criterion)
    val_results.append(val_acc), val_loss_list.append(val_loss)

    print(f'Val Acc: {val_acc :.3f} \t Val Loss: {val_loss :.3f}')

    if val_acc > val_acc_max:
        # store in on harddis
        if training == True:
            print("Saving new weights")
            torch.save(model.state_dict(), 'weights.pts')
        val_acc_max = val_acc

# ''' Testing the Model'''
# test_acc, test_loss = run_epoch(epoch_num=e, dataloader=test_loader, model=model, optimizer=None, criterion=criterion)

# print(f'Test Acc: {test_acc :.3f} \t Test Loss: {test_loss :.3f}')

