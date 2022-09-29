import os.path
from pathlib import Path
import requests
import matplotlib.pyplot as plt
import numpy as np
import torch.utils.data


def Download_MNIST(path='data', show=False):
    DATA_PATH = Path(path)
    PATH = DATA_PATH / "mnist"

    PATH.mkdir(parents=True, exist_ok=True)
    URL = "https://github.com/pytorch/tutorials/raw/master/_static/"
    FILENAME = "mnist.pkl.gz"

    if not (PATH / FILENAME).exists():
            content = requests.get(URL + FILENAME).content
            (PATH / FILENAME).open("wb").write(content)

    import pickle
    import gzip

    with gzip.open((PATH / FILENAME).as_posix(), "rb") as f:
            ((x_train, y_train), (x_valid, y_valid), (x_test, y_test)) = pickle.load(f, encoding="latin-1")

    if show:
        fig_mnist, axes = plt.subplots(2,5)

        numbers, ind = np.unique(y_train, return_index=True)

        for i in range(len(axes)):
                for j in range(len(axes[0])):
                        axes[i][j].imshow(x_train[ind][j + i * 5].reshape(28,28))
                        axes[i][j].set_xticklabels([])
                        axes[i][j].set_yticklabels([])
        plt.show()
    return ((x_train, y_train), (x_valid, y_valid), (x_test, y_test))


class Classification_Dataset(torch.utils.data.Dataset):
    def __init__(self, preloaded_data=None, preloaded_labels=None, paths=None):
        self.data = preloaded_data
        self.labels = preloaded_labels
        self.paths = paths

    def __getitem__(self, index):
        '''
        :param index: nbr of the data sample to be taken from the dataloader
        :return: input data to the neural network
        '''
        one_data_sample = self.data[index]#.reshape(28,28)
        one_label_sample = self.labels[index]#.reshape(28,28)

        batch = {'data' : one_data_sample,
                 'label' : one_label_sample,
                 'index' : index}

        return batch


    def __len__(self):
        '''
        :return: maximal index
        '''

        return len(self.data)

if __name__ == '__main__':
    # Get the data
    train_data, validation_data, test_mnist = Download_MNIST()
    # Wrap it in Dataset class to have all together and iterable
    dataset = Classification_Dataset(preloaded_data=train_data[0], preloaded_labels=train_data[1])
    # Wrap it in DataLoader Class to have specified batch_size and shuffle for maximal diversity and randomness to prevent overfitting
    trn_loader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=True)
