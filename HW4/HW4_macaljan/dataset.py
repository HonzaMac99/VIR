import torch
from torch.utils.data import Dataset
import numpy as np
from PIL import Image,ImageOps
import random
import glob

class myDataset(Dataset):
    def __init__(self, path, augment=False):
        self.root_dir = path
        self.list = glob.glob(self.root_dir+"/*.jpg")
        self.size = len(self.list) # dataset size
        self.output_xsize = 10 # size of the output tensor
        self.output_ysize = 10 # size of the output tensor
        self.augment = augment # augment data by flipping images?

    def __getitem__(self, item):
        # get paths
        [imgpath, labpath] = self.get_paths(item)
        # load image
        image = Image.open(imgpath).convert('RGB')
        image = image.resize((640,640))
        # load labels
        with open(labpath) as f:
            lines = f.readlines()
        boxes = []
        for l in lines:
            boxes.append(np.asarray(l.split()).astype(np.float32))

        # do flip if augment is on
        flip = False
        if self.augment:
            if random.randint(0, 1):
                flip = True
                image = ImageOps.mirror(image)

        # create label matrix
        label_matrix = torch.zeros(self.output_ysize,self.output_xsize, 5)
        for box in boxes:
            if box[0] == 1:
                if flip:
                    x = 1-box[1]
                else:
                    x = box[1]
                y = box[2]
                w = box[3]
                h = box[4]
                cls = box[0]
                i, j = int(self.output_ysize*y), int(self.output_xsize*x)
                x_cell, y_cell = self.output_xsize * x - j, self.output_ysize * y - i
                w_cell, h_cell = (w, h)
                if label_matrix[i,j,0] == 0:
                    box_coors = torch.tensor([cls, x_cell, y_cell, w_cell, h_cell])
                    label_matrix[i,j,0:5] = box_coors

        return {'image': np.moveaxis(np.array(image),-1,0).astype(np.float32), 'label': label_matrix, "im_path": imgpath}

    def __len__(self):
        return self.size

    def get_paths(self, index):
        """
        :param index: integer to specify data
        :return: paths of the image and label
        """
        imgpath = self.list[index].rstrip()
        labpath = imgpath.replace('images', 'labels').replace('.jpg', '.txt')
        return imgpath, labpath




