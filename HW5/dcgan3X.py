#conda install pytorch=0.4.1 cuda92 -c pytorch
#pip install --user torchvision
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
CUR_DIR = os.getcwd()
resdir = os.path.join(CUR_DIR, 'results_5k')
resdir_exists = os.path.isdir(resdir)
if not (resdir_exists):
  os.mkdir(resdir)

#---
import io, hashlib
import numpy as np
import pandas as pd
from PIL import Image
from datetime import datetime
#---
from inception import InceptionV3
from fid_score import calculate_frechet_distance as cfid

#---
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.utils as vutils
#---
from torchvision import datasets
from torchvision.transforms import Lambda, Resize, ToTensor, Normalize
from torchvision.transforms import ToPILImage
from torchvision.utils import make_grid
#---
from torch.utils.data import Dataset, DataLoader
#---
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
#---
import matplotlib.pyplot as plt

# helper function for saving 4x4 images canvases
def shb2(batch, epoch=0):
  rows, cols = 4, 4
  figure = plt.figure(figsize=(rows, cols))
  plt.title("Images")
  plt.axis("off")
  grid_img = make_grid(batch[:rows*cols], nrow=rows, padding=1, pad_value=1)
  grid_img = grid_img.permute(1, 2, 0)
  if epoch == -1:
    #real images
    #IMG_PATH = f'G:\\VIR2021\\results_5k\\0_reals.png'
    IMG_PATH = os.path.join(resdir, '0_reals.png')
  else:
    #IMG_PATH = f'G:\\VIR2021\\results_5k\\img5k_{epoch}.png'
    IMG_PATH = os.path.join(resdir, f'fake_{epoch}.png')
  PIL_image = Image.fromarray(np.uint8(grid_img)).convert('RGB')
  PIL_image.save(IMG_PATH)

# helper function for test_functions
def print_to_string(*args, **kwargs):
  output = io.StringIO()
  print(*args, file=output, **kwargs)
  contents = output.getvalue()
  output.close()
  return contents

#---
IMG_DIR_5k = os.path.join(CUR_DIR, 'celeba_crop_5k')
ANNO_FILE  = os.path.join(CUR_DIR, 'list_attr_celeba.txt')
#IMG_DIR_5k = 'G:\\Data\\celeba_crop_5k'
#ANNO_FILE = 'G:\\Data\\celeba\\Anno\\list_attr_celeba.txt'

class CustomImageDataset(Dataset):
  def __init__(self, annotations_file, img_dir, mode='train',
               train_length = 5000, test_length = None):
    self.img_labels = pd.read_csv(ANNO_FILE, skiprows=1, delim_whitespace=True)["Male"]
    self.img_dir = img_dir
    self.img_list = sorted(os.listdir(self.img_dir))
    self.mode = mode
    if self.mode == 'train':
      self.img_list = self.img_list[:train_length]
    elif self.mode == 'test':
      self.img_list = self.img_list[train_length:train_length+test_length]
    self.transform = True
    self.label_transform = Lambda(lambda y:torch.tensor((y+1)//2, dtype=torch.int8))

  def __len__(self):
      return len(self.img_list)

  def __getitem__(self, idx):
    img_name = self.img_list[idx]
    img_path = os.path.join(self.img_dir, img_name)
    PILImage = Image.open(img_path)
    img_name = img_name[:6]+'.jpg'
    label = self.img_labels.loc[img_name]
    if self.transform:
      PILImage = Resize(64)(PILImage)
      image = ToTensor()(PILImage) * 255
      ### --- Task 1 ---
      ### image is float32 tensor of shape (3, 64, 64) with values
      ### in range [0, 255]. Rescale image tensor to range [-1, 1]
      image = image / 255 * 2 - 1
      ### --- END of Task 1 ---
      #print(image.min(), image.max())
    if self.label_transform:
      label = self.label_transform(label)

    return image, label

td_train = CustomImageDataset(ANNO_FILE, IMG_DIR_5k)
print(f'num images ... {len(td_train):d}')
BATCH_SIZE = 64
FID_BATCH = 1024
NZ = 100
dataloader = DataLoader(td_train, batch_size=BATCH_SIZE, shuffle=False)

def test_imgdataset():
  test_td_train = CustomImageDataset(ANNO_FILE, IMG_DIR_5k)
  datalen = len(test_td_train)
  assert datalen == 5000
  eps = 0.1
  # first image
  image = test_td_train.__getitem__(0)[0].numpy()
  assert (image.min() >= -(1 + eps)) & (image.max() <= 1 + eps)
  # last image
  image = test_td_train.__getitem__(datalen-1)[0].numpy()
  assert (image.min() >= -(1 + eps)) & (image.max() <= 1 + eps)
  print('Task 1 - passed ... 10 pts.')

if __name__ == '__main__':
  test_imgdataset()

#---
def weights_init(m):
  classname = m.__class__.__name__
  if classname.find('Conv') != -1:
    nn.init.normal_(m.weight.data, 0.0, 0.02)
  elif classname.find('BatchNorm') != -1:
    nn.init.normal_(m.weight.data, 1.0, 0.02)
    nn.init.constant_(m.bias.data, 0)

class Generator(nn.Module):
  def __init__(self, device):
    super(Generator, self).__init__()
    self.device = device
    # ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding)
    nz = NZ; ngf = 64; nc = 3
    #---
    # input size - nz=100, 1, 1
    self.ConvT_1 = nn.ConvTranspose2d(nz,      ngf * 8, 4, 1, 0, bias=False)
    self.BN_1 = nn.BatchNorm2d(ngf * 8)
    self.RELU_1 = nn.ReLU()
    # state size - ngf*8=512 x 4 x 4
    self.ConvT_2 = nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False)
    self.BN_2 = nn.BatchNorm2d(ngf * 4)
    self.RELU_2 = nn.ReLU()
    # state size - ngf*4=256 x 8 x 8
    ### --- Task 2 ---
    ### Specify generator's layer following the pattern of the 2nd layer
    ### so that produced tensor sizes correspond to the indicated state sizes.
    self.ConvT_3 = nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False)
    self.BN_3 = nn.BatchNorm2d(ngf * 2)
    self.RELU_3 = nn.ReLU()
    # state size - ngf*2=128 x 16 x 16
    self.ConvT_4 = nn.ConvTranspose2d(ngf * 2, ngf, 4, 2, 1, bias=False)
    self.BN_4 = nn.BatchNorm2d(ngf)
    self.RELU_4 = nn.ReLU()
    # state size - ngf*1= 64 x 32 x 32
    self.ConvT_5 = nn.ConvTranspose2d(ngf, 3, 4, 2, 1, bias=False)
    # state size - 3 x 64 x 64
    ### Here add a simple Tanh layer to have an output image in range [-1, 1].
    self.Tanh_5 = nn.Tanh()
    ### --- END of Task 2 ---


  def forward(self, input):
    # input size - 100, 1, 1
    x = self.ConvT_1(input)
    x = self.BN_1(x)
    x = self.RELU_1(x)
    # state size - 512 x 4 x 4
    x = self.ConvT_2(x)
    x = self.BN_2(x)
    x = self.RELU_2(x)
    # state size - 256 x 8 x 8
    x = self.ConvT_3(x)
    x = self.BN_3(x)
    x = self.RELU_3(x)
    # state size - 128 x 16 x 16
    x = self.ConvT_4(x)
    x = self.BN_4(x)
    x = self.RELU_4(x)
    # state size - 64 x 32 x 32
    x = self.ConvT_5(x)
    x = self.Tanh_5(x)
    # state size - 3 x 64 x 64
    return x

class Discriminator(nn.Module):
  def __init__(self, device):
    super(Discriminator, self).__init__()
    self.device = device
    ndf = 64; nc = 3
    # torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride=1, padding=0
    # state size -   3 x 64 x 64
    self.Conv2d_1 = nn.Conv2d(nc , ndf * 1, 4, 2, 1, bias=False)
    self.LRELU_1 = nn.LeakyReLU(0.2)
    # state size -  64 x 32 x 32
    self.Conv2d_2 = nn.Conv2d(ndf * 1, ndf * 2, 4, 2, 1, bias=False)
    self.BN_2 = nn.BatchNorm2d(ndf * 2)
    self.LRELU_2 = nn.LeakyReLU(0.2)
    # state size - 128 x 16 x 16
    ### --- Task 3 ---
    ### Specify next discriminator's layers following the pattern of the 2nd layer
    ### so that produced tensor sizes correspond to the indicated state sizes.
    self.Conv2d_3 = nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False)
    self.BN_3 = nn.BatchNorm2d(ndf * 4)
    self.LRELU_3 = nn.LeakyReLU(0.2)
    # state size - 256 x 8 x 8
    self.Conv2d_4 = nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False)
    self.BN_4 = nn.BatchNorm2d(ndf * 8)
    self.LRELU_4 = nn.LeakyReLU(0.2)
    # state size - 512 x 4 x 4
    self.Conv2d_5 = nn.Conv2d(ndf * 8, 1, 4, 1, 0, bias=False)
    ### Here add a simple Sigmoid layer to get prob. of classification in range [0, 1].
    self.Sigmoid = nn.Sigmoid()
    # state size - 1 x 1 x 1
    ### --- END of Task 3 ---

  def forward(self, input):
    #  3 x 64 x 64
    x = self.Conv2d_1(input)
    x = self.LRELU_1(x)
    # 64 x 32 x 32
    x = self.Conv2d_2(x)
    x = self.BN_2(x)
    x = self.LRELU_2(x)
    # 128 x 16 x 16
    x = self.Conv2d_3(x)
    x = self.BN_3(x)
    x = self.LRELU_3(x)
    # 256 x 8 x 8
    x = self.Conv2d_4(x)
    x = self.BN_4(x)
    x = self.LRELU_4(x)
    # 512 x 4 x 4
    x = self.Conv2d_5(x)
    # 1 x 1 x 1
    x = self.Sigmoid(x)
    # 1 x 1 x 1
    x = x.squeeze()
    return x

#---
netG = Generator(device=device).to(device)
netG.apply(weights_init)
#print(netG)
#fixed_noise = torch.randn((BATCH_SIZE, NZ, 1, 1), device=device)
#image = netG(fixed_noise).detach() # 3x64x64 fake image

def test_generator():
  gen = Generator(device=device).to(device)
  gen_hash = hashlib.md5(print_to_string(gen).encode()).hexdigest()
  assert gen_hash == '20593cfd083a0e414f6dcac6425f34de'
  print('Task 2 - passed ... 35 pts.')

if __name__ == '__main__':
  test_generator()

#---
netD = Discriminator(device=device).to(device)
netD.apply(weights_init)
#print(netD)
#image, label = next(iter(dataloader))
#image = image.to(device)
#realD = netD(image).detach()
#print(f'D prob - {realD[0]:.2f}')

def test_discriminator():
  disc = Discriminator(device=device).to(device)
  disc_hash = hashlib.md5(print_to_string(disc).encode()).hexdigest()
  assert disc_hash == 'c30cfd8614294c1ee2791c099b188b24'
  print('Task 3 - passed ... 35 pts.')

if __name__ == '__main__':
  test_discriminator()

#---
def disc_loss(disc_real, disc_fake):
  loss = -torch.mean(torch.log(disc_real) + torch.log(1. - disc_fake))
  return loss

def gen_loss(disc_fake):
  ### --- Task 4 ---
  ### Specify generator's loss function (replace torch.tensor(0))
  ### The specification of discriminator loss may help.
  loss = - torch.mean(torch.log(disc_fake))
  ### --- END of Task 4 ---
  return loss

def test_gen_loss():
  x = np.linspace(0, np.pi, num=64)
  x = np.exp(np.sin(x))
  disc_fake = torch.tensor(x, dtype=torch.float32)
  loss = np.floor(gen_loss(disc_fake).numpy()*(-1000)).astype(int)
  assert loss == 626
  print('Task 4 - passed ... 20 pts.')

if __name__ == '__main__':
  test_gen_loss()

# Setup Adam optimizers for both G and D
beta1 = 0.5
lr = 0.0002
optimizerD = optim.Adam(netD.parameters(), lr=lr, betas=(beta1, 0.999))
optimizerG = optim.Adam(netG.parameters(), lr=lr, betas=(beta1, 0.999))

#---
# FID preparation
def fid_reals(dataloader):
  # This function computes FID score between two chunks
  # of real images giving lower bound for FID with generated images.
  # It is affected by FID_BATCH option (the higher the lower)
  # mu_1, sgm_1 from the first chunk is later used for computing
  # FID for generated images

  incV3 = InceptionV3().to(device)
  incV3.eval()  # switch to evaluation mode (BatchNorm, DropOut, etc ...)
  real_act = []
  for i, (X, y) in enumerate(dataloader):
    images = X
    images = np.clip((images + 1) * 125.5, 0, 255)  # [-1,1] ---> [0,255]
    if i == 0:
      # save real images
      shb2(images, epoch=-1)
    images = images.to(device)
    # ---
    with torch.no_grad():
      images = images / 255  # [0,255] ---> [0,1]
      act = incV3(images)[0]  # InceptionV3 activations (BS, 2048, 1, 1)
    act = act.squeeze(3).squeeze(2).cpu()  # (BS, 2048)
    real_act.append(act)
    # ---
    if i == 2 * (FID_BATCH // BATCH_SIZE) - 1:  # two chunks
      break

  # ---
  real_act = torch.cat(real_act, dim=0).numpy()
  chunk_split = (FID_BATCH // BATCH_SIZE) * BATCH_SIZE
  chunk_1 = real_act[:chunk_split]
  chunk_2 = real_act[chunk_split:]
  real_mu_1 = np.mean(chunk_1, axis=0)  # (2048,)
  real_sigma_1 = np.cov(chunk_1, rowvar=False)  # (2048,2048)
  real_mu_2 = np.mean(chunk_2, axis=0)
  real_sigma_2 = np.cov(chunk_2, rowvar=False)
  # ---
  # FID between two groups of real images, lower bound for GAN FID
  fid = cfid(real_mu_1, real_sigma_1, real_mu_2, real_sigma_2)
  print(f'fid lwbd        {fid:.2f}')

  return real_mu_1, real_sigma_1, incV3

#---
# Training Loop
NUM_EPOCHS = 100
if __name__ == '__main__':
  print("---")
  print("Starting Training Loop ...")
  print(f"NUM_EPOCHS ...  {NUM_EPOCHS}")
  print("---")
  REAL_MU_1, REAL_SIGMA_1, incV3 = fid_reals(dataloader)
else:
  NUM_EPOCHS = 0
  print("Testing mode. No learning.")

# Output of the generator before learning starts
noise = torch.randn(BATCH_SIZE, NZ, 1, 1, device=device)
fake = netG(noise).detach().cpu()
fake = np.clip((fake+1)*125.5,0,255) #[-1,1] ---> [0,255]
shb2(fake, epoch=0)
#---

for epoch in range(NUM_EPOCHS):
  # For each batch in the dataloader
  for i, data in enumerate(dataloader, 0):
    ############################
    # (1) Update D network: maximize E_x[log(D(x))] + E_z[log(1 - D(G(z)))]
    # which corresponds to minimization of the discriminator loss
    # disc_loss = -torch.mean(torch.log(disc_real) + torch.log(1. - disc_fake))
    ############################
    netD.zero_grad()
    real_images = data[0].to(device)
    b_size = real_images.size(0)  # fractional batches 5000//BATCH_SIZE != 0
    disc_real = netD(real_images)

    nz = NZ # nz = 100
    noise = torch.randn(b_size, nz, 1, 1, device=device)
    fake_images = netG(noise)
    disc_fake = netD(fake_images.detach())
    #---
    errD = disc_loss(disc_real, disc_fake)
    errD.backward()
    #---
    # Update D
    optimizerD.step()

    ############################
    # (2) Update G network: maximize E_z[log(D(G(z)))]
    # which corresponds to minimization of the generator loss
    ############################
    netG.zero_grad()
    disc_fake = netD(fake_images)
    errG = gen_loss(disc_fake)
    errG.backward()
    #---
    # Update G
    optimizerG.step()

  # Save fake images for the first 10 epochs to show detail progress learning
  if epoch > 0 and epoch < 10:
    noise = torch.randn(BATCH_SIZE, NZ, 1, 1, device=device)
    fake = netG(noise).detach().cpu()
    fake = np.clip((fake + 1) * 125.5, 0, 255)  # [-1,1] ---> [0,255]
    shb2(fake, epoch)

  # FID computation at start and every 10-th epoch
  if epoch % 10 == 0:
    fake_act = []
    for i in range(FID_BATCH//BATCH_SIZE):
      noise = torch.randn(BATCH_SIZE, NZ, 1, 1, device=device)
      fake = netG(noise).detach().cpu()
      fake = np.clip((fake+1)*125.5,0,255) #[-1,1] ---> [0,255]
      if i == 0:
        shb2(fake, epoch)
      #---
      with torch.no_grad():
        fake = fake/255 #[0,255] ---> [0,1]
        fake = fake.to(device)
        act = incV3(fake)[0]
      fake_act.append(act.squeeze(3).squeeze(2).cpu())
    #---
    fake_act = torch.cat(fake_act, dim=0).numpy()
    fake_mu = np.mean(fake_act, axis=0)
    fake_sigma = np.cov(fake_act, rowvar=False)
    fid = cfid(REAL_MU_1, REAL_SIGMA_1, fake_mu, fake_sigma)
    #---
    now = datetime.now().strftime('%H:%M:%S')
    print(f'fid ...  {epoch:4d}, {fid:6.2f}, {now}')