import glob
import random
import os
import numpy as np
import time
import datetime
import sys

from torch.utils.data import Dataset
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F

from torch.autograd import Variable
from torch.utils.data import DataLoader
import torch
from visdom import Visdom

import itertools
from PIL import Image

# ドメインAとドメインBの画像データセット生成クラス
class ImageDataset(Dataset):
    def __init__(self, root, transforms_=None, unaligned=False, mode='media'):
        self.transform = transforms.Compose(transforms_)
        self.unaligned = unaligned

        self.files_A = glob.glob(os.path.join(root,#path  ,)
        self.files_B = glob.glob(os.path.join(root,# path,))

    def __getitem__(self, index):
        item_A = self.transform(Image.open(self.files_A[index % len(self.files_A)]).convert('RGB'))

        if self.unaligned:
            item_B = self.transform(Image.open(self.files_B[random.randint(0, len(self.files_B) - 1)]).convert('RGB'))
        else:
            item_B = self.transform(Image.open(self.files_B[index % len(self.files_B)]).convert('RGB'))

        return {'A': item_A, 'B': item_B}

    def __len__(self):
        return max(len(self.files_A), len(self.files_B))



# parameter
class Opts_test():
  def __init__(self):
    self.batch_size = 1
    self.dataroot = '/media'
    self.size = 256
    self.input_nc = 3
    self.output_nc = 3
    self.cpu = True
    self.n_cpu = 8
    self.device_name = 'cuda:0'
    self.device = torch.device(self.device_name)
    self.load_weight = False
    self.generator_A2B = './django_app/models/netG_A2B.pth' 
    self.generator_B2A = './django_app/models/netG_B2A.pth'
    self.cuda = False
  
opt2 = Opts_test()



# call network

# Generator
netG_A2B = Generator(opt2.input_nc, opt2.output_nc)
netG_B2A = Generator(opt2.input_nc, opt2.output_nc)

# Load state dicts
netG_A2B.load_state_dict(torch.load(opt2.generator_A2B))
netG_B2A.load_state_dict(torch.load(opt2.generator_B2A))

# Set model's test mode
netG_A2B.eval()
netG_B2A.eval()

# Inputs & targets memory allocation
Tensor = torch.cuda.FloatTensor if opt2.cuda else torch.Tensor
input_A = Tensor(opt2.batch_size, opt2.input_nc, opt2.size, opt2.size)
input_B = Tensor(opt2.batch_size, opt2.output_nc, opt2.size, opt2.size)


# Dataset loader
transforms_ = [transforms.Resize(int(opt2.size*1.0),Image.BICUBIC),
          transforms.RandomCrop(opt2.size),
          transforms.ToTensor(),
          transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))]
dataloader = DataLoader(ImageDataset(opt2.dataroot, transforms_=transforms_, mode='train'),
                        batch_size=opt2.batch_size, shuffle=False, num_workers=opt2.n_cpu)




##### make img　#####
from torchvision.utils import save_image

num_create = 100

for i, batch in enumerate(dataloader):
    # Set model input
    real_A = Variable(input_A.copy_(batch['A']))
    real_B = Variable(input_B.copy_(batch['B']))

    # Generate output
    fake_B = 0.5*(netG_A2B(real_A).data + 1.0)
    fake_A = 0.5*(netG_B2A(real_B).data + 1.0)

    out_img1 = torch.cat([real_A, fake_B], dim=2)
    out_img2 = torch.cat([real_B, fake_A], dim=2)

    # Save image files
    save_image(out_img1, 'output/A/%04d.png' % (i+1))
    save_image(out_img2, 'output/B/%04d.png' % (i+1))

    if i > num_create:
        break