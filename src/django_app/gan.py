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





# parameter
class Opts_test():
  def __init__(self):
    self.batch_size = 1
    self.dataroot = '/content/cycle_dataset'
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



# ネットワーク呼び出し

# 生成器G
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