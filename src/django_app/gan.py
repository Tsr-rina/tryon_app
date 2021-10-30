# import glob
# import random
# import os
# import numpy as np
# import time
# import datetime
# import sys

# from torch.utils.data import Dataset
# import torchvision.transforms as transforms
# import torch.nn as nn
# import torch.nn.functional as F

# from torch.autograd import Variable
# from torch.utils.data import DataLoader
# import torch
# from visdom import Visdom

# import itertools
# from PIL import Image
# from torchvision.utils import save_image
# import cv2

# # ドメインAとドメインBの画像データセット生成クラス
# class ImageDataset(Dataset):
#     def __init__(self, root, rA, rB,transforms_=None, unaligned=False, mode='media'):
#         print("ImageDatasetにきました")
#         self.transform = transforms.Compose(transforms_)
#         self.unaligned = unaligned


#         self.files_A = os.path.join(root,rA)
#         self.files_B = os.path.join(root,rB)
#         # print("files_A:{}".format(self.files_A))

#     # pair
#     def __getitem__(self, index):

#         item_A = self.transform(Image.open(self.files_A).convert('RGB'))
#         item_B = self.transform(Image.open(self.files_B).convert('RGB'))

#         return {'A': item_A, 'B': item_B}

#     # data 
#     def __len__(self):
#         return max(len(self.files_A), len(self.files_B))

# # ResidualBlock
# # class ResidualBlock(nn.Module):
# #     def __init__(self, in_features):
# #         super(ResidualBlock, self).__init__()

# #         self.conv_block = nn.Sequential(
# #             nn.ReflectionPad2d(1),
# #             nn.Conv2d(in_features, in_features, 3),
# #             nn.InstanceNorm2d(in_features),
# #             nn.ReLU(inplace=True),
# #             nn.ReflectionPad2d(1),
# #             nn.Conv2d(in_features, in_features, 3),
# #             nn.InstanceNorm2d(in_features)
# #         )

# #     def forward(self, x):
# #         return x + self.conv_block(x)

# # class Opts_test():
# #     def __init__(self):
# #         self.batch_size = 1
# #         self.dataroot = './media/'
# #         self.size = 256
# #         self.input_nc = 3
# #         self.output_nc = 3
# #         self.cpu = True
# #         self.n_cpu = 4
# #         self.device_name = 'cuda:0'
# #         self.device = torch.device(self.device_name)
# #         self.load_weight = False
# #         self.generator_A2B = './django_app/models/netG_A2B.pth' 
# #         self.generator_B2A = './django_app/models/netG_B2A.pth'
# #         self.cuda = False

    
# # opt2 = Opts_test()

# # パラメータ設定
# class Opts_test():
#     def __init__(self):
#         self.n_epochs = 50
#         self.start_epoch = 0
#         self.save_data_interval = 10
#         self.save_image_interval = 10
#         self.log_interval = 20
#         self.sample_interval = 10
#         self.batch_size = 1
#         self.size = 256
#         self.n_cpu = 4
#         self.cpu = False
#         self.device_name = "cuda:0"
#         self.device = torch.device(self.device_name)
#         # self.dataroot = '/content/cycle_dataset'
#         self.dataroot = './media'
#         self.output_dir = 'output'
#         self.log_dir = './logs'
#         self.phase = 'train'
#         self.lambda_L1 = 100.0
#         self.epochs_lr_decay = 0
#         self.epochs_lr_decay_start = -1
#         self.path_to_generator = None
#         self.path_to_discriminator = None
#         self.device_name = "cuda:0"
#         self.device = torch.device(self.device_name)
#         self.generator = "./django_app/models/model_G.pth"

# opt2 = Opts_test()


# # class Generator(nn.Module):
# #     def __init__(self, input_nc, output_nc, n_residual_blocks=9):
# #         super(Generator, self).__init__()
# #         print("Generator通過1")

# #         self.model = nn.Sequential(
# #             nn.ReflectionPad2d(3),
# #             nn.Conv2d(input_nc, 64, 7),
# #             nn.InstanceNorm2d(64),
# #             nn.ReLU(inplace=True),

# #             nn.Conv2d(64, 128, 3, stride=2, padding=1),
# #             nn.InstanceNorm2d(128),
# #             nn.ReLU(inplace=True),

# #             nn.Conv2d(128, 256, 3, stride=2, padding=1),
# #             nn.InstanceNorm2d(256),
# #             nn.ReLU(inplace=True),

# #             ResidualBlock(256),
# #             ResidualBlock(256),
# #             ResidualBlock(256),
# #             ResidualBlock(256),
# #             ResidualBlock(256),
# #             ResidualBlock(256),
# #             ResidualBlock(256),
# #             ResidualBlock(256),
# #             ResidualBlock(256),

# #             nn.ConvTranspose2d(256, 128, 3, stride=2, padding=1, output_padding=1),
# #             nn.InstanceNorm2d(128),
# #             nn.ReLU(inplace=True),

# #             nn.ConvTranspose2d(128, 64, 3, stride=2, padding=1, output_padding=1),
# #             nn.InstanceNorm2d(64),
# #             nn.ReLU(inplace=True),

# #             nn.ReflectionPad2d(3),
# #             nn.Conv2d(64, 3, 7),
# #             nn.Tanh()
# #         )
# #         print("Generator通過2")

# #     def forward(self, x):
# #         print("エラー3")
# #         return self.model(x)

# # 生成器Gのクラス定義
# class Generator(nn.Module):
#     def __init__(self):
#         super(Generator, self).__init__()
#         # U-NetのEocoder部分
#         self.down0 = nn.Conv2d(3, 64, kernel_size=4, stride=2, padding=1)
#         self.down1 = self.__encoder_block(64, 128)
#         self.down2 = self.__encoder_block(128, 256)
#         self.down3 = self.__encoder_block(256, 512)
#         self.down4 = self.__encoder_block(512, 512)
#         self.down5 = self.__encoder_block(512, 512)
#         self.down6 = self.__encoder_block(512, 512)
#         self.down7 = self.__encoder_block(512, 512, use_norm=False)
#         # U-NetのDecoder部分
#         self.up7 = self.__decoder_block(512, 512)
#         self.up6 = self.__decoder_block(1024, 512, use_dropout=True)
#         self.up5 = self.__decoder_block(1024, 512, use_dropout=True)
#         self.up4 = self.__decoder_block(1024, 512, use_dropout=True)
#         self.up3 = self.__decoder_block(1024, 256)
#         self.up2 = self.__decoder_block(512, 128)
#         self.up1 = self.__decoder_block(256, 64)
#         # Gの最終出力
#         self.up0 = nn.Sequential(
#             self.__decoder_block(128, 3, use_norm=False),
#             nn.Tanh(),
#         )

#     def __encoder_block(self, input, output, use_norm=True):
#         # LeakyReLU＋Downsampling
#         layer = [
#             nn.LeakyReLU(0.2, True),
#             nn.Conv2d(input, output, kernel_size=4, stride=2, padding=1)
#         ]
#         # BatchNormalization
#         if use_norm:
#             layer.append(nn.BatchNorm2d(output))
#         return nn.Sequential(*layer)

#     def __decoder_block(self, input, output, use_norm=True, use_dropout=False):
#         # ReLU＋Upsampling
#         layer = [
#             nn.ReLU(True),
#             nn.ConvTranspose2d(input, output, kernel_size=4, stride=2, padding=1)
#         ]
#         # BachNormalization
#         if use_norm:
#             layer.append(nn.BatchNorm2d(output))
#         # Dropout
#         if use_dropout:
#             layer.append(nn.Dropout(0.5))
#         return nn.Sequential(*layer)



#     def forward(self, x):
#         # 偽物画像の生成
#         x0 = self.down0(x)
#         x1 = self.down1(x0)
#         x2 = self.down2(x1)
#         x3 = self.down3(x2)
#         x4 = self.down4(x3)
#         x5 = self.down5(x4)
#         x6 = self.down6(x5)
#         x7 = self.down7(x6)
#         y7 = self.up7(x7)
#         # Encoderの出力をDecoderの入力にSkipConnectionで接続
#         y6 = self.up6(self.concat(x6, y7))
#         y5 = self.up5(self.concat(x5, y6))
#         y4 = self.up4(self.concat(x4, y5))
#         y3 = self.up3(self.concat(x3, y4))
#         y2 = self.up2(self.concat(x2, y3))
#         y1 = self.up1(self.concat(x1, y2))
#         y0 = self.up0(self.concat(x0, y1))

#         return y0

#     def concat(self, x, y):
#         # 特徴マップの結合
#         return torch.cat([x, y], dim=1)

# Generator = Generator()

# # call network

# class CallNetWork():
#     def __init__(self, rH, rA, rB):
#         self.rH = rH
#         self.rA = rA
#         self.rB = rB
#         print("エラー1")
        
#         # Generator
#         # netG_A2B = Generator(opt2.input_nc, opt2.output_nc)
#         # netG_B2A = Generator(opt2.input_nc, opt2.output_nc)
#         print("エラー2.5")

#         # Load state dicts
#         self.Generator = Generator.load_state_dict(torch.load(opt2.generator, map_location=torch.device('cpu')))
#         # self.netG_B2A = netG_B2A.load_state_dict(torch.load(opt2.generator_B2A, map_location=torch.device('cpu')))


#         # Set model's test mode
#         # netG_A2B.eval()
#         # netG_B2A.eval()
#         print("エラー3.5")

#         # Inputs & targets memory allocation
#         # self.Tensor = torch.cuda.FloatTensor if opt2.cuda else torch.Tensor
#         # self.input_A = self.Tensor(opt2.batch_size, opt2.input_nc, opt2.size, opt2.size)
#         # self.input_B = self.Tensor(opt2.batch_size, opt2.output_nc, opt2.size, opt2.size)
#         print("エラー4")

#         self.transforms_ = [transforms.Resize(int(opt2.size*1.12),Image.BICUBIC),
#                 transforms.RandomCrop(opt2.size),
#                 # transforms.RandoHorizontalFlip(),
#                 transforms.ToTensor(),
#                 transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))]
#         self.dataloader = DataLoader(ImageDataset(opt2.dataroot, self.rA, self.rB, transforms_=self.transforms_, unaligned=True),
#                                 batch_size=opt2.batch_size, shuffle=False, num_workers=opt2.n_cpu)
        

#     def forward(self):
#         print("ここまで来ました")
#         Tensor = torch.Tensor
#         # 持ってきた画像を前処理して入力 or ajustmentで前処理してから持ってくる
#         num_create = 0
#         for batch_num, data in enumerate(self.dataloader):
#             print(batch_num)
#             input_A = Tensor(8, 3, 256, 256)
#             # real_A = Variable(input_A.copy_(data['A']))
#             # save_image(real_A, "real_A.png")
#             fake_B = self.Generator(data['A'])
#             # save_image(fake_B, "fake_B.png")
#             # two_img = torch.cat([real_A, fake_B], dim=2)
#             # save_image(two_img, "made_img"+str(batch_num)+".png")
#             if batch_num >= num_create:
#                 break
#             print("ここ通った？")
#         return fake_B
    
#     def overlayImage(src, overlay, location):
#         overlay_height, overlay_width = overlay.shape[:2]

#         # 背景をPIL形式に変換
#         src = cv2.cvtColor(src, cv2.COLOR_BGR2RGB)
#         pil_src = Image.fromarray(src)
#         pil_src = pil_src.convert('RGBA')

#         # オーバーレイをPIL形式に変換
#         overlay = cv2.cvtColor(overlay, cv2.COLOR_BGRA2RGBA)
#         pil_overlay = Image.fromarray(overlay)
#         pil_overlay = pil_overlay.convert('RGBA')

#         # 画像を合成
#         pil_tmp = Image.new('RGBA', pil_src.size, (255, 255, 255, 0))
#         pil_tmp.paste(pil_overlay, location, pil_overlay)
#         result_image = Image.alpha_composite(pil_src, pil_tmp)

#         # OpenCV形式に変換
#         return cv2.cvtColor(np.asarray(result_image), cv2.COLOR_RGBA2BGRA)
