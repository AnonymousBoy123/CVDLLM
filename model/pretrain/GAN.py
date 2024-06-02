import argparse   # argparse 是python自带的命令行参数解析包
import os
import numpy as np
import math

import torchvision.transforms as transforms
from torchvision.utils import save_image

from torch.utils.data import DataLoader # 数据加载
from torchvision import datasets
from torch.autograd import Variable

import torch.nn as nn
import torch.nn.functional as F
import torch

os.makedirs("images", exist_ok=True) # 用于递归创建目录

# 创建解析器
parser = argparse.ArgumentParser()
parser.add_argument("--n_epochs", type=int, default=200, help="number of epochs of training")
parser.add_argument("--batch_size", type=int, default=64, help="size of the batches")
parser.add_argument("--lr", type=float, default=0.0002, help="adam: learning rate")  # 使用的adam 学习速率 设置的比较小
parser.add_argument("--b1", type=float, default=0.5, help="adam: decay of first order momentum of gradient")   # 优化器adam 的两个参数 矩估计的指数衰减率
parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")
parser.add_argument("--n_cpu", type=int, default=8, help="number of cpu threads to use during batch generation")  # 使用cpu 的数量
parser.add_argument("--latent_dim", type=int, default=100, help="dimensionality of the latent space") # 随机噪声z的维度
parser.add_argument("--img_size", type=int, default=28, help="size of each image dimension") # 输入图像的尺寸
parser.add_argument("--channels", type=int, default=1, help="number of image channels")  # 输入图像的channel数
parser.add_argument("--sample_interval", type=int, default=400, help="interval betwen image samples") # 保存生成模型图像的间隔
opt = parser.parse_args() # 所有参数输出
print(opt)

img_shape = (opt.channels, opt.img_size, opt.img_size)  # 图像尺寸 (1.28.28)

cuda = True if torch.cuda.is_available() else False # 使用cuda


class Generator(nn.Module): # 生成器 5个全连接层
    def __init__(self):
        super(Generator, self).__init__()

        def block(in_feat, out_feat, normalize=True):
            layers = [nn.Linear(in_feat, out_feat)]  # Linear全连接层 输入维度，输出维度
            if normalize:  # 要不要正则化
                layers.append(nn.BatchNorm1d(out_feat, 0.8))
            layers.append(nn.LeakyReLU(0.2, inplace=True))  # LeakyReLU的激活函数
            return layers

        self.model = nn.Sequential(
            *block(opt.latent_dim, 128, normalize=False), # 调整维度，不正则化
            *block(128, 256),
            *block(256, 512),
            *block(512, 1024),  # 这三正则化
            nn.Linear(1024, int(np.prod(img_shape))),  # 全连接层 24转化为784维度
            nn.Tanh()  # 值域转化为（-1，1）
        )

    def forward(self, z):  # 把随机噪声z输入到定义的模型
        img = self.model(z)
        img = img.view(img.size(0), *img_shape)
        return img


class Discriminator(nn.Module): # 判别器
    def __init__(self):
        super(Discriminator, self).__init__()

        self.model = nn.Sequential(
            nn.Linear(int(np.prod(img_shape)), 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 1),
            nn.Sigmoid(),  # 把值域变化到 0-1
        )

    def forward(self, img):
        img_flat = img.view(img.size(0), -1)
        validity = self.model(img_flat)

        return validity


# Loss function 损失函数
adversarial_loss = torch.nn.BCELoss()

# Initialize generator and discriminator
generator = Generator()  # 实例化
discriminator = Discriminator()

# cuda 加速
if cuda:
    generator.cuda()
    discriminator.cuda()
    adversarial_loss.cuda()

# Configure data loader 数据加载
os.makedirs("../../data/mnist", exist_ok=True)
dataloader = torch.utils.data.DataLoader(
    datasets.MNIST(
        "../../data/mnist",
        train=True,  # 用来训练的
        download=True,  # 不存在的话就下载
        transform=transforms.Compose(  # 加载后进行变形
            [transforms.Resize(opt.img_size), transforms.ToTensor(), transforms.Normalize([0.5], [0.5])]
        ),  # 调整到需要大小，转成Tensor，归一化 X_N = (x - 0.5)/0.5
    ),
    batch_size=opt.batch_size,
    shuffle=True,  # 将序列的所有元素随机排序
)

# Optimizers 优化器Adam
optimizer_G = torch.optim.Adam(generator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))  # 使用参数的学习速率
optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))

Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor

# ----------
#  Training
# ----------

for epoch in range(opt.n_epochs):
    for i, (imgs, _) in enumerate(dataloader):   # 从数据集和随机向量仲获取输入

        # 分别计算loss，使用反向传播更新模型
        # Adversarial ground truths t是标注.正确的t标注是ground truth
        valid = Variable(Tensor(imgs.size(0), 1).fill_(1.0), requires_grad=False)  # 判定1为真
        fake = Variable(Tensor(imgs.size(0), 1).fill_(0.0), requires_grad=False)  # 判定0为假

        # Configure input
        real_imgs = Variable(imgs.type(Tensor))

        # -----------------
        #  Train Generator
        # -----------------

        optimizer_G.zero_grad()  # 把生成器的梯度清0

        # Sample noise as generator input
        z = Variable(Tensor(np.random.normal(0, 1, (imgs.shape[0], opt.latent_dim))))   # 定义随机噪声，从正态分布均匀采样  opt.latent_dim = 100

        # Generate a batch of images
        gen_imgs = generator(z)  # 生成图像

        # Loss measures generator's ability to fool the discriminator
        g_loss = adversarial_loss(discriminator(gen_imgs), valid)   # 对抗loss 最小化需要判别器的输出和真（1）尽量接近

        g_loss.backward()  # 反向传播
        optimizer_G.step()  # 模型更新

        # ---------------------
        #  Train Discriminator
        # ---------------------

        optimizer_D.zero_grad()

        # Measure discriminator's ability to classify real from generated samples
        real_loss = adversarial_loss(discriminator(real_imgs), valid)
        fake_loss = adversarial_loss(discriminator(gen_imgs.detach()), fake)
        d_loss = (real_loss + fake_loss) / 2

        d_loss.backward()
        optimizer_D.step()

        print(
            "[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f]"
            % (epoch, opt.n_epochs, i, len(dataloader), d_loss.item(), g_loss.item())
        )

        batches_done = epoch * len(dataloader) + i  # 算一个总的batch
        if batches_done % opt.sample_interval == 0:   # 当batch数等于设定参数的倍数的时候
            save_image(gen_imgs.data[:25], "images/%d.png" % batches_done, nrow=5, normalize=True)  # 保存图像

            os.makedirs("model", exist_ok=True)  # 保存模型
            torch.save(generator, 'model/generator.pkl')
            torch.save(discriminator, 'model/discriminator.pkl')

