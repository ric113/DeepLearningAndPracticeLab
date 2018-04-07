from __future__ import print_function
import matplotlib
matplotlib.use('Agg')   # for work stattion
import matplotlib.pyplot as plt

import os
# os.environ['CUDA_VISIBLE_DEVICES'] = '1'

import numpy as np
from models import *
import pandas as pd
import sys
import random
from PIL import Image
import math
import argparse

import torch
import torch.optim

from torch.autograd import Variable
from utils.denoising_utils import *

torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True
dtype = torch.cuda.FloatTensor

imsize =-1
PLOT = True
sigma = 25
sigma_ = sigma/255.

parser = argparse.ArgumentParser(description='Deep image prior - Compare noise impedance')
parser.add_argument('--image', '-i', type=str, default='GT_f16.png',help='image file name to compare(e.g. GT_f16.png)')
args = parser.parse_args()

targetImg = args.image
filePath = 'data/denoising/' + targetImg

# pure Img
img_pil = crop_image(get_image(filePath, imsize)[0], d=32)
img_np = pil_to_np(img_pil)

WIDTH = img_pil.size[0]
HEIGHT = img_pil.size[1]


i = 0   # for closure() 


def getShuffleImg(img_pil):
    BLOCKLEN = 1 # Adjust and be careful here.


    xblock = WIDTH / BLOCKLEN
    yblock = HEIGHT / BLOCKLEN
    blockmap = [(xb*BLOCKLEN, yb*BLOCKLEN, (xb+1)*BLOCKLEN, (yb+1)*BLOCKLEN)
            for xb in range(math.floor(xblock)) for yb in range(math.floor(yblock))]

    shuffle = list(blockmap)
    random.shuffle(shuffle)

    img_shf = Image.new(img_pil.mode, (WIDTH, HEIGHT))
    for box, sbox in zip(blockmap, shuffle):
        c = img_pil.crop(sbox)
        img_shf.paste(c, box)

    np_img_shf = pil_to_np(img_shf)
    return np_img_shf

def getNoiseImg():
    noise01 = np.float32(np.random.uniform(0,1,(3, WIDTH, HEIGHT)))
    return noise01

def getNetwork():

    pad = 'reflection'
    input_depth = 3
    figsize = 5 

    net = skip(
        input_depth, 3, 
        num_channels_down = [8, 16, 32, 64, 128], 
        num_channels_up   = [8, 16, 32, 64, 128],
        num_channels_skip = [0, 0, 0, 4, 4], 
        upsample_mode='bilinear',
        need_sigmoid=True, need_bias=True, pad=pad, act_fun='LeakyReLU').type(dtype)
    
    return net

def runNetwork(case, np_targetImg):
    print('Current case : ' + case)
    # Param 
    INPUT = 'noise' # 'meshgrid'
    OPT_OVER = 'net' # 'net,input'
    reg_noise_std = 1./30. # set to 1./20. for sigma=50
    LR = 0.01
    OPTIMIZER='adam' # 'LBFGS'
    show_every = 500
    num_iter=2400
    input_depth = 3

    net = getNetwork()

    net_input = get_noise(input_depth, INPUT, (HEIGHT, WIDTH)).type(dtype).detach()

    # Compute number of parameters
    s  = sum([np.prod(list(p.size())) for p in net.parameters()]); 
    print ('Number of params: %d' % s)

    # Loss
    mse = torch.nn.MSELoss().type(dtype)

    net_input_saved = net_input.data.clone()
    noise = net_input.data.clone()

    var_targetImg = np_to_var(np_targetImg).type(dtype)
   

    loss = []
    def closure():
        
        global i
        
        if reg_noise_std > 0:
            net_input.data = net_input_saved + (noise.normal_() * reg_noise_std)
        
        out = net(net_input)
    
        total_loss = mse(out, var_targetImg)
        total_loss.backward()
            
        print ('Iteration %05d    Loss %f' % (i, total_loss.data[0]), '\r', end='')
          
        i += 1
        loss.append(total_loss.data[0])
        
        return total_loss
    
    p = get_params(OPT_OVER, net, net_input)
    optimize(OPTIMIZER, p, closure, LR, num_iter)
    out_np = var_to_np(net(net_input))
    print('---')
    plot_image_grid(case, [np.clip(out_np, 0, 1), np_targetImg], factor=13)

    global i
    i = 0

    return loss

def main():
    loss_img = runNetwork('PureImg', img_np)
    loss_img_noise = runNetwork('ImgWithNoise', img_noisy_np)
    loss_img_shuffle = runNetwork('SuffuledImg', img_shf_np)
    loss_noise = runNetwork('01NoiseImg', img_01noise_np)

    df = pd.DataFrame()

    df['img'] = loss_img
    df['img_noise'] = loss_img_noise
    df['img_shuffle'] = loss_img_shuffle
    df['noise'] = loss_noise

    df.to_csv('log.csv')

# noisy Img
img_noisy_pil, img_noisy_np = get_noisy_image(img_np, sigma_)

# suffled Img
img_shf_np = getShuffleImg(img_pil)

# U(0, 1) noise Img
img_01noise_np = getNoiseImg()

main()
