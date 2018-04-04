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

## deJPEG 
fname = 'data/denoising/F16_GT.png'

## denoising
        
if fname == 'data/denoising/F16_GT.png':
    # Add synthetic noise
    img_pil = crop_image(get_image(fname, imsize)[0], d=32)
    img_np = pil_to_np(img_pil)
    
    img_noisy_pil, img_noisy_np = get_noisy_image(img_np, sigma_)
    
    #if PLOT:
        #plot_image_grid([img_np, img_noisy_np], 4, 6)
else:
    assert False

def getShuffleImg(img_pil):
    BLOCKLEN = 1 # Adjust and be careful here.

    width, height = img_pil.size

    xblock = width / BLOCKLEN
    yblock = height / BLOCKLEN
    blockmap = [(xb*BLOCKLEN, yb*BLOCKLEN, (xb+1)*BLOCKLEN, (yb+1)*BLOCKLEN)
            for xb in range(math.floor(xblock)) for yb in range(math.floor(yblock))]

    shuffle = list(blockmap)
    random.shuffle(shuffle)

    img_shf = Image.new(img_pil.mode, (width, height))
    for box, sbox in zip(blockmap, shuffle):
        c = img_pil.crop(sbox)
        img_shf.paste(c, box)

    np_img_shf = pil_to_np(img_shf)
    return np_img_shf

def getNoiseImg():
    noise01 = np.float32(np.random.uniform(0,1,(3, 512, 512)))
    return noise01

i = 0

def buildAndRunNetwork(np_targetImg):

    var_targetImg = np_to_var(np_targetImg).type(dtype)
    INPUT = 'noise' # 'meshgrid'
    pad = 'reflection'
    OPT_OVER = 'net' # 'net,input'

    reg_noise_std = 1./30. # set to 1./20. for sigma=50
    LR = 0.01

    OPTIMIZER='adam' # 'LBFGS'
    show_every = 500

    num_iter=2400
    input_depth = 3
    figsize = 5 
    
    net = skip(
                input_depth, 3, 
                num_channels_down = [8, 16, 32, 64, 128], 
                num_channels_up   = [8, 16, 32, 64, 128],
                num_channels_skip = [0, 0, 0, 4, 4], 
                upsample_mode='bilinear',
                need_sigmoid=True, need_bias=True, pad=pad, act_fun='LeakyReLU')

    net = net.type(dtype)
    
    net_input = get_noise(input_depth, INPUT, (img_pil.size[1], img_pil.size[0])).type(dtype).detach()

    # Compute number of parameters
    s  = sum([np.prod(list(p.size())) for p in net.parameters()]); 
    print ('Number of params: %d' % s)

    # Loss
    mse = torch.nn.MSELoss().type(dtype)

    # img_noisy_var = np_to_var(img_noisy_np).type(dtype)

    net_input_saved = net_input.data.clone()
    noise = net_input.data.clone()

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
    return loss


loss_img = buildAndRunNetwork(img_np)
loss_img_noise = buildAndRunNetwork(img_noisy_np)
img_shf_np = getShuffleImg(img_pil)
loss_img_shuffle = buildAndRunNetwork(img_shf_np)
noise_np = getNoiseImg()
loss_noise = buildAndRunNetwork(noise_np)

df = pd.DataFrame()

df['img'] = loss_img
df['img_noise'] = loss_img_noise
df['img_shuffle'] = loss_img_shuffle
df['noise'] = loss_noise

df.to_csv('log.csv')
