from __future__ import print_function
import matplotlib
matplotlib.use('Agg')   # for work station
import matplotlib.pyplot as plt

import os
# os.environ['CUDA_VISIBLE_DEVICES'] = '1'

import numpy as np
from models import *

import torch
import torch.optim

from torch.autograd import Variable
from utils.denoising_utils import *
from skimage.measure import compare_psnr
import argparse

torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True
dtype = torch.cuda.FloatTensor

imsize =-1
PLOT = True
sigma = 25
sigma_ = sigma/255.

parser = argparse.ArgumentParser(description='Deep image prior - denosing')
parser.add_argument('--image', '-i', type=str, default='f16.png', help='image file name to denoise(e.g. a.png)')
args = parser.parse_args()

filePath = 'data/denoising/' + str(args.image)
img_noisy_pil = crop_image(get_image(filePath, imsize)[0], d=32)
img_noisy_np = pil_to_np(img_noisy_pil)

GT_filePath = 'data/denoising/GT_' + str(args.image)
GT_img_pil = crop_image(get_image(GT_filePath, imsize)[0], d=32)
GT_img_np = pil_to_np(GT_img_pil)

print('Before PSNR:' + str(compare_psnr(GT_img_np, img_noisy_np)) + '\n')

if PLOT:
    plot_image_grid('GTvsNOISY', [GT_img_np, img_noisy_np], 4, 5)


INPUT = 'noise' # 'meshgrid'
pad = 'reflection'
OPT_OVER = 'net' # 'net,input'

reg_noise_std = 1./30. # set to 1./20. for sigma=50
LR = 0.01

OPTIMIZER='adam' # 'LBFGS'
show_every = 300

# build network 
num_iter=1800
input_depth = 32
figsize = 4 

net = get_net(input_depth, 'skip', pad,
        skip_n33d=128, 
        skip_n33u=128, 
        skip_n11=4, 
        num_scales=5,
        upsample_mode='bilinear').type(dtype)

net_input = get_noise(input_depth, INPUT, (img_noisy_pil.size[1], img_noisy_pil.size[0])).type(dtype).detach()

# Compute number of parameters
s  = sum([np.prod(list(p.size())) for p in net.parameters()]); 
print ('Number of params: %d' % s)

# Loss
mse = torch.nn.MSELoss().type(dtype)

img_noisy_var = np_to_var(img_noisy_np).type(dtype)

net_input_saved = net_input.data.clone()
noise = net_input.data.clone()

results = []
i = 0
def closure():
    
    global i
    
    if reg_noise_std > 0:
        net_input.data = net_input_saved + (noise.normal_() * reg_noise_std)
    
    out = net(net_input)
   
    total_loss = mse(out, img_noisy_var)
    total_loss.backward()
        
    print ('Iteration %05d    Loss %f' % (i, total_loss.data[0]), '\r', end='')
    if  PLOT and (i+1) % show_every == 0:
        out_np = var_to_np(out)
        results.append(out_np)
        #plot_image_grid(str(i) + 'th_iter', [np.clip(out_np, 0, 1)], factor=figsize, nrow=1)
        
    i += 1

    return total_loss



p = get_params(OPT_OVER, net, net_input)
optimize(OPTIMIZER, p, closure, LR, num_iter)

if PLOT:
    j = num_iter / show_every   
    k = int(j / 3)

    for r in range(k):
        plot_image_grid('fig_'+ str(r), [np.clip(results[r * 3], 0, 1),np.clip(results[r * 3 + 1], 0, 1) ,np.clip(results[r * 3+2], 0, 1)], factor=20)

out_np = var_to_np(net(net_input))
q = plot_image_grid('RESvsNOISY', [np.clip(out_np, 0, 1), img_noisy_np], factor=13)


print('\nAfter PSNR :' + str(compare_psnr(GT_img_np, out_np)))
