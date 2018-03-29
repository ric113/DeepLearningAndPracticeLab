'''Train CIFAR10 with PyTorch.'''
from __future__ import print_function

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

import torchvision
import torchvision.transforms as transforms

import os
import argparse

from models.resnet import ResNet20
from models.resnet import ResNet56
from models.resnet import ResNet110
from models.plainCNN import plainCNN20 
from models.plainCNN import plainCNN56
from models.plainCNN import plainCNN110

from utils import progress_bar
from torch.autograd import Variable

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use('Agg')   # for work stattion
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches


parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
parser.add_argument('--resume', '-r', action='store_true', help='resume from checkpoint')
parser.add_argument('--model', default="RESNET", type=str, help='model setting')
parser.add_argument('--layer', default=20, type=int, help='layer')
args = parser.parse_args()

use_cuda = torch.cuda.is_available()
best_acc = 0  # best test accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch

print('==> Network :' + args.model + ', Layer :' + str(args.layer))

# Data
print('==> Preparing data..')
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4824, 0.4467), (0.2471, 0.2435, 0.2616)),
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4824, 0.4467), (0.2471, 0.2435, 0.2616)),
])

trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=True, num_workers=2)

testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=2)

classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

# Model
if args.resume:
    # Load checkpoint.
    print('==> Resuming from checkpoint..')
    assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
    checkpoint = torch.load('./checkpoint/ckpt.t7')
    net = checkpoint['net']
    best_acc = checkpoint['acc']
    start_epoch = checkpoint['epoch']
else:
    print('==> Building model..')
    if(args.model == "RESNET"):
        if(args.layer == 20):
            net = ResNet20()
        elif(args.layer == 56):
            net = ResNet56()
        elif(args.layer == 110):
            net = ResNet110()
    elif(args.model == "CNN"):
        if(args.layer == 20):
            net = plainCNN20()
        elif(args.layer == 56):
            net = plainCNN56()
        elif(args.layer == 110):
            net = plainCNN110()

if use_cuda:
    net.cuda()
    net = torch.nn.DataParallel(net, device_ids=range(torch.cuda.device_count()))
    cudnn.benchmark = True

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=1e-4)

scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[81,122], gamma=0.1)

trainAccs = []
testAccs = []
trainLoss = []
testLoss = []

# Training
def train(epoch):
    print('\nEpoch: %d' % epoch)
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    batch = 0

    for batch_idx, (inputs, targets) in enumerate(trainloader):
        if use_cuda:
            inputs, targets = inputs.cuda(), targets.cuda()
        optimizer.zero_grad()
        inputs, targets = Variable(inputs), Variable(targets)
        outputs = net(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        train_loss += loss.data[0]
        _, predicted = torch.max(outputs.data, 1)
        total += targets.size(0)
        correct += predicted.eq(targets.data).cpu().sum()

        progress_bar(batch_idx, len(trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
            % (train_loss/(batch_idx+1), 100.*correct/total, correct, total))
        batch = batch_idx
        
    trainAccs.append(100.*(1.0-correct/total))
    trainLoss.append(train_loss/(batch+1))

def test(epoch):
    global best_acc
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    batch = 0
    for batch_idx, (inputs, targets) in enumerate(testloader):
        if use_cuda:
            inputs, targets = inputs.cuda(), targets.cuda()
        inputs, targets = Variable(inputs, volatile=True), Variable(targets)
        outputs = net(inputs)
        loss = criterion(outputs, targets)

        test_loss += loss.data[0]
        _, predicted = torch.max(outputs.data, 1)
        total += targets.size(0)
        correct += predicted.eq(targets.data).cpu().sum()

        progress_bar(batch_idx, len(testloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
            % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))
        
        batch = batch_idx

    # Save checkpoint.
    acc = 100.*correct/total
    testAccs.append(100.*(1.0-correct/total))
    testLoss.append(test_loss/(batch+1))

    if acc > best_acc:
        print('Saving..')
        state = {
            'net': net.module if use_cuda else net,
            'acc': acc,
            'epoch': epoch,
        }
        if not os.path.isdir('checkpoint'):
            os.mkdir('checkpoint')
        torch.save(state, './checkpoint/ckpt.t7')
        best_acc = acc

'''
def plotResult():
    # Training
    plt.plot(epochs, trainAccs)
    plt.xlabel('epoch')
    plt.ylabel('error rate(%)')
    plt.title('Training Error rate')
    plt.savefig("training.png")
    plt.clf()

    # Testing
    plt.plot(epochs, testAccs)
    plt.xlabel('epoch')
    plt.ylabel('error rate(%)')
    plt.title('Testing Error rate')
    plt.savefig("testing.png")
    plt.clf()

    # Both
    plt.plot(epochs, trainAccs, color="green")
    plt.plot(epochs, testAccs, color="red")
    plt.xlabel('epoch')
    plt.ylabel('error rate(%)')
    plt.title('Error rate')   
    green_patch = mpatches.Patch(color='green', label='training')
    red_patch = mpatches.Patch(color='red', label='testing')
    plt.legend(handles=[red_patch, green_patch])
    plt.savefig("both.png")
'''

def log():
    df = pd.DataFrame()

    df['train Error rate'] = trainAccs
    df['train Loss'] = trainLoss
    df['test Error rate'] = testAccs
    df['test Loss'] = testLoss

    df.to_csv('log_' + args.model + str(args.layer) + '.csv')

for epoch in range(start_epoch, start_epoch+164):
    scheduler.step()
    train(epoch)
    test(epoch)

#plotResult()
log()