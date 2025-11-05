#!/usr/bin/env python3 -u
# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in
# the root directory of this source tree.
from __future__ import print_function

import argparse
import csv
import os

import numpy as np
import torch
from torch.autograd import Variable
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from command_dataset import *
import models
from utils import progress_bar




def mixup_data(x, y, alpha=1.0, use_cuda=True):
    '''Returns mixed inputs, pairs of targets, and lambda'''
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 0

    batch_size = x.size()[0]
    if use_cuda:
        index = torch.randperm(batch_size).to("cuda")
    else:
        index = torch.randperm(batch_size)

    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam


def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


def load_data():
        # Data
        print(f'==> Preparing {args.dataset} data..')
        if args.augment:
            transform_train = transforms.Compose([
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465),
                                    (0.2023, 0.1994, 0.2010)),
            ])
        else:
            transform_train = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465),
                                    (0.2023, 0.1994, 0.2010)),
            ])


        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
        if args.dataset == "CIFAR10":
            trainset = datasets.CIFAR10(root='~/data', train=True, download=True,
                                        transform=transform_train)
            trainloader = torch.utils.data.DataLoader(trainset,
                                                    batch_size=args.batch_size,
                                                    shuffle=True, num_workers=8)

            testset = datasets.CIFAR10(root='~/data', train=False, download=True,
                                    transform=transform_test)
            testloader = torch.utils.data.DataLoader(testset, batch_size=args.batch_size,
                                                    shuffle=False, num_workers=8)
        elif args.dataset == "FASHIONMNIST":
            if args.augment:
                transform_train = transforms.Compose([
                    transforms.RandomCrop(32, padding=4),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    transforms.Normalize((0.2860,),(0.3530,))
                ])
            else:
                transform_train = transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize((0.2860,),
                                        (0.3530,))
                ])


            transform_test = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.2860,), (0.3530,)),
            ])
            trainset = datasets.FashionMNIST(
                root="~/data",
                train=True,
                download=True,
                transform=transform_train
            )
            trainloader = torch.utils.data.DataLoader(
                trainset,
                batch_size=args.batch_size,
                shuffle=True,
                num_workers=8
            )

            testset = datasets.FashionMNIST(
                root="~/data",
                train=False,
                download=True,
                transform=transform_test
            )
            testloader = torch.utils.data.DataLoader(
                testset,
                batch_size=args.batch_size,
                shuffle=False,
                num_workers=8
            )
        elif args.dataset == "CIFAR100":
            trainset = datasets.CIFAR100(root='~/data', train=True, download=True,
                                        transform=transform_train)
            trainloader = torch.utils.data.DataLoader(trainset,
                                                    batch_size=args.batch_size,
                                                    shuffle=True, num_workers=8)

            testset = datasets.CIFAR100(root='~/data', train=False, download=True,
                                    transform=transform_test)
            testloader = torch.utils.data.DataLoader(testset, batch_size=100,
                                                    shuffle=False, num_workers=8)
        elif args.dataset == "COMMAND":
            trainloader, testloader, valloader = load_command_data(batch_size = args.batch_size)
            print(f"Load {args.dataset} dataset")
            return trainloader, testloader, valloader
        return trainloader, testloader, None

def train(epoch, trainloader, data="CIFAR10"):
    print(f"\nEpoch: {epoch}")
    net.train()
    train_loss = 0.0
    reg_loss = 0.0  
    correct = 0.0   # keep as float to accumulate weighted counts
    total = 0
    if data == "COMMAND":
        for batch_idx, (inputs, targets) in enumerate(trainloader):
            # move to device
            inputs, targets = inputs.to(device), targets.to(device)
            # apply mixup
            alpha = 0.0 if epoch < 4 else args.alpha
            inputs, targets_a, targets_b, lam = mixup_data(
                    inputs, targets, alpha=alpha, use_cuda=True
                )
                # forward
            outputs = net(inputs)
            loss = mixup_criterion(criterion, outputs, targets_a, targets_b, lam)
            
            # update loss
            train_loss += loss.item()

            # predictions
            _, predicted = outputs.max(1)
            total += targets.size(0)

            # weighted correct counts (no .data / .cpu() gymnastics needed)
            correct += (
                lam * predicted.eq(targets_a).sum().item()
                + (1 - lam) * predicted.eq(targets_b).sum().item()
            )

            # backward + optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            progress_bar(
                batch_idx, len(trainloader),
                "Loss: %.3f | Reg: %.5f | Acc: %.3f%% (%d/%d)" % (
                    train_loss / (batch_idx + 1),
                    reg_loss / (batch_idx + 1),
                    100.0 * correct / total,
                    correct,
                    total
                )
            )
    else:
        for batch_idx, (inputs, targets) in enumerate(trainloader):
            # move to device
            inputs, targets = inputs.to(device), targets.to(device)
            # apply mixup
            inputs, targets_a, targets_b, lam = mixup_data(
                inputs, targets, args.alpha, use_cuda
            )
            # forward
            outputs = net(inputs)
            loss = mixup_criterion(criterion, outputs, targets_a, targets_b, lam)
            train_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)

            correct += (lam * predicted.eq(targets_a.data).cpu().sum().float()
                    + (1 - lam) * predicted.eq(targets_b.data).cpu().sum().float())

            # backward + optimize

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            progress_bar(
                batch_idx, len(trainloader),
                "Loss: %.3f | Reg: %.5f | Acc: %.3f%% (%d/%d)" % (
                    train_loss / (batch_idx + 1),
                    reg_loss / (batch_idx + 1),
                    100.0 * correct / total,
                    correct,
                    total
                )
            )

    return (train_loss / (batch_idx + 1),
            reg_loss / (batch_idx + 1),
            100.0 * correct / total)
def test(epoch, testloader):
    global best_acc
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            if use_cuda:
                inputs, targets = inputs.cuda(), targets.cuda()

            outputs = net(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += targets.size(0)
            correct += predicted.eq(targets.data).cpu().sum()

            progress_bar(batch_idx, len(testloader),
                     'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                     % (test_loss/(batch_idx+1), 100.*correct/total,
                        correct, total))
    acc = 100.*correct/total
    if epoch == start_epoch + args.epoch - 1 or acc > best_acc:
        checkpoint(acc, epoch)
    if acc > best_acc:
        best_acc = acc
    return (test_loss/(batch_idx+1), 100.*correct/total)


def checkpoint(acc, epoch):
    # Save checkpoint.
    print('Saving..')
    state = {
        'net': net,
        'acc': acc,
        'epoch': epoch,
        'rng_state': torch.get_rng_state()
    }
    if not os.path.isdir('checkpoint'):
        os.mkdir('checkpoint')
    torch.save(state, './checkpoint/ckpt.t7' + args.name + '_'
               + str(args.seed))


def adjust_learning_rate(optimizer, epoch):
    """decrease the learning rate at 100 and 150 epoch"""
    lr = args.lr
    if epoch >= 100:
        lr /= 10
    if epoch >= 150:
        lr /= 10
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def adjust_learning_rate_wide(optimizer, epoch):
    """decrease the learning rate at 100 and 150 epoch"""
    lr = args.lr
    if epoch >= 60:
        lr /= 10
    if epoch >= 120:
        lr /= 10
    if epoch >= 180:
        lr /= 10
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def adjust_learning_rate_dense(optimizer, epoch):
    """decrease the learning rate at 100 and 150 epoch"""
    lr = args.lr
    if epoch >= 150:
        lr /= 10
    if epoch >= 225:
        lr /= 10
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
def adjust_learning_rate_vgg(optimizer, epoch):
    """decrease the learning rate at 100 and 150 epoch"""
    lr = args.lr
    if epoch >= 10:
        lr /= 10
    if epoch >= 20:
        lr /= 30
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
    parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
    parser.add_argument('--resume', '-r', action='store_true',
                        help='resume from checkpoint')
    parser.add_argument('--model', default="ResNet18", type=str,
                        help='model type (default: ResNet18)')
    parser.add_argument('--name', default='0', type=str, help='name of run')
    parser.add_argument('--seed', default=0, type=int, help='random seed')
    parser.add_argument('--batch-size', default=128, type=int, help='batch size')
    parser.add_argument('--epoch', default=200, type=int,
                        help='total epochs to run')
    parser.add_argument('--no-augment', dest='augment', action='store_false',
                        help='use standard augmentation (default: True)')
    parser.add_argument('--decay', default=1e-4, type=float, help='weight decay')
    parser.add_argument('--alpha', default=1., type=float,
                        help='mixup interpolation coefficient (default: 1)')
    parser.add_argument('--adjustlr', default="100-150", type=str, help="Adjusted learning rate based on epochs")
    parser.add_argument('--dataset', default="CIFAR10",type=str, help="DATASET" )
    args = parser.parse_args()

    use_cuda = torch.cuda.is_available()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    best_acc = 0  # best test accuracy
    start_epoch = 0  # start from epoch 0 or last checkpoint epoch

    if args.seed != 0:
        torch.manual_seed(args.seed)
    
    trainloader, testloader, valloader = load_data()
    # Model
    if args.resume:
        # Load checkpoint.
        print('==> Resuming from checkpoint..')
        assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
        checkpoint = torch.load('./checkpoint/ckpt.t7' + args.name + '_'
                                + str(args.seed))
        net = checkpoint['net']
        best_acc = checkpoint['acc']
        start_epoch = checkpoint['epoch'] + 1
        rng_state = checkpoint['rng_state']
        torch.set_rng_state(rng_state)
    else:
        print('==> Building model..')
        net = models.__dict__[args.model]()

    if not os.path.isdir('results'):
        os.mkdir('results')
    logname = ('results/log2_' + net.__class__.__name__ + '_' + args.name + '_'
            + str(args.seed) + '.csv')

    if use_cuda:  # or "cuda:0"
        net = net.to(device)           # move model to GPU
        if torch.cuda.device_count() > 1:
            net = torch.nn.DataParallel(net)
        cudnn.benchmark = True
        print(f"Using CUDA with {torch.cuda.device_count()} device(s)")
    else:
        device = torch.device("cpu")
        net = net.to(device)
        print("Using CPU")

    criterion = nn.CrossEntropyLoss()    
    optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0.9,
                        weight_decay=args.decay)
    if args.model in ["LeNet","VGG11", "Abalone", "Arcene"]:
        optimizer = optim.Adam(net.parameters(), lr=args.lr, weight_decay=args.decay)
    if not os.path.exists(logname):
        with open(logname, 'w') as logfile:
            logwriter = csv.writer(logfile, delimiter=',')
            logwriter.writerow(['epoch', 'train loss', 'reg loss', 'train acc',
                                'test loss', 'test acc'])
            
    for epoch in range(start_epoch, args.epoch):
        train_loss, reg_loss, train_acc = train(epoch, trainloader, data=args.dataset)
        test_loss, test_acc = test(epoch, testloader)
        if args.adjustlr == "100-150":
            adjust_learning_rate(optimizer, epoch)
        if args.adjustlr == "60-120-180":
            adjust_learning_rate_wide(optimizer, epoch)
        if args.adjustlr == "10-20-30":
            adjust_learning_rate_vgg(optimizer, epoch)
        with open(logname, 'a') as logfile:
            logwriter = csv.writer(logfile, delimiter=',')
            logwriter.writerow([epoch, train_loss, reg_loss, train_acc, test_loss,
                                test_acc])
        if valloader:
            if epoch == (args.epoch-1):
                val_loss, val_acc = test(epoch, valloader)
                with open(logname, 'a') as logfile:
                    logwriter = csv.writer(logfile, delimiter=',')
                    logwriter.writerow([epoch, train_loss, reg_loss, train_acc, val_loss, val_acc])
    

    