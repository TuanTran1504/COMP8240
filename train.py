#!/usr/bin/env python3 -u
# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in
# the root directory of this source tree.
#
# Modifications © 2025, Dinh Tuan Tran. Refactor for clarity/determinism, FMNIST support,
# safer MixUp accuracy, AMP option, unified LR scheduler, and cleaner dataset handling.

import argparse
import csv
import os
import random
from typing import Optional, Tuple

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.optim as optim
import torchvision.datasets as tvds
import torchvision.transforms as T

import models
from utils import progress_bar
from command_dataset import load_command_data  # avoid wildcard imports


# ---------------------------
# Utilities
# ---------------------------

def set_seed(seed: int) -> None:
    if seed is None:
        return
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # Reproducibility (slower but stable)
    cudnn.deterministic = True
    cudnn.benchmark = False


def parse_milestones(spec: str) -> list[int]:
    # "100-150" -> [100, 150]; "60-120-180" -> [60, 120, 180]
    if not spec:
        return []
    return [int(x) for x in spec.split("-") if x.strip().isdigit()]


def save_checkpoint(state: dict, name: str, seed: int, outdir: str = "checkpoint") -> str:
    os.makedirs(outdir, exist_ok=True)
    path = os.path.join(outdir, f"ckpt_{name}_{seed}.pt")
    torch.save(state, path)
    print(f"Saved: {path}")
    return path


def mixup_data(x: torch.Tensor, y: torch.Tensor, alpha: float, device: torch.device):
    """Return mixed inputs, paired targets, and lam."""
    if alpha > 0.0:
        lam = float(np.random.beta(alpha, alpha))
    else:
        lam = 1.0  # effectively no mixup

    batch_size = x.size(0)
    index = torch.randperm(batch_size, device=device)
    mixed_x = lam * x + (1.0 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam


def mixup_loss(criterion, pred, y_a, y_b, lam: float):
    return lam * criterion(pred, y_a) + (1.0 - lam) * criterion(pred, y_b)


def mixup_accuracy(logits: torch.Tensor, y_a: torch.Tensor, y_b: torch.Tensor, lam: float) -> Tuple[float, int]:
    """Weighted top-1 accuracy for MixUp batches."""
    pred = logits.argmax(dim=1)
    correct_a = (pred == y_a).sum().item()
    correct_b = (pred == y_b).sum().item()
    weighted_correct = lam * correct_a + (1.0 - lam) * correct_b
    total = y_a.size(0)
    return float(weighted_correct), int(total)


# ---------------------------
# Data
# ---------------------------

def build_transforms_cifar(augment: bool) -> Tuple[T.Compose, T.Compose]:
    if augment:
        train_tf = T.Compose([
            T.RandomCrop(32, padding=4),
            T.RandomHorizontalFlip(),
            T.ToTensor(),
            T.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
    else:
        train_tf = T.Compose([
            T.ToTensor(),
            T.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
    test_tf = T.Compose([
        T.ToTensor(),
        T.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    return train_tf, test_tf


def build_transforms_fmnist(augment: bool) -> tuple:
    import torchvision.transforms as T
    # Always 32x32, always 1 channel
    common = [T.Resize(32), T.Grayscale(num_output_channels=1)]
    if augment:
        train_tf = T.Compose([
            *common,
            T.RandomCrop(32, padding=4),
            T.RandomHorizontalFlip(),
            T.ToTensor(),
            T.Normalize(mean=(0.2860,), std=(0.3530,)),
        ])
    else:
        train_tf = T.Compose([
            *common,
            T.ToTensor(),
            T.Normalize(mean=(0.2860,), std=(0.3530,)),
        ])
    test_tf = T.Compose([
        *common,
        T.ToTensor(),
        T.Normalize(mean=(0.2860,), std=(0.3530,)),
    ])
    return train_tf, test_tf

def get_loaders(args) -> Tuple[torch.utils.data.DataLoader,
                               torch.utils.data.DataLoader,
                               Optional[torch.utils.data.DataLoader]]:
    root = os.path.expanduser(args.data_root)
    pin = torch.cuda.is_available()
    if args.dataset.upper() == "CIFAR10":
        ttr, tte = build_transforms_cifar(args.augment)
        trainset = tvds.CIFAR10(root=root, train=True, download=True, transform=ttr)
        testset = tvds.CIFAR10(root=root, train=False, download=True, transform=tte)
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size,
                                                  shuffle=True, num_workers=args.num_workers,
                                                  pin_memory=pin)
        testloader = torch.utils.data.DataLoader(testset, batch_size=args.batch_size,
                                                 shuffle=False, num_workers=args.num_workers,
                                                 pin_memory=pin)
        return trainloader, testloader, None

    if args.dataset.upper() == "CIFAR100":
        ttr, tte = build_transforms_cifar(args.augment)
        trainset = tvds.CIFAR100(root=root, train=True, download=True, transform=ttr)
        testset = tvds.CIFAR100(root=root, train=False, download=True, transform=tte)
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size,
                                                  shuffle=True, num_workers=args.num_workers,
                                                  pin_memory=pin)
        testloader = torch.utils.data.DataLoader(testset, batch_size=args.batch_size,
                                                 shuffle=False, num_workers=args.num_workers,
                                                 pin_memory=pin)
        return trainloader, testloader, None

    if args.dataset.upper() == "FASHIONMNIST":
        ttr, tte = build_transforms_fmnist(args.augment)
        trainset = tvds.FashionMNIST(root=root, train=True, download=True, transform=ttr)
        testset = tvds.FashionMNIST(root=root, train=False, download=True, transform=tte)
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size,
                                                  shuffle=True, num_workers=args.num_workers,
                                                  pin_memory=pin)
        testloader = torch.utils.data.DataLoader(testset, batch_size=args.batch_size,
                                                 shuffle=False, num_workers=args.num_workers,
                                                 pin_memory=pin)
        return trainloader, testloader, None

    if args.dataset.upper() == "COMMAND":
        # Assumes this returns (trainloader, testloader, valloader)
        trainloader, testloader, valloader = load_command_data(batch_size=args.batch_size)
        print("Loaded COMMAND dataset via custom loader.")
        return trainloader, testloader, valloader

    raise ValueError(f"Unknown dataset: {args.dataset}")


# ---------------------------
# Train / Eval
# ---------------------------

def train_one_epoch(
    epoch: int,
    net: nn.Module,
    trainloader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    device: torch.device,
    args,
) -> Tuple[float, float, float]:
    net.train()
    running_loss = 0.0
    running_reg = 0.0  
    weighted_correct = 0.0
    total = 0

    # MixUp warmup (e.g., COMMAND first few epochs w/o MixUp)
    effective_alpha = 0.0 if (args.dataset.upper() == "COMMAND" and epoch < args.mixup_warmup_epochs) else args.alpha

    scaler = torch.amp.GradScaler(enabled=args.amp)

    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs = inputs.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        if effective_alpha > 0.0:
            inputs, targets_a, targets_b, lam = mixup_data(inputs, targets, effective_alpha, device)
        else:
            targets_a, targets_b, lam = targets, targets, 1.0

        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast(enabled=args.amp):
            outputs = net(inputs)
            loss = mixup_loss(criterion, outputs, targets_a, targets_b, lam)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        running_loss += loss.item()
        if lam == 1.0:
            # standard accuracy
            pred = outputs.argmax(dim=1)
            weighted_correct += (pred == targets).sum().item()
            total += targets.size(0)
        else:
            wc, tot = mixup_accuracy(outputs, targets_a, targets_b, lam)
            weighted_correct += wc
            total += tot

        progress_bar(
            batch_idx, len(trainloader),
            "Loss: %.3f | Reg: %.5f | Acc: %.3f%% (%d/%d)" % (
                running_loss / (batch_idx + 1),
                running_reg / (batch_idx + 1),
                100.0 * (weighted_correct / max(1, total)),
                int(weighted_correct),
                total
            )
        )

    avg_loss = running_loss / max(1, batch_idx + 1)
    avg_reg = running_reg / max(1, batch_idx + 1)  
    acc = 100.0 * (weighted_correct / max(1, total))
    return avg_loss, avg_reg, acc


@torch.no_grad()
def evaluate(
    epoch: int,
    net: nn.Module,
    loader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> Tuple[float, float]:
    net.eval()
    total = 0
    correct = 0
    running_loss = 0.0
    for batch_idx, (inputs, targets) in enumerate(loader):
        inputs = inputs.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        outputs = net(inputs)
        loss = criterion(outputs, targets)
        running_loss += loss.item()

        pred = outputs.argmax(dim=1)
        correct += (pred == targets).sum().item()
        total += targets.size(0)

        progress_bar(
            batch_idx, len(loader),
            "Loss: %.3f | Acc: %.3f%% (%d/%d)" % (
                running_loss / (batch_idx + 1),
                100.0 * (correct / max(1, total)),
                correct,
                total
            )
        )
    avg_loss = running_loss / max(1, batch_idx + 1)
    acc = 100.0 * (correct / max(1, total))
    return avg_loss, acc


# ---------------------------
# Main
# ---------------------------

def main():
    parser = argparse.ArgumentParser(description="PyTorch Training with MixUp")
    parser.add_argument("--dataset", default="CIFAR10", type=str,
                        choices=["CIFAR10", "CIFAR100", "FASHIONMNIST", "COMMAND"])
    parser.add_argument("--data-root", default="~/data", type=str)
    parser.add_argument("--model", default="ResNet18", type=str, help="Model class name exposed in models.__init__")
    parser.add_argument("--name", default="run0", type=str, help="Run name (used in logs/checkpoints)")
    parser.add_argument("--seed", default=0, type=int, help="Random seed (0 to skip deterministic setup)")
    parser.add_argument("--batch-size", default=128, type=int)
    parser.add_argument("--num-workers", default=4, type=int)
    parser.add_argument("--epoch", default=200, type=int)
    parser.add_argument("--lr", default=0.1, type=float)
    parser.add_argument("--decay", default=1e-4, type=float)
    parser.add_argument("--momentum", default=0.9, type=float)
    parser.add_argument("--augment", dest="augment", action="store_true", help="Enable standard augmentation")
    parser.add_argument("--no-augment", dest="augment", action="store_false", help="Disable augmentation")
    parser.set_defaults(augment=True)

    # Learning-rate milestones; use e.g. "100-150" or "60-120-180"
    parser.add_argument("--adjustlr", default="100-150", type=str,
                        help="Milestones string: e.g., '100-150' or '60-120-180'")
    parser.add_argument("--resume", "-r", action="store_true", help="Resume from checkpoint")

    # MixUp
    parser.add_argument("--alpha", default=1.0, type=float, help="MixUp interpolation coefficient α")
    parser.add_argument("--mixup-warmup-epochs", default=4, type=int,
                        help="Epochs with alpha=0 for COMMAND dataset (spectrogram warmup)")


    # AMP
    parser.add_argument("--amp", action="store_true", help="Enable mixed-precision (AMP)")

    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        print(f"Using CUDA with {torch.cuda.device_count()} device(s)")
    else:
        print("Using CPU")

    if args.seed != 0:
        set_seed(args.seed)
    else:
        # Fast mode (non-deterministic)
        cudnn.benchmark = True

    # Data
    trainloader, testloader, valloader = get_loaders(args)

    # Model
    print("==> Building model..")
    if not hasattr(models, args.model):
        raise ValueError(f"Model {args.model} not found in models package.")
    net = getattr(models, args.model)()
    net = net.to(device)
    if torch.cuda.is_available() and torch.cuda.device_count() > 1:
        net = nn.DataParallel(net)

    # Criterion / Optimizer / Scheduler
    criterion = nn.CrossEntropyLoss()
    if args.model in ["LeNet", "VGG11", "Abalone", "Arcene"]:
        optimizer = optim.Adam(net.parameters(), lr=args.lr, weight_decay=args.decay)
    else:
        optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.decay)

    milestones = parse_milestones(args.adjustlr)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=milestones, gamma=0.1)

    # Logging
    os.makedirs("results", exist_ok=True)
    logname = os.path.join("results", f"log_{net.__class__.__name__}_{args.name}_{args.seed}.csv")
    if not os.path.exists(logname):
        with open(logname, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["epoch", "train_loss", "reg_loss", "train_acc", "test_loss", "test_acc"])

    # Resume
    best_acc = 0.0
    start_epoch = 0
    if args.resume:
        ckpt_path = os.path.join("checkpoint", f"ckpt_{args.name}_{args.seed}.pt")
        assert os.path.isfile(ckpt_path), f"No checkpoint found at {ckpt_path}"
        print(f"==> Resuming from {ckpt_path}")
        ckpt = torch.load(ckpt_path, map_location=device)
        net.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        scheduler.load_state_dict(ckpt["scheduler"])
        best_acc = ckpt.get("best_acc", 0.0)
        start_epoch = ckpt.get("epoch", 0) + 1
        if "rng_state" in ckpt:
            torch.set_rng_state(ckpt["rng_state"])

    # Train loop
    last_test_loss, last_test_acc = None, None
    for epoch in range(start_epoch, args.epoch):
        print(f"\nEpoch {epoch}/{args.epoch - 1}")
        train_loss, reg_loss, train_acc = train_one_epoch(
            epoch, net, trainloader, criterion, optimizer, device, args
        )
        test_loss, test_acc = evaluate(epoch, net, testloader, criterion, device)
        last_test_loss, last_test_acc = test_loss, test_acc

        scheduler.step()

        # Save best / last
        is_last = (epoch == args.epoch - 1)
        is_best = test_acc > best_acc
        if is_best:
            best_acc = test_acc

        state = {
            "epoch": epoch,
            "model": net.module.state_dict() if isinstance(net, nn.DataParallel) else net.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "best_acc": best_acc,
            "rng_state": torch.get_rng_state(),
            "args": vars(args),
        }
        if is_best or is_last:
            tag = "best" if is_best else "last"
            save_checkpoint(state, f"{args.name}_{tag}", args.seed)

        with open(logname, "a", newline="") as f:
            w = csv.writer(f)
            w.writerow([epoch, f"{train_loss:.6f}", f"{reg_loss:.6f}", f"{train_acc:.4f}",
                        f"{test_loss:.6f}", f"{test_acc:.4f}"])

    # Optional final validation pass for COMMAND
    if valloader is not None:
        print("\n==> Final validation (COMMAND)")
        val_loss, val_acc = evaluate(args.epoch - 1, net, valloader, criterion, device)
        with open(logname, "a", newline="") as f:
            w = csv.writer(f)
            w.writerow(["final_val", "", "", "", f"{val_loss:.6f}", f"{val_acc:.4f}"])

    print(f"\nDone. Last test acc: {last_test_acc:.3f}%  | Best test acc: {best_acc:.3f}%")
    print(f"Logs: {logname}")


if __name__ == "__main__":
    main()
