"""
pretrain_target.py
------------------
Step 1: Train a target classifier on MNIST (or CIFAR-10) and save it.
Step 2: Pass the saved checkpoint to train_sadvgan.py via --target_ckpt

Usage:
    python pretrain_target.py --dataset mnist
    python pretrain_target.py --dataset cifar10
"""

import argparse
import os
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from tqdm import tqdm


# ---------------------------------------------------------------------------
# Models (same ones used in train_sadvgan.py)
# ---------------------------------------------------------------------------

class SmallCNN_MNIST(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),   # GlobalAvgPool: gradient = 1/(H*W) uniform
                                        # → ไม่มี periodic gradient structure
                                        # → G ไม่สามารถ exploit spatial periodicity
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),              # 128×1×1 → 128
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        return self.classifier(self.features(x))


class SmallCNN_CIFAR(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1),  nn.BatchNorm2d(64),  nn.ReLU(),
            nn.Conv2d(64, 128, 3, padding=1),nn.BatchNorm2d(128), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(128, 256, 3, padding=1),nn.BatchNorm2d(256),nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(256, 256, 3, padding=1),nn.BatchNorm2d(256),nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256 * 4 * 4, 512),
            nn.ReLU(),
            nn.Linear(512, num_classes),
        )

    def forward(self, x):
        return self.classifier(self.features(x))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset',    default='mnist', choices=['mnist', 'cifar10'])
    parser.add_argument('--epochs',     type=int, default=20)
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--lr',         type=float, default=0.001)
    parser.add_argument('--save_dir',   default='./checkpoints/')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'[Pretrain] Device: {device}  |  Dataset: {args.dataset.upper()}')

    os.makedirs(args.save_dir, exist_ok=True)

    # ----------------------------------------------------------------
    # Dataset
    # ----------------------------------------------------------------
    if args.dataset == 'mnist':
        transform = transforms.Compose([transforms.ToTensor()])
        train_set = torchvision.datasets.MNIST(
            './data', train=True,  download=True, transform=transform)
        test_set  = torchvision.datasets.MNIST(
            './data', train=False, download=True, transform=transform)
        model = SmallCNN_MNIST().to(device)
    else:
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465),
                                  (0.2023, 0.1994, 0.2010)),
        ])
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465),
                                  (0.2023, 0.1994, 0.2010)),
        ])
        train_set = torchvision.datasets.CIFAR10(
            './data', train=True,  download=True, transform=transform_train)
        test_set  = torchvision.datasets.CIFAR10(
            './data', train=False, download=True, transform=transform_test)
        model = SmallCNN_CIFAR().to(device)

    train_loader = DataLoader(train_set, batch_size=args.batch_size,
                              shuffle=True,  num_workers=2, pin_memory=True)
    test_loader  = DataLoader(test_set,  batch_size=args.batch_size,
                              shuffle=False, num_workers=2, pin_memory=True)

    # ----------------------------------------------------------------
    # Optimizer & scheduler
    # ----------------------------------------------------------------
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs)
    criterion = nn.CrossEntropyLoss()

    best_acc  = 0.0
    save_path = os.path.join(args.save_dir, f'target_{args.dataset}_best.pth')

    # ----------------------------------------------------------------
    # Training loop
    # ----------------------------------------------------------------
    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss, correct, total = 0.0, 0, 0

        for images, labels in tqdm(train_loader,
                                   desc=f'Epoch {epoch}/{args.epochs}',
                                   leave=False):
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            logits = model(images)
            loss   = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * images.size(0)
            correct    += (logits.argmax(1) == labels).sum().item()
            total      += images.size(0)

        scheduler.step()
        train_acc = 100.0 * correct / total

        # Evaluate on test set
        model.eval()
        t_correct, t_total = 0, 0
        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(device), labels.to(device)
                preds = model(images).argmax(1)
                t_correct += (preds == labels).sum().item()
                t_total   += images.size(0)
        test_acc = 100.0 * t_correct / t_total

        print(f'Epoch {epoch:3d} | '
              f'Train Loss: {total_loss/total:.4f} | '
              f'Train Acc: {train_acc:.2f}% | '
              f'Test Acc:  {test_acc:.2f}%')

        if test_acc > best_acc:
            best_acc = test_acc
            torch.save(model.state_dict(), save_path)
            print(f'  --> Best model saved  (test acc = {best_acc:.2f}%)')

    print(f'\n[Pretrain] Done!  Best test accuracy: {best_acc:.2f}%')
    print(f'[Pretrain] Checkpoint saved to: {save_path}')
    print(f'\n[Next step] Run SAdvGAN with:')
    print(f'  python train_sadvgan.py --dataset {args.dataset} '
          f'--epochs 60 --batch_size 256 --target_ckpt {save_path}')


if __name__ == '__main__':
    main()