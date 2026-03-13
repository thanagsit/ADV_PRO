import os
"""
train_sadvgan.py
----------------
Example training script for SAdvGAN on MNIST and CIFAR-10.

Usage
-----
# MNIST (semi-white box, target = Resnet32)
python train_sadvgan.py --dataset mnist --epochs 60

# CIFAR-10 (semi-white box, target = Resnet32)
python train_sadvgan.py --dataset cifar10 --epochs 60
"""

import argparse
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

from advGan_sadvgan import SAdvGAN_Attack


# ---------------------------------------------------------------------------
# Tiny target classifiers for demonstration
# (replace with your pretrained Resnet20/32/44 or DenseNet etc.)
# ---------------------------------------------------------------------------

class SmallCNN_MNIST(nn.Module):
    """A lightweight CNN that mimics a small ResNet for MNIST."""
    def __init__(self, num_classes=10):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),   # GlobalAvgPool: ∂output/∂pixel = 1/784 uniform
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        return self.classifier(self.features(x))


class SmallCNN_CIFAR(nn.Module):
    """A lightweight CNN for CIFAR-10."""
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
# Dataset configs
# ---------------------------------------------------------------------------

DATASET_CFG = {
    'mnist': dict(
        image_nc      = 1,
        box_min       = 0.0,
        box_max       = 1.0,
        eps           = 76.5,          # L-inf = 0.3 * 255  (paper: 0.3 on MNIST)
        c             = 0.3,           # hinge threshold
        num_classes   = 10,
        img_size      = 28,
    ),
    'cifar10': dict(
        image_nc      = 3,
        box_min       = 0.0,
        box_max       = 1.0,
        eps           = 8.0,           # L-inf = 8/255 * 255  (paper: 8 on CIFAR)
        c             = 8.0 / 255.0,   # hinge threshold
        num_classes   = 10,
        img_size      = 32,
    ),
}


def get_dataloader(dataset: str, batch_size: int = 128):
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])

    if dataset == 'mnist':
        train_set = torchvision.datasets.MNIST(
            root='./data', train=True, download=True, transform=transform
        )
    elif dataset == 'cifar10':
        train_set = torchvision.datasets.CIFAR10(
            root='./data', train=True, download=True, transform=transform
        )
    else:
        raise ValueError(f"Unknown dataset: {dataset}")

    return DataLoader(
        train_set, batch_size=batch_size, shuffle=True,
        num_workers=4, pin_memory=True, drop_last=True
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description='Train SAdvGAN')
    parser.add_argument('--dataset',     default='mnist',
                        choices=['mnist', 'cifar10'])
    parser.add_argument('--epochs',      type=int, default=60)
    parser.add_argument('--batch_size',  type=int, default=128)
    parser.add_argument('--save_path',   default='output_sadvgan/')
    parser.add_argument('--adv_lambda',  type=float, default=10.0,
                        help='Weight of adversarial loss')
    parser.add_argument('--rec_lambda',  type=float, default=0.0,
                        help='Weight of MSE reconstruction loss (improves SSIM)')
    parser.add_argument('--tv_lambda',   type=float, default=0.0,
                        help='Weight of Total Variation loss (smooths perturbation)')
    parser.add_argument('--eps',          type=float, default=None,
                        help='L-inf budget in 0-255 scale (overrides dataset default)')
    parser.add_argument('--resume_epoch',  type=int,   default=0,
                        help='Resume from this epoch checkpoint (0=train from scratch)')
    parser.add_argument('--alpha',       type=float, default=1.0,
                        help='Weight of RaGAN loss')
    parser.add_argument('--beta',        type=float, default=1.0,
                        help='Weight of hinge loss')
    parser.add_argument('--c',           type=float, default=None,
                        help='Hinge threshold (overrides dataset default)')
    parser.add_argument('--noise_std',   type=float, default=0.1,
                        help='Std of Gaussian noise in second AE branch')
    parser.add_argument('--target_ckpt', default=None,
                        help='Path to a pretrained target classifier .pth '
                             '(if None, trains a fresh small CNN)')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'[SAdvGAN] Device: {device}')

    cfg = DATASET_CFG[args.dataset]

    # ----------------------------------------------------------------
    # Target classifier
    # ----------------------------------------------------------------
    if args.dataset == 'mnist':
        target_model = SmallCNN_MNIST(num_classes=cfg['num_classes']).to(device)
    else:
        target_model = SmallCNN_CIFAR(num_classes=cfg['num_classes']).to(device)

    if args.target_ckpt is not None:
        target_model.load_state_dict(
            torch.load(args.target_ckpt, map_location=device)
        )
        print(f'[SAdvGAN] Loaded target model from {args.target_ckpt}')
    else:
        print('[SAdvGAN] WARNING: using randomly initialised target model. '
              'Pass --target_ckpt to use a pretrained classifier.')

    target_model.eval()
    for p in target_model.parameters():
        p.requires_grad_(False)

    # ----------------------------------------------------------------
    # Data
    # ----------------------------------------------------------------
    loader = get_dataloader(args.dataset, batch_size=args.batch_size)

    # ----------------------------------------------------------------
    # SAdvGAN
    # ----------------------------------------------------------------
    attack = SAdvGAN_Attack(
        device          = device,
        model           = target_model,
        model_num_labels= cfg['num_classes'],
        image_nc        = cfg['image_nc'],
        box_min         = cfg['box_min'],
        box_max         = cfg['box_max'],
        eps             = args.eps if args.eps is not None else cfg['eps'],
        save_path       = args.save_path,
        c               = args.c if args.c is not None else cfg['c'],
        alpha           = args.alpha,
        beta            = args.beta,
        adv_lambda      = args.adv_lambda,
        rec_lambda      = args.rec_lambda,
        tv_lambda       = args.tv_lambda,
        noise_std       = args.noise_std,
        epochs          = args.epochs,
    )

    print(f'[SAdvGAN] Starting training on {args.dataset.upper()} '
          f'for {args.epochs} epochs …')
    if args.resume_epoch > 0:
        ckpt = os.path.join(args.save_path, 'saved_model', f'netG_epoch_{args.resume_epoch}.pth')
        attack.load_generator(ckpt)
        print(f'[Resume] Loaded generator from epoch {args.resume_epoch}')
    attack.train(loader, start_epoch=args.resume_epoch + 1)
    print('[SAdvGAN] Training complete.')


if __name__ == '__main__':
    main()