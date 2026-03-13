"""
evaluate_sadvgan.py
-------------------
วัด Attack Success Rate ของ SAdvGAN บน MNIST
- โหลด generator checkpoint
- สร้าง adversarial images
- วัด: accuracy ของ target model บน adv images (ยิ่งต่ำ = โจมตีดี)
- วัด: SSIM / L-inf / L2 ของ perturbation
- บันทึก adversarial image ตัวอย่าง

Usage:
    python evaluate_sadvgan.py \
        --target_ckpt ./checkpoints/target_mnist_best.pth \
        --gen_ckpt    ./output_sadvgan/saved_model/netG_epoch_60.pth \
        --dataset     mnist
"""

import argparse
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from tqdm import tqdm
from PIL import Image

import models_sadvgan as models


# ── Target model (เหมือนใน train_sadvgan.py) ─────────────────────────────

class SmallCNN_MNIST(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),   # GlobalAvgPool: gradient uniform → no stripes
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128, 256), nn.ReLU(),
            nn.Linear(256, num_classes),
        )
    def forward(self, x):
        return self.classifier(self.features(x))


class SmallCNN_CIFAR(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1),   nn.BatchNorm2d(64),  nn.ReLU(),
            nn.Conv2d(64, 128, 3, padding=1),  nn.BatchNorm2d(128), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(128, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(256, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256 * 4 * 4, 512), nn.ReLU(),
            nn.Linear(512, num_classes),
        )
    def forward(self, x):
        return self.classifier(self.features(x))


# ── Generator wrapper ─────────────────────────────────────────────────────

def make_adv(netG, x, eps, box_min=0.0, box_max=1.0):
    raw = netG(x)
    if raw.shape != x.shape:
        raw = raw[:, :, :x.shape[2], :x.shape[3]]
    adv = torch.clamp(raw, -eps, eps) + x
    adv = torch.clamp(adv, box_min, box_max)
    pert = adv - x   # actual perturbation applied (ไม่ใช่ raw output)
    return pert, adv


# ── Metrics ───────────────────────────────────────────────────────────────

def linf_norm(pert):
    return pert.abs().max().item()

def l2_norm_mean(pert):
    b = pert.shape[0]
    return torch.norm(pert.view(b, -1), p=2, dim=1).mean().item()

def ssim_approx(x, adv):
    """Simplified per-batch SSIM approximation (no external library needed)."""
    mu_x   = x.mean()
    mu_adv = adv.mean()
    sigma_x   = x.var()
    sigma_adv = adv.var()
    sigma_xy  = ((x - mu_x) * (adv - mu_adv)).mean()
    c1, c2 = 0.01**2, 0.03**2
    ssim = ((2*mu_x*mu_adv + c1) * (2*sigma_xy + c2)) / \
           ((mu_x**2 + mu_adv**2 + c1) * (sigma_x + sigma_adv + c2))
    return ssim.item()


# ── Save sample images ────────────────────────────────────────────────────

def save_comparison(orig, adv, out_dir, n=10):
    """Save all n pairs as a SINGLE combined grid image.
    Layout: n columns × 2 rows (top=original, bottom=adversarial)
    with labels and a metric-style header.
    """
    from PIL import ImageDraw, ImageFont
    os.makedirs(out_dir, exist_ok=True)

    SCALE   = 5          # upscale each 28×28 → 140×140
    PAD     = 8          # gap between cells
    HEADER  = 32         # top label area height
    FOOTER  = 20         # bottom metric area height
    ROW_LBL = 20         # row label height

    orig_np = orig[:n].cpu().numpy()
    adv_np  = adv[:n].cpu().numpy()

    def to_rgb(arr):  # (C,H,W) float → (H,W,3) uint8
        img = (arr.transpose(1,2,0) * 255).clip(0,255).astype(np.uint8)
        if img.shape[2] == 1:
            img = np.repeat(img, 3, axis=2)
        return img

    h, w = orig_np.shape[2], orig_np.shape[3]
    cw, ch = w * SCALE, h * SCALE

    total_w = PAD + n * (cw + PAD)
    total_h = HEADER + ROW_LBL + ch + PAD + ROW_LBL + ch + PAD + FOOTER

    canvas = Image.new('RGB', (total_w, total_h), (16, 16, 22))
    draw   = ImageDraw.Draw(canvas)

    try:
        font_hd = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf', 14)
        font_sm = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf', 11)
    except:
        font_hd = font_sm = ImageFont.load_default()

    # Row labels
    draw.text((PAD, HEADER + 3),                          'Original',     fill=(100,180,255), font=font_hd)
    draw.text((PAD, HEADER + ROW_LBL + ch + PAD + 3),    'Adversarial',  fill=(255,140,100), font=font_hd)

    for i in range(n):
        o = Image.fromarray(to_rgb(orig_np[i])).resize((cw, ch), Image.NEAREST)
        a = Image.fromarray(to_rgb(adv_np[i])).resize((cw, ch), Image.NEAREST)

        x = PAD + i * (cw + PAD)
        y_orig = HEADER + ROW_LBL
        y_adv  = HEADER + ROW_LBL + ch + PAD + ROW_LBL

        canvas.paste(o, (x, y_orig))
        canvas.paste(a, (x, y_adv))

        # Thin separator between pairs
        if i < n - 1:
            sep_x = x + cw + PAD // 2
            draw.line([(sep_x, HEADER), (sep_x, total_h - FOOTER - 2)], fill=(45,45,55), width=1)

    # Footer
    draw.text((PAD, total_h - FOOTER + 4),
              '* = adversarial example  |  visually identical, classifier fooled',
              fill=(80, 80, 90), font=font_sm)

    out_path = os.path.join(out_dir, 'comparison_grid.png')
    canvas.save(out_path)
    print(f"[Eval] Combined grid saved → {out_path}")


# ── Main ──────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset',     default='mnist', choices=['mnist','cifar10'])
    parser.add_argument('--target_ckpt', required=True)
    parser.add_argument('--gen_ckpt',    required=True)
    parser.add_argument('--batch_size',  type=int, default=256)
    parser.add_argument('--output_dir',  default='./eval_output/')
    parser.add_argument('--eps',          type=float, default=None,
                        help='L-inf budget in 0-255 scale (overrides dataset default)')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'[Eval] Device: {device}')

    # ── Config ────────────────────────────────────────────────────────────
    if args.dataset == 'mnist':
        image_nc, num_classes = 1, 10
        eps = (args.eps if args.eps is not None else 76.5) / 255.0
        box_min, box_max = 0.0, 1.0
        transform = transforms.ToTensor()
        test_set = torchvision.datasets.MNIST(
            './data', train=False, download=True, transform=transform)
        target_model = SmallCNN_MNIST(num_classes).to(device)
    else:
        image_nc, num_classes = 3, 10
        eps = 8.0 / 255.0
        box_min, box_max = 0.0, 1.0
        transform = transforms.ToTensor()
        test_set = torchvision.datasets.CIFAR10(
            './data', train=False, download=True, transform=transform)
        target_model = SmallCNN_CIFAR(num_classes).to(device)

    test_loader = DataLoader(test_set, batch_size=args.batch_size,
                             shuffle=False, num_workers=2)

    # ── Load models ───────────────────────────────────────────────────────
    target_model.load_state_dict(torch.load(args.target_ckpt, map_location=device))
    target_model.eval()
    print(f'[Eval] Target model loaded from {args.target_ckpt}')

    netG = models.NoiseFusionGenerator(
        in_channels=image_nc, out_channels=image_nc,
        feat_channels=32, noise_std=0.0,
    ).to(device)
    netG.load_state_dict(torch.load(args.gen_ckpt, map_location=device))
    netG.eval()
    print(f'[Eval] Generator loaded from {args.gen_ckpt}')

    # ── Evaluate ──────────────────────────────────────────────────────────
    orig_correct = 0   # correctly classified on clean images
    adv_correct  = 0   # still correct on adversarial (lower = better attack)
    total        = 0
    linf_list, l2_list, ssim_list = [], [], []
    saved = False

    with torch.no_grad():
        for images, labels in tqdm(test_loader, desc='Evaluating'):
            images, labels = images.to(device), labels.to(device)

            # Clean accuracy
            out_clean = target_model(images)
            orig_correct += (out_clean.argmax(1) == labels).sum().item()

            # Adversarial
            pert, adv = make_adv(netG, images, eps, box_min, box_max)
            out_adv  = target_model(adv)
            adv_correct += (out_adv.argmax(1) == labels).sum().item()

            # Metrics
            linf_list.append(linf_norm(pert))
            l2_list.append(l2_norm_mean(pert))
            ssim_list.append(ssim_approx(images, adv))

            total += images.size(0)

            if not saved:
                save_comparison(images, adv, args.output_dir)
                saved = True

    # ── Report ────────────────────────────────────────────────────────────
    clean_acc   = 100.0 * orig_correct / total
    adv_acc     = 100.0 * adv_correct  / total
    attack_rate = clean_acc - adv_acc

    print("\n" + "="*50)
    print(f"  Dataset       : {args.dataset.upper()}")
    print(f"  Total samples : {total}")
    print("-"*50)
    print(f"  Clean Accuracy    : {clean_acc:.2f}%")
    print(f"  Adv Accuracy      : {adv_acc:.2f}%   (lower = better attack)")
    print(f"  Attack Success    : {attack_rate:.2f}%  (= clean - adv)")
    print("-"*50)
    print(f"  Mean L-inf  : {np.mean(linf_list):.4f}  (budget={eps:.4f})")
    print(f"  Mean L2     : {np.mean(l2_list):.4f}")
    print(f"  Mean SSIM   : {np.mean(ssim_list):.4f}  (1.0=identical)")
    print("="*50)

    # Save results to txt
    os.makedirs(args.output_dir, exist_ok=True)
    with open(os.path.join(args.output_dir, 'results.txt'), 'w') as f:
        f.write(f"Clean Accuracy : {clean_acc:.2f}%\n")
        f.write(f"Adv Accuracy   : {adv_acc:.2f}%\n")
        f.write(f"Attack Success : {attack_rate:.2f}%\n")
        f.write(f"Mean L-inf     : {np.mean(linf_list):.4f}\n")
        f.write(f"Mean L2        : {np.mean(l2_list):.4f}\n")
        f.write(f"Mean SSIM      : {np.mean(ssim_list):.4f}\n")
    print(f"[Eval] Results saved to {args.output_dir}/results.txt")


if __name__ == '__main__':
    main()