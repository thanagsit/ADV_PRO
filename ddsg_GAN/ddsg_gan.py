"""
DDSG-GAN — v8.1 (Sweet Spot + Graph Retain Bug Fix)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from collections import deque
import numpy as np
import matplotlib.pyplot as plt
import os

# ──────────────────────────────────────────────────────────────────────────────
# 1.  GENERATOR
# ──────────────────────────────────────────────────────────────────────────────

class Generator(nn.Module):
    def __init__(self, in_ch=1):
        super().__init__()
        # เอา inplace=True ออกเพื่อป้องกันบั๊ก Autograd
        self.enc = nn.Sequential(
            nn.Conv2d(in_ch, 64, 3, padding=1), nn.InstanceNorm2d(64), nn.ReLU(),
            nn.Conv2d(64,   64, 3, padding=1), nn.InstanceNorm2d(64), nn.ReLU(),
            nn.Conv2d(64,   64, 3, padding=1), nn.InstanceNorm2d(64), nn.ReLU(),
            nn.Conv2d(64,  128, 3, padding=1), nn.InstanceNorm2d(128),nn.ReLU(),
            nn.Conv2d(128, 128, 3, padding=1), nn.InstanceNorm2d(128),nn.ReLU(),
        )
        self.dec = nn.Sequential(
            nn.Conv2d(128, 128, 3, padding=1), nn.InstanceNorm2d(128),nn.ReLU(),
            nn.Conv2d(128,  64, 3, padding=1), nn.InstanceNorm2d(64), nn.ReLU(),
            nn.Conv2d(64,  in_ch,3,padding=1),
        )

    def forward(self, x):
        raw_delta = self.dec(self.enc(x))
        return torch.tanh(raw_delta)

def ste_clamp(x, min_val, max_val):
    return x + (x.clamp(min_val, max_val) - x).detach()

# ──────────────────────────────────────────────────────────────────────────────
# 2.  DISCRIMINATOR D1
# ──────────────────────────────────────────────────────────────────────────────

class DiscriminatorD1(nn.Module):
    def __init__(self, in_ch=1, num_classes=10):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32,   64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64,  128, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(128, 256, 3, padding=1), nn.ReLU(), nn.AdaptiveAvgPool2d(1),
        )
        self.fc = nn.Linear(256, num_classes)

    def forward(self, x):
        return self.fc(self.net(x).flatten(1))

# ──────────────────────────────────────────────────────────────────────────────
# 3.  DISCRIMINATOR D2
# ──────────────────────────────────────────────────────────────────────────────

class DiscriminatorD2(nn.Module):
    def __init__(self, in_ch=1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, 32, 3, stride=2, padding=1), nn.LeakyReLU(0.2),
            nn.Conv2d(32,   64, 3, stride=2, padding=1), nn.LeakyReLU(0.2),
            nn.Conv2d(64,  128, 3, stride=2, padding=1), nn.LeakyReLU(0.2),
            nn.AdaptiveAvgPool2d(1),
        )
        self.fc = nn.Linear(128, 1)

    def forward(self, x):
        return self.fc(self.net(x).flatten(1))

# ──────────────────────────────────────────────────────────────────────────────
# 4.  CIRCULAR QUEUE
# ──────────────────────────────────────────────────────────────────────────────

class CircularQueue:
    def __init__(self, maxlen=60001):
        self.q = deque(maxlen=maxlen)

    def add(self, xs, ys):
        for x, y in zip(xs.cpu(), ys.cpu()):
            self.q.append((x, y))

    def sample(self, n):
        if len(self.q) < n:
            return None, None
        idx = np.random.choice(len(self.q), n, replace=False)
        xs, ys = zip(*[self.q[i] for i in idx])
        return torch.stack(xs), torch.stack(ys)

    def __len__(self):
        return len(self.q)

# ──────────────────────────────────────────────────────────────────────────────
# 5.  C&W ATTACK LOSS
# ──────────────────────────────────────────────────────────────────────────────

def cw_loss(logits, y, targeted=False, kappa=0.0):
    B = logits.size(0)
    true_l = logits[range(B), y]
    others = logits.clone()
    others[range(B), y] = float('-inf')
    best_o = others.max(1).values

    if not targeted:
        return F.relu(true_l - best_o + kappa).mean()
    else:
        return F.relu(best_o - true_l + kappa).mean()

# ──────────────────────────────────────────────────────────────────────────────
# 6.  TRAINER
# ──────────────────────────────────────────────────────────────────────────────

class DDSGGANTrainer:
    def __init__(self, T, G, D1, D2,
                 eps=0.3, n=20, m=20, queue_max=60001,
                 beta1=2.0, beta2=1.0, n_g=5, d1_update_interval=10,
                 kappa=5.0, eta_g=0.001, eta_d1=0.0001, eta_d2=0.0001,
                 targeted=False, device='cpu'):

        self.T        = T
        self.G        = G.to(device)
        self.D1       = D1.to(device)
        self.D2       = D2.to(device)
        self.eps      = eps
        self.n        = n
        self.m        = m
        self.beta1    = beta1
        self.beta2    = beta2
        self.n_g      = n_g
        self.d1_upd   = d1_update_interval
        self.kappa    = kappa
        self.targeted = targeted
        self.device   = device
        self.queue    = CircularQueue(maxlen=queue_max)

        self.opt_G  = optim.Adam(G.parameters(),  lr=eta_g,  betas=(0.5, 0.999))
        self.opt_D1 = optim.Adam(D1.parameters(), lr=eta_d1, betas=(0.9, 0.999))
        self.opt_D2 = optim.Adam(D2.parameters(), lr=eta_d2, betas=(0.5, 0.999))

    def _xhat(self, x):
        delta = self.G(x) * self.eps
        return ste_clamp(x + delta, 0.0, 1.0)

    @torch.no_grad()
    def _query_T(self, x):
        return self.T(x.to(self.device)).argmax(1)
        
    def _set_requires_grad(self, nets, requires_grad=False):
        """ฟังก์ชันช่วยแช่แข็ง/ปลดล็อกโมเดล"""
        for net in nets:
            for param in net.parameters():
                param.requires_grad = requires_grad

    def pretrain_d1(self, loader, epochs=5):
        print(f"  Pre-training D1 for {epochs} epochs to match T...")
        self.D1.train()
        for ep in range(1, epochs + 1):
            total_loss, n = 0.0, 0
            for x, _ in loader:
                x = x.to(self.device)
                lbl = self._query_T(x)
                loss = F.cross_entropy(self.D1(x), lbl)
                self.opt_D1.zero_grad()
                loss.backward()
                self.opt_D1.step()
                total_loss += loss.item() * x.size(0)
                n += x.size(0)
            print(f"    D1 pretrain epoch {ep}/{epochs} — loss={total_loss/n:.4f}")

    def train_step(self, x, y, it):
        x, y = x.to(self.device), y.to(self.device)
        B = x.size(0)

        with torch.no_grad():
            x_buf = self._xhat(x)
        if it > self.n and it % self.m == 0:
            self.queue.add(x_buf, self._query_T(x_buf))

        # --- Update D1 ---
        ld1 = torch.tensor(0.0)
        if it % self.d1_upd == 0:
            self.D1.train()
            self.opt_D1.zero_grad()
            lbl_x = self._query_T(x)
            xq, yq = self.queue.sample(B)
            if xq is not None and it > self.n:
                xq, yq = xq.to(self.device), yq.to(self.device)
                ld1 = (self.beta1 * F.cross_entropy(self.D1(x), lbl_x) +
                       self.beta2 * F.cross_entropy(self.D1(xq.detach()), yq))
            else:
                ld1 = F.cross_entropy(self.D1(x), lbl_x)
            ld1.backward()
            self.opt_D1.step()

        # --- Update D2 ---
        self.D2.train()
        self.opt_D2.zero_grad()
        ones  = torch.ones (B, 1, device=self.device)
        zeros = torch.zeros(B, 1, device=self.device)
        ld2 = (F.binary_cross_entropy_with_logits(self.D2(x), ones) +
               F.binary_cross_entropy_with_logits(self.D2(x_buf.detach()), zeros))
        ld2.backward()
        self.opt_D2.step()

        # --- Update Generator ---
        # 💡 แช่แข็ง D1 และ D2 ไม่ให้เก็บค่า Gradient ตอนอัปเดต G
        self._set_requires_grad([self.D1, self.D2], False)
        self.D1.eval()
        
        lg_last, delta_last = 0.0, 0.0
        for _ in range(self.n_g):
            self.G.train()
            self.opt_G.zero_grad()
            x_hat   = self._xhat(x)
            logits  = self.D1(x_hat)
            
            # 1. Loss โจมตี
            lg_cw = cw_loss(logits, y, targeted=self.targeted, kappa=self.kappa)
            
            # 2. Loss ความเนียน
            ones_g = torch.ones(B, 1, device=self.device)
            lg_d2_for_g = F.binary_cross_entropy_with_logits(self.D2(x_hat), ones_g)
            
            # 3. รวม Loss (gamma2 = 0.01 สำหรับ Sweet Spot)
            gamma2 = 0.01 
            lg = lg_cw + (gamma2 * lg_d2_for_g)

            if not (torch.isnan(lg) or torch.isinf(lg)):
                lg.backward()
                torch.nn.utils.clip_grad_norm_(self.G.parameters(), max_norm=1.0)
                self.opt_G.step()
                lg_last    = lg.item()
                delta_last = x_hat.detach().sub(x).abs().mean().item()

        self.D1.train()
        # 💡 ปลดล็อกการแช่แข็ง เพื่อให้พร้อมอัปเดตในรอบถัดไป
        self._set_requires_grad([self.D1, self.D2], True)
        
        return dict(ld1=ld1.item() if isinstance(ld1, torch.Tensor) else ld1,
                    ld2=ld2.item(), lg=lg_last, delta=delta_last)

    def train(self, loader, epochs=50, log_every=200):
        it = 0
        for ep in range(1, epochs + 1):
            for bi, (x, y) in enumerate(loader):
                it += 1
                s = self.train_step(x, y, it)
                if bi % log_every == 0:
                    print(f"[{ep:3d}/{epochs}] batch {bi:4d}/{len(loader)} | "
                          f"L_D1={s['ld1']:.4f}  L_D2={s['ld2']:.4f}  "
                          f"L_G={s['lg']:.4f}  |δ|={s['delta']:.4f}  "
                          f"Q={len(self.queue)}")
        print("Training complete.")
        return self.G

# ──────────────────────────────────────────────────────────────────────────────
# 7.  EVALUATION & VISUALIZATION HELPERS
# ──────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def make_aes(G, loader, eps, device):
    G.eval()
    all_xhat, all_y = [], []
    for x, y in loader:
        x = x.to(device)
        delta = G(x) * eps
        xhat  = (x + delta).clamp(0.0, 1.0)
        all_xhat.append(xhat.cpu())
        all_y.append(y)
    return torch.cat(all_xhat), torch.cat(all_y)

@torch.no_grad()
def asr(T, xhat, y_true, targeted=False, device='cpu'):
    T.eval()
    preds = T(xhat.to(device)).argmax(1).cpu()
    return ((preds != y_true) if not targeted else (preds == y_true)).float().mean().item()

@torch.no_grad()
def surrogate_stats(D1, T, loader, device):
    D1.eval(); T.eval()
    c_d1 = sim = total = 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        pd1 = D1(x).argmax(1); pt = T(x).argmax(1)
        c_d1  += (pd1 == y).sum().item()
        sim   += (pd1 == pt).sum().item()
        total += y.size(0)
    return c_d1/total, sim/total

def save_model(G, filename="ddsg_generator_mnist.pth"):
    torch.save(G.state_dict(), filename)
    print(f"  [+] Model saved successfully to {filename}")

@torch.no_grad()
def visualize_aes(G, T, loader, eps, device, num_images=10, filename="adv_results.png"):
    G.eval()
    T.eval()

    x, y_true = next(iter(loader))
    x, y_true = x.to(device), y_true.to(device)

    x = x[:num_images]
    y_true = y_true[:num_images]

    delta = G(x) * eps
    x_hat = (x + delta).clamp(0.0, 1.0)

    pred_orig = T(x).argmax(1)
    pred_adv = T(x_hat).argmax(1)

    x_np = x.cpu().squeeze().numpy()
    x_hat_np = x_hat.cpu().squeeze().numpy()

    fig, axes = plt.subplots(2, num_images, figsize=(num_images * 2, 4.5))
    fig.suptitle("DDSG-GAN Adversarial Examples", fontsize=16)

    for i in range(num_images):
        axes[0, i].imshow(x_np[i], cmap='gray', vmin=0, vmax=1)
        axes[0, i].set_title(f"Orig: {pred_orig[i].item()}", color='blue')
        axes[0, i].axis('off')

        axes[1, i].imshow(x_hat_np[i], cmap='gray', vmin=0, vmax=1)
        color = 'red' if pred_orig[i].item() != pred_adv[i].item() else 'green'
        axes[1, i].set_title(f"Adv: {pred_adv[i].item()}", color=color)
        axes[1, i].axis('off')

    plt.tight_layout()
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    print(f"  [+] Visualization saved to {filename}")

# ──────────────────────────────────────────────────────────────────────────────
# 8.  TARGET MODEL (Model A)
# ──────────────────────────────────────────────────────────────────────────────

def model_A():
    return nn.Sequential(
        nn.Conv2d(1, 64, 5, padding=2), nn.ReLU(),
        nn.Conv2d(64, 64, 5, padding=2), nn.ReLU(),
        nn.Dropout(0.25), nn.Flatten(),
        nn.Linear(64*28*28, 128), nn.ReLU(),
        nn.Dropout(0.5), nn.Linear(128, 10))

# ──────────────────────────────────────────────────────────────────────────────
# 9.  DEMO RUN
# ──────────────────────────────────────────────────────────────────────────────

def demo_mnist(epochs=50, device='cpu'):
    print(f"\n=== DDSG-GAN on MNIST | device={device} ===\n")
    tf = transforms.ToTensor()
    tr = datasets.MNIST('./data', train=True,  download=True, transform=tf)
    te = datasets.MNIST('./data', train=False, download=True, transform=tf)
    tr_loader = DataLoader(tr, 128, shuffle=True,  num_workers=2, pin_memory=True)
    te_loader = DataLoader(te, 256, shuffle=False, num_workers=2, pin_memory=True)

    T = model_A().to(device)
    opt_t = optim.Adam(T.parameters(), lr=1e-3)
    sched = optim.lr_scheduler.StepLR(opt_t, step_size=3, gamma=0.5)
    print("Pre-training target model T (10 epochs)...")
    for ep in range(1, 11):
        T.train()
        for x, y in tr_loader:
            x, y = x.to(device), y.to(device)
            F.cross_entropy(T(x), y).backward()
            opt_t.step(); opt_t.zero_grad()
        sched.step()
        T.eval()
        with torch.no_grad():
            ok = sum((T(x.to(device)).argmax(1)==y.to(device)).sum().item() for x,y in te_loader)
        print(f"  T epoch {ep}/10 — test acc: {ok/len(te)*100:.2f}%")
    T.eval()
    for p in T.parameters(): p.requires_grad_(False)

    G  = Generator(in_ch=1)
    D1 = DiscriminatorD1(in_ch=1, num_classes=10)
    D2 = DiscriminatorD2(in_ch=1)

    trainer = DDSGGANTrainer(
        T=T, G=G, D1=D1, D2=D2,
        eps=0.3, n=20, m=20, queue_max=60001,
        beta1=2.0, beta2=1.0, n_g=5, d1_update_interval=10,
        kappa=5.0, eta_g=0.001, eta_d1=0.0001, eta_d2=0.0001,
        targeted=False, device=device,
    )

    print("\nPre-training D1 to match T...")
    trainer.pretrain_d1(tr_loader, epochs=5)

    acc, sim = surrogate_stats(D1, T, te_loader, device)
    print(f"  D1 after pretrain — Accuracy: {acc*100:.2f}%  Similarity: {sim*100:.2f}%\n")

    trainer.train(tr_loader, epochs=epochs, log_every=200)

    print("\nEvaluating surrogate model D1 ...")
    acc, sim = surrogate_stats(D1, T, te_loader, device)
    print(f"  D1 Accuracy: {acc*100:.2f}%   Similarity with T: {sim*100:.2f}%")

    print("\nComputing Attack Success Rate (untargeted) ...")
    xhat, y_true = make_aes(G, te_loader, eps=0.3, device=device)
    rate = asr(T, xhat, y_true, targeted=False, device=device)
    print(f"  Attack Success Rate (untargeted): {rate*100:.2f}%")
    
    print("\nSaving model and generating visualization...")
    save_model(G, filename="ddsg_generator_mnist.pth")
    visualize_aes(G, T, te_loader, eps=0.3, device=device, num_images=10, filename="adv_results.png")
    
    return G, D1

if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    demo_mnist(epochs=50, device=device)