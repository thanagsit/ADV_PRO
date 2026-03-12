import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as T
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"[INFO] Device: {device}")
if device == 'cuda':
    print(f"[INFO] GPU: {torch.cuda.get_device_name(0)}")

# ════════════════════════════════════════════════════════════════════════
#  1.  DCT / IDCT  (Eq.4, 5)
# ════════════════════════════════════════════════════════════════════════

def dct_1d(x: torch.Tensor) -> torch.Tensor:
    N = x.shape[-1]
    x_v = torch.cat([x[..., ::2], x[..., 1::2].flip([-1])], dim=-1)
    Vc = torch.view_as_real(torch.fft.fft(x_v, dim=-1))
    k = torch.arange(N, dtype=x.dtype, device=x.device).expand_as(x)
    W_r = torch.cos(np.pi * k / (2 * N))
    W_i = -torch.sin(np.pi * k / (2 * N))
    V = Vc[..., 0] * W_r - Vc[..., 1] * W_i
    return 2 * V

def idct_1d(X: torch.Tensor) -> torch.Tensor:
    N = X.shape[-1]
    X_v = X / 2.0
    k = torch.arange(N, dtype=X.dtype, device=X.device).expand_as(X)
    W_r = torch.cos(np.pi * k / (2 * N))
    W_i = torch.sin(np.pi * k / (2 * N))
    V_t_r = X_v * W_r - 0 * W_i
    V_t_i = X_v * W_i + 0 * W_r
    V_i = torch.cat([V_t_i[..., :1] * 0, -X_v.flip([-1])[..., :-1]], dim=-1)
    V = torch.view_as_complex(torch.stack([V_t_r, V_i], dim=-1))
    v = torch.fft.irfft(V, n=N, dim=-1)
    res = torch.zeros_like(v)
    res[..., ::2] = v[..., :N - N // 2]
    res[..., 1::2] = v[..., :N // 2].flip([-1])
    return res

def dct_2d(x: torch.Tensor) -> torch.Tensor:
    return dct_1d(dct_1d(x).transpose(-1, -2)).transpose(-1, -2)

def idct_2d(x: torch.Tensor) -> torch.Tensor:
    return idct_1d(idct_1d(x).transpose(-1, -2)).transpose(-1, -2)

# ════════════════════════════════════════════════════════════════════════
#  2.  Generator + Discriminator
# ════════════════════════════════════════════════════════════════════════

def weights_init(m):
    if isinstance(m, nn.Conv2d):
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif isinstance(m, nn.BatchNorm2d):
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)

class ResnetBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.block = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(dim, dim, 3, bias=False),
            nn.BatchNorm2d(dim), nn.ReLU(True),
            nn.ReflectionPad2d(1),
            nn.Conv2d(dim, dim, 3, bias=False),
            nn.BatchNorm2d(dim),
        )
    def forward(self, x): return x + self.block(x)

class Generator(nn.Module):
    def __init__(self, in_nc, out_nc):
        super().__init__()
        self.enc = nn.Sequential(
            nn.Conv2d(in_nc, 16, 3, 1, 1), nn.InstanceNorm2d(16), nn.ReLU(),
            nn.Conv2d(16, 32, 3, 2, 1), nn.InstanceNorm2d(32), nn.ReLU(),
        )
        self.bottleneck = nn.Sequential(*[ResnetBlock(32) for _ in range(4)])
        self.dec = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(32, 16, 3, 1, 1), nn.InstanceNorm2d(16), nn.ReLU(),
            nn.Conv2d(16, out_nc, 3, 1, 1), nn.Tanh(),
        )
    def forward(self, x): return self.dec(self.bottleneck(self.enc(x)))

class Discriminator(nn.Module):
    def __init__(self, nc):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(nc, 16, 4, 2, 1), nn.LeakyReLU(0.2),
            nn.Conv2d(16, 32, 4, 2, 1), nn.BatchNorm2d(32), nn.LeakyReLU(0.2),
            nn.Conv2d(32, 1, 4, 1, 0), nn.Sigmoid(),
        )
    def forward(self, x): return self.net(x).view(-1)

# ════════════════════════════════════════════════════════════════════════
#  3.  AdvGAN-G Attack
# ════════════════════════════════════════════════════════════════════════

class AdvGAN_G:
    def __init__(self, model, eps=0.25, adv_lambda=15.0, alpha=0.1, beta=0.001, sigma=0.02):
        self.model = model.eval()
        self.eps, self.adv_lambda = eps, adv_lambda
        self.alpha, self.beta, self.sigma = alpha, beta, sigma
        
        self.netG = Generator(1, 1).to(device)
        self.netD = Discriminator(1).to(device)
        self.netG.apply(weights_init)
        self.netD.apply(weights_init)
        
        self.lr = 1e-3
        self.opt_G = torch.optim.Adam(self.netG.parameters(), lr=self.lr)
        self.opt_D = torch.optim.Adam(self.netD.parameters(), lr=self.lr)

    def train_step(self, x, y):
        G_x = self.netG(x)
        pert_clean = torch.clamp(G_x, -self.eps, self.eps)
        x_adv_clean = torch.clamp(x + pert_clean, 0, 1)

        self.opt_D.zero_grad()
        d_real = self.netD(x)
        d_fake = self.netD(x_adv_clean.detach())
        
        if d_real.mean().item() < 0.95 or d_fake.mean().item() > 0.05:
            loss_D = F.mse_loss(d_real, torch.ones_like(d_real)) + F.mse_loss(d_fake, torch.zeros_like(d_fake))
            loss_hinge = (F.relu(1.0 - d_real) + F.relu(1.0 + d_fake)).mean()
            (loss_D + self.beta * loss_hinge).backward()
            self.opt_D.step()
            loss_D_val = loss_D.item()
        else:
            loss_D_val = 0.0

        self.opt_G.zero_grad()
        
        x_dct = dct_2d(x_adv_clean)
        eps_det = torch.normal(mean=1.0, std=self.sigma, size=x_dct.shape).to(device)
        x_adv_noisy = torch.clamp(idct_2d(x_dct * eps_det), 0, 1)
        
        logits_clean = self.model(x_adv_clean)
        logits_noisy = self.model(x_adv_noisy)
        
        loss_f_adv = - (0.7 * F.cross_entropy(logits_clean, y) + 0.3 * F.cross_entropy(logits_noisy, y))
        
        loss_GAN = F.mse_loss(self.netD(x_adv_clean), torch.ones_like(d_fake))
        loss_perturb = torch.norm(pert_clean.view(x.size(0), -1), p=2, dim=1).mean()
        
        tv_h = torch.sum(torch.abs(pert_clean[:, :, 1:, :] - pert_clean[:, :, :-1, :]))
        tv_w = torch.sum(torch.abs(pert_clean[:, :, :, 1:] - pert_clean[:, :, :, :-1]))
        loss_tv = (tv_h + tv_w) / (x.size(0) * x.size(1) * x.size(2) * x.size(3))
        
        loss_G = self.adv_lambda * loss_f_adv + self.alpha * loss_GAN + (1.5 * loss_perturb) + (2.0 * loss_tv)
        loss_G.backward()
        self.opt_G.step()
        
        return loss_D_val, loss_GAN.item(), loss_perturb.item(), loss_f_adv.item()

    def train(self, loader, epochs=100):
        print(f"\n[INFO] Training AdvGAN-G Generator for {epochs} epochs...")
        for epoch in range(1, epochs + 1):
            if epoch == int(epochs * 0.5):
                for param_group in self.opt_G.param_groups: param_group['lr'] = 1e-4
                for param_group in self.opt_D.param_groups: param_group['lr'] = 1e-4
            elif epoch == int(epochs * 0.8):
                for param_group in self.opt_G.param_groups: param_group['lr'] = 1e-5
                for param_group in self.opt_D.param_groups: param_group['lr'] = 1e-5
                
            sum_d = sum_gan = sum_p = sum_f = 0
            for x, y in loader:
                d, gan, p, f = self.train_step(x.to(device), y.to(device))
                sum_d += d; sum_gan += gan; sum_p += p; sum_f += f
            
            if epoch % 5 == 0 or epoch == 1:
                n = len(loader)
                print(f"[{epoch:3d}/{epochs}] D:{sum_d/n:.3f} GAN:{sum_gan/n:.3f} Pert:{sum_p/n:.4f} f_adv:{sum_f/n:.2f}")

    @torch.no_grad()
    def generate(self, x):
        self.netG.eval()
        G_x = self.netG(x.to(device))
        pert = torch.clamp(G_x, -self.eps, self.eps)
        return torch.clamp(x.to(device) + pert, 0, 1)

# ════════════════════════════════════════════════════════════════════════
#  4.  Target Models สำหรับจำลองสถานการณ์ White-box และ Black-box
# ════════════════════════════════════════════════════════════════════════

# 1. Source Model (White-box: โมเดลหลักที่ Generator มองเห็น)
class SourceCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 32, 3), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3), nn.ReLU(), nn.MaxPool2d(2),
            nn.Flatten(), nn.Linear(64*5*5, 10)
        )
    def forward(self, x): return self.net(x)

# 2. Transfer Model 1 (Black-box: โครงสร้าง VGG ขนาดเล็กที่ Generator ไม่เคยเห็น)
class TransferVGG(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1), nn.ReLU(),
            nn.Conv2d(32, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Flatten(), nn.Linear(64*7*7, 256), nn.ReLU(), nn.Linear(256, 10)
        )
    def forward(self, x): return self.net(x)

# 3. Transfer Model 2 (Black-box: โครงสร้าง Multi-Layer Perceptron)
class TransferMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(28*28, 512), nn.ReLU(),
            nn.Linear(512, 256), nn.ReLU(),
            nn.Linear(256, 10)
        )
    def forward(self, x): return self.net(x)

def train_classifier(model, name, loader):
    print(f"[INFO] Pre-training {name}...")
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    model.train()
    for epoch in range(3): # เทรน 3 Epoch ก็แม่นยำ >98% แล้วสำหรับ MNIST
        for x, y in loader:
            opt.zero_grad()
            F.cross_entropy(model(x.to(device)), y.to(device)).backward()
            opt.step()
    model.eval()

# ════════════════════════════════════════════════════════════════════════
#  5.  Execution & Transferability Evaluation
# ════════════════════════════════════════════════════════════════════════

if __name__ == '__main__':
    tf = T.Compose([T.ToTensor()])
    train_loader = DataLoader(torchvision.datasets.MNIST('./data', train=True, download=True, transform=tf), 
                              batch_size=128, shuffle=True, num_workers=2, pin_memory=True)
    test_loader = DataLoader(torchvision.datasets.MNIST('./data', train=False, download=True, transform=tf), 
                             batch_size=128, shuffle=False, num_workers=2)

    # สร้างและเทรนโมเดลทั้ง 3 ตัว
    model_source = SourceCNN().to(device)
    model_transfer1 = TransferVGG().to(device)
    model_transfer2 = TransferMLP().to(device)
    
    train_classifier(model_source, "SourceCNN (White-box)", train_loader)
    train_classifier(model_transfer1, "TransferVGG (Black-box 1)", train_loader)
    train_classifier(model_transfer2, "TransferMLP (Black-box 2)", train_loader)
    
    # 🌟 ฝึก AdvGAN-G โดยให้โจมตีเฉพาะ SourceCNN เท่านั้น 🌟
    # ใช้ eps=0.25 (ค่ามาตรฐานในงานวิจัยส่วนใหญ่) เพื่อให้เห็น Transferability ชัดเจน
    attacker = AdvGAN_G(model_source, eps=0.25, adv_lambda=15.0, alpha=0.1, beta=0.001, sigma=0.02)
    attacker.train(train_loader, epochs=100)
    
    # === การประเมินผลประสิทธิภาพ (Transferability Evaluation) ===
    print("\n" + "="*50)
    print("      EVALUATING TRANSFERABILITY (ASR %)")
    print("="*50)
    
    metrics = {
        "SourceCNN": {"correct": 0, "fooled": 0, "model": model_source},
        "TransferVGG": {"correct": 0, "fooled": 0, "model": model_transfer1},
        "TransferMLP": {"correct": 0, "fooled": 0, "model": model_transfer2}
    }
    
    for x, y in test_loader:
        x, y = x.to(device), y.to(device)
        adv_x = attacker.generate(x) # สร้างภาพโจมตี (Generator โจมตีแค่ตัวเดียว)
        
        # ทดสอบการโจมตีกับโมเดลทั้ง 3 ตัว
        for name, m_data in metrics.items():
            model = m_data["model"]
            p_ori = model(x).argmax(1)
            p_adv = model(adv_x).argmax(1)
            
            # นับจำนวนที่โมเดลทายถูกตอนแรก (ความแม่นยำตั้งต้น)
            m_data["correct"] += (p_ori == y).sum().item()
            # นับจำนวนที่ตอนแรกทายถูก แต่โดนภาพ Adversarial หลอกให้ทายผิด (ASR)
            m_data["fooled"] += ((p_adv != y) & (p_ori == y)).sum().item()

    # พิมพ์ตารางผลลัพธ์
    print(f"{'Target Model':<25} | {'Attack Type':<15} | {'ASR (%)'}")
    print("-" * 55)
    for name, m_data in metrics.items():
        asr = (m_data["fooled"] / m_data["correct"]) * 100
        attack_type = "White-box" if name == "SourceCNN" else "Black-box"
        print(f"{name:<25} | {attack_type:<15} | {asr:>5.2f}%")
    print("="*50)

    # === เซฟรูปภาพตัวอย่าง ===
    print("\n[INFO] Saving sample images for visualization...")
    os.makedirs('results', exist_ok=True)
    x_viz, y_viz = next(iter(test_loader))
    x_viz, y_viz = x_viz[:5].to(device), y_viz[:5].to(device)
    adv_viz = attacker.generate(x_viz)
    
    p_ori_viz = model_source(x_viz).argmax(1)
    p_adv_viz = model_source(adv_viz).argmax(1)
    
    fig, axes = plt.subplots(2, 5, figsize=(12, 5))
    fig.suptitle("AdvGAN-G Transferability Test", fontsize=16)
    
    for i in range(5):
        ax = axes[0, i]
        ax.imshow(x_viz[i].cpu().squeeze().numpy(), cmap='gray')
        ax.set_title(f"Original\nTrue: {y_viz[i].item()}", color='green')
        ax.axis('off')
        
        ax = axes[1, i]
        ax.imshow(adv_viz[i].cpu().squeeze().numpy(), cmap='gray')
        ax.set_title(f"Adversarial\nSourceCNN Pred: {p_adv_viz[i].item()}", color='red' if p_adv_viz[i]!=y_viz[i] else 'green')
        ax.axis('off')
        
    plt.tight_layout()
    plt.savefig('results/adv_examples.png', bbox_inches='tight')