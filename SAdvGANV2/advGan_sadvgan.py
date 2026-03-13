"""
advGan_sadvgan.py  (v2 – stability fixes)
------------------------------------------
แก้ไขจาก v1:
  1. ใช้ BCE loss แทน log(sigmoid) โดยตรง → ป้องกัน NaN
  2. Label smoothing บน D → ป้องกัน D collapse
  3. Train D ทุก n_d_steps ครั้ง แต่ train G ทุก step → balance D/G
  4. Gradient clipping ทั้ง D และ G
  5. LR ของ D ต่ำกว่า G (0.0001 vs 0.0002)
  6. ตรวจ NaN → skip batch ถ้าเจอ
  7. loss_adv ใช้ mean แทน sum → ค่าไม่พุ่ง
"""

import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from PIL import Image

import models_sadvgan as models


def weights_init(m: nn.Module):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)


def ragan_disc_loss(C_real: torch.Tensor, C_fake: torch.Tensor) -> torch.Tensor:
    """RaGAN D loss with label smoothing (real=0.9, fake=0.1)"""
    C_real_mean = C_real.mean().detach()
    C_fake_mean = C_fake.mean().detach()
    loss = (
        F.binary_cross_entropy_with_logits(
            C_real - C_fake_mean, torch.full_like(C_real, 0.9))
        + F.binary_cross_entropy_with_logits(
            C_fake - C_real_mean, torch.full_like(C_fake, 0.1))
    ) * 0.5
    return loss


def ragan_gen_loss(C_real: torch.Tensor, C_fake: torch.Tensor) -> torch.Tensor:
    """RaGAN G loss"""
    C_real_mean = C_real.mean().detach()
    C_fake_mean = C_fake.mean().detach()
    loss = (
        F.binary_cross_entropy_with_logits(
            C_fake - C_real_mean, torch.ones_like(C_fake))
        + F.binary_cross_entropy_with_logits(
            C_real - C_fake_mean, torch.zeros_like(C_real))
    ) * 0.5
    return loss


class SAdvGAN_Attack:
    def __init__(
        self,
        device,
        model,
        model_num_labels: int,
        image_nc: int,
        box_min: float,
        box_max: float,
        eps: float = 16.0,
        save_path: str = 'output_sadvgan/',
        c: float = 0.3,
        alpha: float = 1.0,
        beta: float = 1.0,
        adv_lambda: float = 10.0,
        rec_lambda: float = 0.0,
        tv_lambda:  float = 0.0,
        noise_std: float = 0.1,
        n_d_steps: int = 2,
        epochs: int = 60,
    ):
        self.device           = device
        self.model            = model
        self.model_num_labels = model_num_labels
        self.image_nc         = image_nc
        self.box_min          = box_min
        self.box_max          = box_max
        self.eps              = eps / 255.0
        self.save_path        = save_path
        self.c                = c
        self.alpha            = alpha
        self.beta             = beta
        self.adv_lambda       = adv_lambda
        self.rec_lambda       = rec_lambda
        self.tv_lambda        = tv_lambda
        self.n_d_steps        = n_d_steps
        self.epochs           = epochs
        self._step            = 0

        os.makedirs(save_path, exist_ok=True)

        self.netG = models.NoiseFusionGenerator(
            in_channels=image_nc, out_channels=image_nc,
            feat_channels=32, noise_std=noise_std,
        ).to(device)

        self.netD = models.SelfAttentionDiscriminator(
            in_channels=image_nc, base_features=64,
        ).to(device)

        self.netG.apply(weights_init)
        self.netD.apply(weights_init)

        # D ใช้ LR ต่ำกว่า G เพื่อไม่ให้ D dominate เร็วเกินไป
        self.optimizer_G = torch.optim.RMSprop(self.netG.parameters(), lr=0.0002)
        self.optimizer_D = torch.optim.RMSprop(self.netD.parameters(), lr=0.0001)

    def _make_adv(self, x):
        raw = self.netG(x)
        if raw.shape != x.shape:
            raw = raw[:, :, :x.shape[2], :x.shape[3]]
        adv = torch.clamp(raw, -self.eps, self.eps) + x
        adv = torch.clamp(adv, self.box_min, self.box_max)
        pert = adv - x
        return pert, adv

    def generate(self, x):
        _, adv = self._make_adv(x)
        return adv

    def train_batch(self, x, labels):
        self._step += 1

        # ── Step 1: Update D (ทุก n_d_steps step, เฉพาะเมื่อ alpha > 0) ─
        loss_D_val = 0.0
        if self.alpha > 0.0 and self._step % self.n_d_steps == 0:
            with torch.no_grad():
                _, adv_det = self._make_adv(x)

            self.optimizer_D.zero_grad()
            C_real = self.netD(x)
            C_fake = self.netD(adv_det)
            loss_D = ragan_disc_loss(C_real, C_fake)

            if not torch.isnan(loss_D):
                loss_D.backward()
                torch.nn.utils.clip_grad_norm_(self.netD.parameters(), 1.0)
                self.optimizer_D.step()
                loss_D_val = loss_D.item()

        # ── Step 2: Update G (ทุก step) ─────────────────────────────────
        self.optimizer_G.zero_grad()

        pert, adv = self._make_adv(x)

        if self.alpha > 0.0:
            C_real       = self.netD(x).detach()
            C_fake       = self.netD(adv)
            loss_G_ragan = ragan_gen_loss(C_real, C_fake)
        else:
            loss_G_ragan = torch.tensor(0.0, device=x.device)

        pert_norm  = torch.norm(pert.view(pert.shape[0], -1), p=2, dim=1)
        loss_hinge = torch.mean(F.relu(pert_norm - self.c))

        # L_adv: C&W logit-based loss (paper formulation)
        # relu(logit_y - max_other + kappa) → minimize = attack until margin < kappa
        # ∂loss/∂logit = 1 (constant) เมื่อ active ← gradient ไม่ biased ต่อ pixel ไหน
        # → perturbation กระจายสม่ำเสมอทั่วภาพ (ต่างจาก -CE ที่ focus digit region)
        logits      = self.model(adv)
        logits      = torch.clamp(logits, -50.0, 50.0)   # ป้องกัน logit explosion → NaN ใน attention
        one_hot     = F.one_hot(labels, logits.shape[1]).float()
        logit_y     = (logits * one_hot).sum(dim=1)
        mask        = F.one_hot(labels, logits.shape[1]).bool()
        logit_other = logits.masked_fill(mask, -1e9).max(dim=1).values
        kappa       = self.adv_lambda
        loss_adv    = torch.clamp(logit_y - logit_other + kappa, min=0).mean()

        # L_rec: MSE (ควบคุมขนาด perturbation โดยรวม)
        loss_rec  = F.mse_loss(adv, x)

        # L_tv: Total Variation loss บน perturbation (ไม่ใช่ adv)
        # TV ลงโทษ spatial discontinuity → ต่อต้าน stripe artifacts
        # pert = adv - x (actual perturbation ที่ใส่เข้าไป)
        diff_h = pert[:, :, :, 1:] - pert[:, :, :, :-1]   # horizontal diff (แนวตั้ง stripe)
        diff_v = pert[:, :, 1:, :] - pert[:, :, :-1, :]   # vertical diff
        loss_tv = diff_h.abs().mean() + diff_v.abs().mean()
        loss_spread = loss_tv   # reuse spread slot for logging

        loss_G = (
            loss_adv
            + self.alpha      * loss_G_ragan
            + self.beta       * loss_hinge
            + self.rec_lambda * loss_rec
            + self.tv_lambda  * loss_tv
        )

        if not torch.isnan(loss_G):
            loss_G.backward()
            torch.nn.utils.clip_grad_norm_(self.netG.parameters(), 5.0)
            self.optimizer_G.step()

        return loss_D_val, loss_G_ragan.item(), loss_hinge.item(), loss_adv.item(), loss_rec.item(), loss_spread.item()

    def train(self, train_dataloader, start_epoch: int = 1):
        for epoch in tqdm(range(start_epoch, self.epochs + 1)):

            if epoch == 50:
                for opt in [self.optimizer_G, self.optimizer_D]:
                    for g in opt.param_groups: g['lr'] *= 0.1
            if epoch == 80:
                for opt in [self.optimizer_G, self.optimizer_D]:
                    for g in opt.param_groups: g['lr'] *= 0.1

            sD, sG, sH, sA, sR, sT, n = 0., 0., 0., 0., 0., 0., 0

            for data in train_dataloader:
                images, labels = (data[0], data[2]) if len(data) == 3 else data
                images = images.to(self.device)
                labels = labels.to(self.device)
                try:
                    ld, lg, lh, la, lr, lt = self.train_batch(images, labels)
                    if any(torch.isnan(torch.tensor(v)) for v in [ld, lg, la, lr]):
                        continue  # skip NaN batch
                    sD += ld; sG += lg; sH += lh; sA += la; sR += lr; sT += lt; n += 1
                except RuntimeError as e:
                    err_str = str(e)
                    if 'nan' in err_str.lower():
                        print(f"[Warning] NaN batch, skipping")
                        continue
                    elif 'CUDA' in err_str:
                        print(f"[Warning] CUDA error at epoch {epoch}, saving & stopping")
                        d = os.path.join(self.save_path, 'saved_model')
                        os.makedirs(d, exist_ok=True)
                        torch.save(self.netG.state_dict(),
                                   os.path.join(d, f'netG_epoch_{epoch}_crash.pth'))
                        print(f"[Warning] Saved: netG_epoch_{epoch}_crash.pth")
                        return
                    raise

            print(f"Epoch {epoch:3d} | loss_D: {sD/n:.4f} | "
                  f"loss_G_RaGAN: {sG/n:.4f} | "
                  f"loss_hinge: {sH/n:.4f} | loss_adv: {sA/n:.4f} | loss_rec: {sR/n:.5f} | loss_spread: {sT/n:.5f}")

            if epoch % 10 == 0:
                d = os.path.join(self.save_path, 'saved_model')
                os.makedirs(d, exist_ok=True)
                torch.save(self.netG.state_dict(),
                           os.path.join(d, f'netG_epoch_{epoch}.pth'))

    def load_generator(self, path):
        self.netG.load_state_dict(torch.load(path, map_location=self.device))
        print(f"[SAdvGAN] Loaded generator from {path}")

    def save_image(self, images, names, output_dir):
        os.makedirs(output_dir, exist_ok=True)
        for i, name in enumerate(names):
            Image.fromarray(images[i].astype('uint8')).save(
                os.path.join(output_dir, name))

    def save(self, data_loader):
        self.netG.eval()
        with torch.no_grad():
            for images, images_ID, _ in tqdm(data_loader):
                images = images.to(self.device)
                adv_np = np.transpose(
                    self.generate(images).cpu().numpy(), (0,2,3,1)) * 255
                self.save_image(adv_np, images_ID, self.save_path)
        self.netG.train()