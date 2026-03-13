"""
models_sadvgan.py
-----------------
SAdvGAN model architectures based on:
  "SAdvGAN: Multiple Information Fusion For Adversary Generation"
  Sun et al., ICTAI 2021

Components implemented:
  1. ResidualBlock         – shared building block
  2. SelfAttention         – SAGAN-style self-attention with spectral norm
  3. DeepAutoEncoder       – single encoder-decoder (ResNet backbone)
  4. NoiseFusionGenerator  – dual autoencoder + feature-fusion head (Sec. III-A)
  5. SelfAttentionDiscriminator – conv + self-attention + spectral norm  (Sec. III-B)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def sn_conv(in_c, out_c, kernel, stride=1, padding=0):
    """Conv2d wrapped with spectral normalisation."""
    return nn.utils.spectral_norm(
        nn.Conv2d(in_c, out_c, kernel, stride=stride, padding=padding)
    )


def sn_deconv(in_c, out_c, kernel, stride=1, padding=0):
    """ConvTranspose2d wrapped with spectral normalisation."""
    return nn.utils.spectral_norm(
        nn.ConvTranspose2d(in_c, out_c, kernel, stride=stride, padding=padding)
    )


# ---------------------------------------------------------------------------
# 1. Residual Block
# ---------------------------------------------------------------------------

class ResidualBlock(nn.Module):
    """Standard pre-activation residual block used inside the autoencoders."""

    def __init__(self, channels: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.relu(x + self.block(x))


# ---------------------------------------------------------------------------
# 2. Self-Attention Module  (SAGAN-style, Fig. 2 of the paper)
# ---------------------------------------------------------------------------

class SelfAttention(nn.Module):
    """
    Maps convolution feature maps → self-attention feature maps.

    f(x) = W_f x  (key)
    g(x) = W_g x  (query)
    h(x) = W_h x  (value)
    attention = softmax(f^T g)
    o_j = h * attention
    y_i = gamma * o_i + x_i      (gamma is learned, initialised to 0)

    All projections use spectral normalisation as in the discriminator.
    """

    def __init__(self, in_channels: int):
        super().__init__()
        inter = max(in_channels // 8, 1)
        half  = max(in_channels // 2, 1)

        self.f = sn_conv(in_channels, inter, 1)   # key
        self.g = sn_conv(in_channels, inter, 1)   # query
        self.h = sn_conv(in_channels, half,  1)   # value
        self.v = sn_conv(half, in_channels, 1)    # output projection
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.size()
        N = H * W

        # Key, Query, Value
        f = self.f(x).view(B, -1, N).permute(0, 2, 1)  # (B, N, C//8)
        g = self.g(x).view(B, -1, N)                    # (B, C//8, N)
        h = self.h(x).view(B, -1, N)                    # (B, C//2, N)

        # Attention map  (B, N, N)
        attn = F.softmax(torch.bmm(f, g), dim=-1)

        # Attended output  (B, C//2, N) → (B, C//2, H, W)
        o = torch.bmm(h, attn.permute(0, 2, 1)).view(B, -1, H, W)
        o = self.v(o)                                    # (B, C, H, W)

        return self.gamma * o + x


# ---------------------------------------------------------------------------
# 3. Deep AutoEncoder  (single branch of the generator)
# ---------------------------------------------------------------------------

class DeepAutoEncoder(nn.Module):
    """
    Encoder-decoder with residual blocks.
    Outputs a feature tensor of `feat_channels` channels (before Tanh),
    so that two branches can be fused before the final 1x1 projection.

    Architecture
    ------------
    Encoder : 3 × (Conv + BN + ReLU)  → 8 → 16 → feat_channels
    Bottleneck : 4 × ResidualBlock(feat_channels)
    Decoder : 3 × (ConvTranspose + BN + ReLU) → feat_channels → 16 → 8
              + final ConvTranspose → feat_channels  (raw features, no activation)
    """

    def __init__(self, in_channels: int, feat_channels: int = 32):
        super().__init__()

        # --- Encoder ---
        self.enc1 = nn.Sequential(
            nn.Conv2d(in_channels, 8, 3, padding=1, bias=False),
            nn.BatchNorm2d(8),
            nn.ReLU(inplace=True),
        )
        self.enc2 = nn.Sequential(
            nn.Conv2d(8, 16, 3, padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
        )
        self.enc3 = nn.Sequential(
            nn.Conv2d(16, feat_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(feat_channels),
            nn.ReLU(inplace=True),
        )

        # --- Bottleneck (residual blocks) ---
        self.bottleneck = nn.Sequential(
            ResidualBlock(feat_channels),
            ResidualBlock(feat_channels),
            ResidualBlock(feat_channels),
            ResidualBlock(feat_channels),
        )

        # --- Decoder ---
        self.dec1 = nn.Sequential(
            nn.ConvTranspose2d(feat_channels, 16, 3, padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
        )
        self.dec2 = nn.Sequential(
            nn.ConvTranspose2d(16, 8, 3, padding=1, bias=False),
            nn.BatchNorm2d(8),
            nn.ReLU(inplace=True),
        )
        # Raw feature output (no activation) – activation applied after fusion
        self.dec3 = nn.ConvTranspose2d(8, feat_channels, 3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.enc1(x)
        x = self.enc2(x)
        x = self.enc3(x)
        x = self.bottleneck(x)
        x = self.dec1(x)
        x = self.dec2(x)
        x = self.dec3(x)
        return x  # raw features, shape (B, feat_channels, H, W)


# ---------------------------------------------------------------------------
# 4. Noise Fusion Generator  (Sec. III-A)
# ---------------------------------------------------------------------------

class NoiseFusionGenerator(nn.Module):
    """
    Two-branch deep autoencoder generator with feature-level fusion.

    Branch 1 (ae1): processes the clean input image.
    Branch 2 (ae2): processes the input image with added Gaussian white noise.
                    The stacked autoencoder implicitly performs denoising,
                    introducing diversity into the perturbation.

    Both branches produce `feat_channels`-dimensional feature maps which are
    concatenated and projected to `out_channels` via a fusion head.

    Parameters
    ----------
    in_channels  : number of image channels (1 for grayscale, 3 for RGB)
    out_channels : perturbation channels (= in_channels)
    feat_channels: internal feature width of each autoencoder (default 32)
    noise_std    : std-dev of Gaussian noise added to the second branch input
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        feat_channels: int = 32,
        noise_std: float = 0.1,
    ):
        super().__init__()
        self.noise_std = noise_std

        self.ae1 = DeepAutoEncoder(in_channels,     feat_channels)
        self.ae2 = DeepAutoEncoder(in_channels,     feat_channels)

        # Fusion head: (2 * feat_channels) → out_channels
        self.fusion = nn.Sequential(
            nn.Conv2d(feat_channels * 2, feat_channels, 1, bias=False),
            nn.BatchNorm2d(feat_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(feat_channels, out_channels, 3, padding=1),
            # NOTE: NO Tanh here — clipping to [-eps, eps] is done externally.
            # Tanh was killing gradients because 94% of its range was being
            # clipped away (eps=16/255=0.063, Tanh range=[-1,1]).
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Branch 1: clean image
        feat1 = self.ae1(x)

        # Branch 2: noisy image  (Gaussian white noise)
        # Note: always apply noise, even at eval time, to maintain branch diversity.
        # Using a fixed seed at eval is NOT used — consistent noise std matches train.
        noise = torch.randn_like(x) * self.noise_std
        feat2 = self.ae2(x + noise)

        # Feature fusion and projection to perturbation space
        fused = torch.cat([feat1, feat2], dim=1)
        return self.fusion(fused)


# ---------------------------------------------------------------------------
# 5. Self-Attention Discriminator  (Sec. III-B)
# ---------------------------------------------------------------------------

class SelfAttentionDiscriminator(nn.Module):
    """
    Discriminator with spectral-normalised convolutions and a self-attention
    layer inserted after the second convolutional block (as in SAGAN).

    The output is a scalar (real / fake score) for use with the
    Relativistic average GAN (RaGAN) loss.

    Architecture
    ------------
    Conv+SN (in → 64)  → LeakyReLU
    Conv+SN (64 → 128) → BN → LeakyReLU
    SelfAttention(128)             ← key SAdvGAN addition
    Conv+SN (128 → 256)→ BN → LeakyReLU
    Conv+SN (256 → 512)→ BN → LeakyReLU
    AdaptiveAvgPool → Flatten → Linear → scalar
    """

    def __init__(self, in_channels: int, base_features: int = 64):
        super().__init__()

        self.block1 = nn.Sequential(
            sn_conv(in_channels,       base_features,     4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.block2 = nn.Sequential(
            sn_conv(base_features,     base_features * 2, 4, stride=2, padding=1),
            nn.BatchNorm2d(base_features * 2),
            nn.LeakyReLU(0.2, inplace=True),
        )

        # Self-attention inserted after block2
        self.attn = SelfAttention(base_features * 2)

        self.block3 = nn.Sequential(
            sn_conv(base_features * 2, base_features * 4, 4, stride=2, padding=1),
            nn.BatchNorm2d(base_features * 4),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.block4 = nn.Sequential(
            sn_conv(base_features * 4, base_features * 8, 4, stride=2, padding=1),
            nn.BatchNorm2d(base_features * 8),
            nn.LeakyReLU(0.2, inplace=True),
        )

        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(base_features * 8, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.block1(x)
        x = self.block2(x)
        x = self.attn(x)          # self-attention (SAdvGAN addition)
        x = self.block3(x)
        x = self.block4(x)
        x = self.pool(x).view(x.size(0), -1)
        return self.classifier(x)  # raw logit, no sigmoid (used by RaGAN)