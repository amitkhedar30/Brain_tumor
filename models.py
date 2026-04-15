# ============================================================
#  models.py  –  UNet | ResUNet | TransUNet (2.5D, 4-class)
# ============================================================

import torch
import torch.nn as nn
import torch.nn.functional as F
from config import IN_CHANNELS, NUM_CLASSES, IMG_SIZE


# ════════════════════════════════════════════════════════════
#  SHARED BUILDING BLOCKS
# ════════════════════════════════════════════════════════════

class ConvBlock(nn.Module):
    """Two consecutive Conv-BN-ReLU layers."""
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )
    def forward(self, x): return self.block(x)


class ResBlock(nn.Module):
    """ConvBlock with residual (identity) shortcut."""
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = ConvBlock(in_ch, out_ch)
        self.skip = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 1, bias=False),
            nn.BatchNorm2d(out_ch),
        ) if in_ch != out_ch else nn.Identity()

    def forward(self, x):
        return F.relu(self.conv(x) + self.skip(x), inplace=True)


class UpBlock(nn.Module):
    """Bilinear up-sample + ConvBlock (used in UNet & ResUNet)."""
    def __init__(self, in_ch, skip_ch, out_ch, use_res=False):
        super().__init__()
        self.up   = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        block_cls = ResBlock if use_res else ConvBlock
        self.conv = block_cls(in_ch + skip_ch, out_ch)

    def forward(self, x, skip):
        x = self.up(x)
        x = torch.cat([x, skip], dim=1)
        return self.conv(x)


# ════════════════════════════════════════════════════════════
#  1.  STANDARD U-NET  (Baseline)
# ════════════════════════════════════════════════════════════

class UNet(nn.Module):
    """
    Classic U-Net: 4 encoder levels + bottleneck + 4 decoder levels.
    Input : [B, IN_CHANNELS, H, W]
    Output: [B, NUM_CLASSES, H, W]  (logits)
    """
    def __init__(self, in_ch=IN_CHANNELS, num_classes=NUM_CLASSES,
                 features=(32, 64, 128, 256)):
        super().__init__()
        self.encoders = nn.ModuleList()
        self.pools    = nn.ModuleList()
        ch = in_ch
        for f in features:
            self.encoders.append(ConvBlock(ch, f))
            self.pools.append(nn.MaxPool2d(2))
            ch = f

        self.bottleneck = ConvBlock(ch, ch * 2)
        ch = ch * 2

        self.decoders = nn.ModuleList()
        for f in reversed(features):
            self.decoders.append(UpBlock(ch, f, f, use_res=False))
            ch = f

        self.head = nn.Conv2d(ch, num_classes, 1)

    def forward(self, x):
        skips = []
        for enc, pool in zip(self.encoders, self.pools):
            x = enc(x); skips.append(x); x = pool(x)
        x = self.bottleneck(x)
        for dec, skip in zip(self.decoders, reversed(skips)):
            x = dec(x, skip)
        return self.head(x)


# ════════════════════════════════════════════════════════════
#  2.  RES-U-NET  (Strong Contender)
# ════════════════════════════════════════════════════════════

class ResUNet(nn.Module):
    """
    U-Net where every ConvBlock is replaced with a ResBlock.
    Residual skip connections fight vanishing gradients and help
    the model learn fine-grained tumour boundaries (low-contrast).
    """
    def __init__(self, in_ch=IN_CHANNELS, num_classes=NUM_CLASSES,
                 features=(32, 64, 128, 256)):
        super().__init__()
        self.encoders = nn.ModuleList()
        self.pools    = nn.ModuleList()
        ch = in_ch
        for f in features:
            self.encoders.append(ResBlock(ch, f))
            self.pools.append(nn.MaxPool2d(2))
            ch = f

        self.bottleneck = ResBlock(ch, ch * 2)
        ch = ch * 2

        self.decoders = nn.ModuleList()
        for f in reversed(features):
            self.decoders.append(UpBlock(ch, f, f, use_res=True))
            ch = f

        self.head = nn.Conv2d(ch, num_classes, 1)

    def forward(self, x):
        skips = []
        for enc, pool in zip(self.encoders, self.pools):
            x = enc(x); skips.append(x); x = pool(x)
        x = self.bottleneck(x)
        for dec, skip in zip(self.decoders, reversed(skips)):
            x = dec(x, skip)
        return self.head(x)


# ════════════════════════════════════════════════════════════
#  3.  TRANS-U-NET  (Absolute Higher / State-of-the-Art)
# ════════════════════════════════════════════════════════════

class TransformerBottleneck(nn.Module):
    """
    Lightweight Vision Transformer block injected at the U-Net bottleneck.
    Splits the feature map into patches, runs multi-head self-attention
    to capture GLOBAL context across the entire brain slice, then
    reshapes back to the original spatial feature map.
    """
    def __init__(self, embed_dim: int, num_heads: int = 4,
                 num_layers: int = 4, patch_size: int = 1):
        super().__init__()
        self.patch_size = patch_size
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=embed_dim * 4,
            dropout=0.1,
            batch_first=True,
            norm_first=True,          # Pre-LN for stable training
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        B, C, H, W = x.shape
        # Flatten spatial dims → sequence of tokens [B, H*W, C]
        tokens = x.flatten(2).permute(0, 2, 1)
        tokens = self.transformer(tokens)
        tokens = self.norm(tokens)
        # Reshape back to feature map [B, C, H, W]
        return tokens.permute(0, 2, 1).reshape(B, C, H, W)


class TransUNet(nn.Module):
    """
    Hybrid CNN + Transformer U-Net.
    - CNN encoder extracts local features
    - Transformer bottleneck captures global brain context
    - CNN decoder reconstructs precise segmentation masks
    """
    def __init__(self, in_ch=IN_CHANNELS, num_classes=NUM_CLASSES,
                 features=(32, 64, 128, 256),
                 tf_heads=4, tf_layers=4):
        super().__init__()

        # ── Encoder (CNN) ────────────────────────────────────
        self.encoders = nn.ModuleList()
        self.pools    = nn.ModuleList()
        ch = in_ch
        for f in features:
            self.encoders.append(ConvBlock(ch, f))
            self.pools.append(nn.MaxPool2d(2))
            ch = f

        # ── Bottleneck: CNN → Transformer → Conv ─────────────
        bottleneck_ch = ch * 2
        self.down_conv = ConvBlock(ch, bottleneck_ch)
        self.transformer = TransformerBottleneck(
            embed_dim=bottleneck_ch,
            num_heads=tf_heads,
            num_layers=tf_layers,
        )
        self.up_conv = ConvBlock(bottleneck_ch, bottleneck_ch)
        ch = bottleneck_ch

        # ── Decoder (CNN) ─────────────────────────────────────
        self.decoders = nn.ModuleList()
        for f in reversed(features):
            self.decoders.append(UpBlock(ch, f, f, use_res=True))
            ch = f

        self.head = nn.Conv2d(ch, num_classes, 1)

    def forward(self, x):
        skips = []
        for enc, pool in zip(self.encoders, self.pools):
            x = enc(x); skips.append(x); x = pool(x)

        # Global context via Transformer
        x = self.down_conv(x)
        x = self.transformer(x)
        x = self.up_conv(x)

        for dec, skip in zip(self.decoders, reversed(skips)):
            x = dec(x, skip)
        return self.head(x)


# ════════════════════════════════════════════════════════════
#  Factory helper
# ════════════════════════════════════════════════════════════

def get_model(name: str) -> nn.Module:
    name = name.lower()
    if name == "unet":      return UNet()
    if name == "resunet":   return ResUNet()
    if name == "transunet": return TransUNet()
    raise ValueError(f"Unknown model: {name}. Choose from unet | resunet | transunet")


# ── quick sanity check ───────────────────────────────────────
if __name__ == "__main__":
    x = torch.randn(2, IN_CHANNELS, IMG_SIZE, IMG_SIZE)
    for name in ("unet", "resunet", "transunet"):
        model = get_model(name)
        out   = model(x)
        params = sum(p.numel() for p in model.parameters()) / 1e6
        print(f"  {name:12s} | output: {tuple(out.shape)} | params: {params:.2f}M")
