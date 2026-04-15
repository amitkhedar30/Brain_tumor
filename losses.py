# ============================================================
#  losses.py  –  Dice Loss + Focal Loss for severe class imbalance
# ============================================================
"""
Why Dice + Focal Loss?
  - Dice Loss handles the macroscopic class imbalance by focusing on OVERLAP,
    penalizing missed tumor regions regardless of their size.
  - Focal Loss mathematically down-weights easy, highly-confident predictions 
    (like the massive healthy brain background) and forces the gradients to 
    focus on the hard-to-classify, low-contrast tumor boundaries.

We use:   Loss = 0.5 * DiceLoss + 0.5 * FocalLoss
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class DiceLoss(nn.Module):
    """
    Soft multi-class Dice Loss.
    Operates on one-hot encoded targets and softmax predictions.
    smooth: Laplace smoothing to avoid division by zero on empty slices.
    """
    def __init__(self, num_classes: int, smooth: float = 1e-5,
                 ignore_background: bool = True):
        super().__init__()
        self.num_classes = num_classes
        self.smooth = smooth
        self.ignore_bg = ignore_background   # skip class-0 (background) in mean

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        probs = F.softmax(logits, dim=1)                     # [B, C, H, W]

        # one-hot encode targets
        B, C, H, W = probs.shape
        one_hot = torch.zeros_like(probs)                    # [B, C, H, W]
        one_hot.scatter_(1, targets.unsqueeze(1), 1.0)

        start_cls = 1 if self.ignore_bg else 0
        dice_per_class = []

        for c in range(start_cls, C):
            p = probs[:, c].reshape(B, -1)                   # [B, H*W]
            t = one_hot[:, c].reshape(B, -1)

            intersection = (p * t).sum(dim=1)
            union        = p.sum(dim=1) + t.sum(dim=1)

            dice = (2.0 * intersection + self.smooth) / (union + self.smooth)
            dice_per_class.append(dice.mean())

        return 1.0 - torch.stack(dice_per_class).mean()


class FocalLoss(nn.Module):
    """
    Focal Loss for dense prediction.
    gamma: Focusing parameter. Higher gamma = more focus on hard examples.
    """
    def __init__(self, gamma: float = 2.0):
        super().__init__()
        self.gamma = gamma
        self.ce = nn.CrossEntropyLoss(reduction='none')

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        # Get standard Cross Entropy loss per pixel
        ce_loss = self.ce(logits, targets)
        
        # Calculate pt (the model's predicted probability for the TRUE class)
        pt = torch.exp(-ce_loss)
        
        # Apply the focal scaling factor: (1 - pt)^gamma
        focal_loss = ((1 - pt) ** self.gamma * ce_loss).mean()
        return focal_loss


class CombinedLoss(nn.Module):
    """
    0.5 * DiceLoss + 0.5 * FocalLoss
    The ultimate combination for rigorous medical image boundary detection.
    """
    def __init__(self, num_classes: int, dice_weight: float = 0.5,
                 focal_weight: float = 0.5):
        super().__init__()
        self.dice = DiceLoss(num_classes)
        self.focal = FocalLoss(gamma=2.0)
        self.dw   = dice_weight
        self.fw   = focal_weight

    def forward(self, logits: torch.Tensor, targets: torch.Tensor):
        d = self.dice(logits, targets)
        f = self.focal(logits, targets)
        return self.dw * d + self.fw * f, d.item(), f.item()


# ── quick test ───────────────────────────────────────────────
if __name__ == "__main__":
    from config import NUM_CLASSES, IMG_SIZE
    B = 2
    # Simulating the 12-channel output
    logits  = torch.randn(B, NUM_CLASSES, IMG_SIZE, IMG_SIZE)
    targets = torch.randint(0, NUM_CLASSES, (B, IMG_SIZE, IMG_SIZE))
    loss_fn = CombinedLoss(NUM_CLASSES)
    total, dl, fl = loss_fn(logits, targets)
    print(f"Combined loss: {total:.4f}  (Dice: {dl:.4f}, Focal: {fl:.4f})")