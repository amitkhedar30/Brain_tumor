# ============================================================
#  metrics.py  –  Dice | IoU | Hausdorff 95 for BraTS
# ============================================================
"""
BraTS evaluation follows three COMPOUND regions:
  WT (Whole Tumour)     = labels 1 + 2 + 3
  TC (Tumour Core)      = labels 1 + 3
  ET (Enhancing Tumour) = label  3
"""

import numpy as np
import torch
from scipy.spatial import cKDTree

# ── label → compound region masks ────────────────────────────

def get_region_masks(pred: np.ndarray, gt: np.ndarray):
    """
    pred, gt : integer arrays with values in {0,1,2,3}
    Returns dict of (pred_binary, gt_binary) per region.
    """
    return {
        "WT": (pred >= 1,          gt >= 1),
        "TC": ((pred == 1) | (pred == 3), (gt == 1) | (gt == 3)),
        "ET": (pred == 3,          gt == 3),
    }

# ── per-region metrics ────────────────────────────────────────

def dice_score(pred_bin: np.ndarray, gt_bin: np.ndarray,
               smooth: float = 1e-5) -> float:
    inter = (pred_bin & gt_bin).sum()
    union = pred_bin.sum() + gt_bin.sum()
    return float((2 * inter + smooth) / (union + smooth))


def iou_score(pred_bin: np.ndarray, gt_bin: np.ndarray,
              smooth: float = 1e-5) -> float:
    inter = (pred_bin & gt_bin).sum()
    union = (pred_bin | gt_bin).sum()
    return float((inter + smooth) / (union + smooth))


def hausdorff_95(pred_bin: np.ndarray, gt_bin: np.ndarray) -> float:
    """
    True 95th-percentile Hausdorff Distance.
    Uses cKDTree for extremely fast nearest-neighbor calculations.
    """
    pred_pts = np.argwhere(pred_bin)
    gt_pts   = np.argwhere(gt_bin)

    if len(pred_pts) == 0 and len(gt_pts) == 0:
        return 0.0
    
    # If one is empty, return a penalty distance (approx diagonal of 240x240 image)
    if len(pred_pts) == 0 or len(gt_pts) == 0:
        return 339.0   

    # Build KD-Trees for fast nearest-neighbor search
    tree_pred = cKDTree(pred_pts)
    tree_gt   = cKDTree(gt_pts)

    # d1: distances from each prediction point to the nearest ground truth point
    d1, _ = tree_gt.query(pred_pts)
    # d2: distances from each ground truth point to the nearest prediction point
    d2, _ = tree_pred.query(gt_pts)

    # Calculate 95th percentile of these distances
    return float(max(np.percentile(d1, 95), np.percentile(d2, 95)))


# ── batch-level evaluation (Optimized for Training Loop) ──────

class MetricTracker:
    """
    Accumulates per-batch metrics and computes epoch averages.
    *NOTE*: HD95 is intentionally removed from this loop to prevent 
    massive training slowdowns. HD95 is calculated in evaluate.py instead.
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self._sums   = {r: {"dice": 0.0, "iou": 0.0} for r in ("WT", "TC", "ET")}
        self._counts = {r: 0 for r in ("WT", "TC", "ET")}

    def update(self, pred_logits: torch.Tensor, targets: torch.Tensor):
        preds = pred_logits.argmax(dim=1).cpu().numpy()   
        gts   = targets.cpu().numpy()

        for b in range(preds.shape[0]):
            regions = get_region_masks(preds[b], gts[b])
            for region, (p_bin, g_bin) in regions.items():
                self._sums[region]["dice"] += dice_score(p_bin, g_bin)
                self._sums[region]["iou"]  += iou_score(p_bin, g_bin)
                self._counts[region] += 1

    def compute(self) -> dict:
        out = {}
        for region in ("WT", "TC", "ET"):
            n = max(self._counts[region], 1)
            out[region] = {
                "dice": self._sums[region]["dice"] / n,
                "iou":  self._sums[region]["iou"]  / n,
            }
            # Adding a placeholder so train.py doesn't crash when logging
            out[region]["hd95"] = 0.0 
        return out

    def summary_string(self) -> str:
        m = self.compute()
        lines = []
        for region in ("WT", "TC", "ET"):
            r = m[region]
            lines.append(f"  {region}: Dice={r['dice']:.4f} | IoU={r['iou']:.4f}")
        return "\n".join(lines)