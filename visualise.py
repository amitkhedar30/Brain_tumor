# ============================================================
#  visualise.py  –  Qualitative results (original | GT | pred)
# ============================================================
"""
Usage:
    python visualise.py --model transunet --n 12

Saves a grid of:
  [Original FLAIR slice] | [Ground Truth mask] | [Predicted mask]
for the best-performing model to results/visual_<model>.png.
Colours: background=black, NCR=red, Edema=yellow, ET=cyan.
"""

import os, argparse, random
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import ListedColormap

from config      import NUM_CLASSES, CHECKPOINT_DIR, RESULTS_DIR, SEED
from dataset     import BraTS25DDataset
from models      import get_model

random.seed(SEED)
torch.manual_seed(SEED)

# ── colour map: 0=BG(black) 1=NCR(red) 2=Edema(yellow) 3=ET(cyan)
CMAP   = ListedColormap(["black", "#E53935", "#FFD600", "#00E5FF"])
LABELS = {0: "Background", 1: "NCR/NET (red)",
          2: "Edema (yellow)", 3: "Enhancing Tumour (cyan)"}


def mask_to_rgb(mask: np.ndarray) -> np.ndarray:
    """Convert integer mask [H,W] → RGB [H,W,3] using CMAP."""
    colours = np.array([
        [0,   0,   0],     # 0 – background
        [229, 57,  53],    # 1 – NCR  red
        [255, 214, 0],     # 2 – edema yellow
        [0,   229, 255],   # 3 – ET cyan
    ], dtype=np.uint8)
    h, w = mask.shape
    rgb  = np.zeros((h, w, 3), dtype=np.uint8)
    for cls_id, colour in enumerate(colours):
        rgb[mask == cls_id] = colour
    return rgb


@torch.no_grad()
def make_visual(model_name: str, n_samples: int = 12):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ── load model ────────────────────────────────────────────
    ckpt_path = os.path.join(CHECKPOINT_DIR, f"{model_name}_best.pth")
    if not os.path.exists(ckpt_path):
        print(f"[ERROR] Checkpoint not found: {ckpt_path}")
        return

    model = get_model(model_name).to(device)
    ckpt  = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    # ── load test slices ──────────────────────────────────────
    try:
        ds = BraTS25DDataset("test")
    except FileNotFoundError:
        ds = BraTS25DDataset("val")

    # pick n_samples random slices that contain tumour
    tumour_indices = [i for i in range(len(ds)) if ds.data[i][1].max() > 0]
    if len(tumour_indices) == 0:
        tumour_indices = list(range(len(ds)))
    chosen = random.sample(tumour_indices, min(n_samples, len(tumour_indices)))

    # ── build figure  (n_samples rows × 3 columns) ───────────
    fig, axes = plt.subplots(len(chosen), 3,
                             figsize=(9, 3 * len(chosen)))
    fig.suptitle(f"{model_name.upper()} – Visual Results\n"
                 f"[Original FLAIR | Ground Truth | Prediction]",
                 fontsize=13, y=1.002)

    col_titles = ["FLAIR (middle slice)", "Ground Truth", "Prediction"]
    for col, title in enumerate(col_titles):
        axes[0, col].set_title(title, fontsize=10, fontweight="bold")

    for row, idx in enumerate(chosen):
        inp_np, mask_np = ds.data[idx]       # [3,H,W], [H,W]
        inp_t  = torch.from_numpy(inp_np).unsqueeze(0).to(device)
        logits = model(inp_t)
        pred   = logits.argmax(dim=1).squeeze().cpu().numpy()

        flair_slice = inp_np[1]              # middle channel = current slice

        axes[row, 0].imshow(flair_slice, cmap="gray")
        axes[row, 1].imshow(mask_to_rgb(mask_np))
        axes[row, 2].imshow(mask_to_rgb(pred))

        for col in range(3):
            axes[row, col].axis("off")

    # ── legend ────────────────────────────────────────────────
    patches = [mpatches.Patch(color=np.array(c) / 255, label=lbl)
               for c, lbl in [
                   ([229, 57, 53],  "NCR/NET"),
                   ([255, 214, 0],  "Edema"),
                   ([0, 229, 255],  "Enhancing Tumour"),
               ]]
    fig.legend(handles=patches, loc="lower center",
               ncol=3, fontsize=9, bbox_to_anchor=(0.5, -0.01))

    plt.tight_layout()
    out = os.path.join(RESULTS_DIR, f"visual_{model_name}.png")
    plt.savefig(out, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"  Visual output saved → {out}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="transunet",
                        choices=["unet", "resunet", "transunet"])
    parser.add_argument("--n", type=int, default=12,
                        help="Number of sample slices to visualise")
    args = parser.parse_args()
    make_visual(args.model, args.n)
