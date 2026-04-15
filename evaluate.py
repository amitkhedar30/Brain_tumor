# ============================================================
#  evaluate.py  –  3D Volume Reconstruction & Evaluation
# ============================================================

import os, argparse, json, glob, random
import numpy as np
import nibabel as nib
import torch
from tqdm import tqdm

# Added BASE_DIR to imports for relative path support
from config import (BASE_DIR, NUM_CLASSES, CHECKPOINT_DIR, RESULTS_DIR, MODELS, 
                    RAW_DATA_DIR, MODALITIES)
from models import get_model
from metrics import get_region_masks, dice_score, iou_score, hausdorff_95

def normalise(volume: np.ndarray) -> np.ndarray:
    mask = volume > 0
    if mask.sum() == 0: return volume.astype(np.float32)
    out = np.zeros_like(volume, dtype=np.float32)
    out[mask] = (volume[mask] - volume[mask].mean()) / (volume[mask].std() + 1e-8)
    return out

def remap_labels(seg: np.ndarray) -> np.ndarray:
    out = seg.copy()
    out[seg == 4] = 3
    return out.astype(np.uint8)

# ── [FIX] Locked Test Split Recovery ─────────────────────────
def get_test_patients():
    """Reads the exact patient IDs saved during preprocessing to prevent leakage."""
    list_path = os.path.join(BASE_DIR, "test_patients.txt")
    
    if not os.path.exists(list_path):
        print(f"[ERROR] {list_path} not found. Run preprocessing.py first!")
        return []
        
    with open(list_path, "r") as f:
        # Load the patient IDs (basenames)
        pids = [line.strip() for line in f if line.strip()]
        
    # Reconstruct full paths using the relative RAW_DATA_DIR
    return [os.path.join(RAW_DATA_DIR, pid) for pid in pids]
# ──────────────────────────────────────────────────────────────

@torch.no_grad()
def evaluate_model_3d(model_name: str, test_dirs: list, device: torch.device) -> dict:
    ckpt_path = os.path.join(CHECKPOINT_DIR, f"{model_name}_best.pth")
    if not os.path.exists(ckpt_path):
        print(f"  [{model_name}] No checkpoint found at {ckpt_path}. Skipping.")
        return {}

    model = get_model(model_name).to(device)
    ckpt  = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    tracker = {"WT": {"dice": [], "iou": [], "hd95": []},
               "TC": {"dice": [], "iou": [], "hd95": []},
               "ET": {"dice": [], "iou": [], "hd95": []}}

    print(f"  Running 3D Inference on {len(test_dirs)} patients...")
    
    for pdir in tqdm(test_dirs, desc=model_name):
        volumes = {}
        for mod in MODALITIES:
            # Constructing path based on the confirmed test IDs
            path = glob.glob(os.path.join(pdir, f"*_{mod}.nii*"))[0]
            volumes[mod] = normalise(nib.load(path).get_fdata())
            
        seg_path = glob.glob(os.path.join(pdir, "*_seg.nii*"))[0]
        gt_vol = remap_labels(nib.load(seg_path).get_fdata())
        
        H, W, D = gt_vol.shape
        pred_vol = np.zeros((H, W, D), dtype=np.uint8)

        for z in range(D):
            channels = []
            for mod in MODALITIES:
                vol = volumes[mod]
                for offset in [-1, 0, 1]:
                    idx = z + offset
                    if idx < 0 or idx >= D:
                        channels.append(np.zeros((H, W), dtype=np.float32))
                    else:
                        channels.append(vol[:, :, idx])
            
            inp_tensor = torch.from_numpy(np.stack(channels, axis=0)).unsqueeze(0).to(device)
            
            with torch.cuda.amp.autocast():
                logits = model(inp_tensor)
            
            pred_mask = logits.argmax(dim=1).squeeze(0).cpu().numpy()
            pred_vol[:, :, z] = pred_mask

        regions = get_region_masks(pred_vol, gt_vol)
        for region, (p_bin, g_bin) in regions.items():
            tracker[region]["dice"].append(dice_score(p_bin, g_bin))
            tracker[region]["iou"].append(iou_score(p_bin, g_bin))
            tracker[region]["hd95"].append(hausdorff_95(p_bin, g_bin))

    final_metrics = {"best_epoch": ckpt.get("epoch", "?")}
    for region in ("WT", "TC", "ET"):
        final_metrics[region] = {
            "dice": np.mean(tracker[region]["dice"]),
            "iou":  np.mean(tracker[region]["iou"]),
            "hd95": np.mean(tracker[region]["hd95"]),
        }
        
    return final_metrics

def main(models_to_eval):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    test_dirs = get_test_patients()
    results = {}

    print(f"\n{'='*55}")
    print(f"  Evaluating 3D Volumes on TEST set  [{device}]")
    print(f"{'='*55}")

    if not test_dirs:
        print("[ERROR] No test patients found. Did you run preprocessing.py?")
        return

    for name in models_to_eval:
        print(f"\n  → {name.upper()}")
        m = evaluate_model_3d(name, test_dirs, device)
        if m:
            results[name] = m
            for region in ("WT", "TC", "ET"):
                r = m[region]
                print(f"     {region}: Dice={r['dice']:.4f} | "
                      f"IoU={r['iou']:.4f} | HD95={r['hd95']:.2f}px")

    out_path = os.path.join(RESULTS_DIR, "all_models_3D_test_results.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n  All 3D results saved → {out_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="all",
                        help="Model name or 'all'")
    args = parser.parse_args()
    models = MODELS if args.model == "all" else [args.model]
    main(models)