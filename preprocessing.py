# ============================================================
#  preprocessing.py  –  BraTS → 2.5D slice extraction
# ============================================================

import os, glob, random
import numpy as np
import nibabel as nib
from tqdm import tqdm
import pickle

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# Added BASE_DIR to imports for relative path support
from config import (BASE_DIR, RAW_DATA_DIR, PROCESSED_DIR, IMG_SIZE,
                    SLICE_THICKNESS, SEED, TRAIN_SPLIT, VAL_SPLIT, MODALITIES)

random.seed(SEED)
np.random.seed(SEED)

# ── helpers ──────────────────────────────────────────────────

def normalise(volume: np.ndarray) -> np.ndarray:
    """Z-score normalise a single MRI volume (non-zero voxels only)."""
    mask = volume > 0
    if mask.sum() == 0:
        return volume.astype(np.float32)
    mean = volume[mask].mean()
    std  = volume[mask].std() + 1e-8
    out  = np.zeros_like(volume, dtype=np.float32)
    out[mask] = (volume[mask] - mean) / std
    return out

def remap_labels(seg: np.ndarray) -> np.ndarray:
    """Remap 4 → 3 so labels are contiguous [0,1,2,3]."""
    out = seg.copy()
    out[seg == 4] = 3
    return out.astype(np.uint8)

def extract_2_5d_slices(volumes_dict, seg_vol, split_type, thickness=SLICE_THICKNESS):
    """
    volumes_dict: dict containing the 4 normalized 3D modalities.
    Returns list of (input_array [12, 240, 240], mask_array [240, 240]).
    """
    D = seg_vol.shape[2]
    slices = []
    
    for z in range(thickness, D - thickness):
        mask_slice = seg_vol[:, :, z]
        has_tumour = mask_slice.max() > 0
        
        # Filtering logic remains: 80% skip for empty train slices
        if split_type == "train" and not has_tumour and random.random() > 0.20:
            continue

        channels = []
        for mod in MODALITIES:
            vol = volumes_dict[mod]
            for offset in range(-thickness, thickness + 1):
                channels.append(vol[:, :, z + offset])

        inp  = np.stack(channels, axis=0).astype(np.float32)   # [12, 240, 240]
        mask = mask_slice.astype(np.uint8)                     # [240, 240]
        slices.append((inp, mask))
        
    return slices

# ── main pipeline ─────────────────────────────────────────────

def process_all_patients():
    # Uses the relative RAW_DATA_DIR from your portable config
    patient_dirs = sorted(glob.glob(os.path.join(RAW_DATA_DIR, "BraTS20_Training_*")))
    if len(patient_dirs) == 0:
        print(f"[WARNING] No patients found in {RAW_DATA_DIR}")
        return

    # 1. Perform Patient-Level Split
    random.shuffle(patient_dirs)
    n = len(patient_dirs)
    n_train = int(n * TRAIN_SPLIT)
    n_val   = int(n * VAL_SPLIT)
    
    train_dirs = patient_dirs[:n_train]
    val_dirs   = patient_dirs[n_train:n_train + n_val]
    test_dirs  = patient_dirs[n_train + n_val:]
    
    # ── [FIX] Lock in the Test Split ────────────────────────
    # Saves patient IDs to test_patients.txt for evaluate.py to use
    test_list_path = os.path.join(BASE_DIR, "test_patients.txt")
    with open(test_list_path, "w") as f:
        for pdir in test_dirs:
            f.write(os.path.basename(pdir) + "\n")
    print(f"  [FIX] Test patient list locked → {test_list_path}")
    # ────────────────────────────────────────────────────────

    print(f"Split: {len(train_dirs)} Train | {len(val_dirs)} Val | {len(test_dirs)} Test (Reserved)")

    # 2. Process Train and Val
    splits_to_process = {"train": train_dirs, "val": val_dirs}
    
    for split_name, directories in splits_to_process.items():
        print(f"\nExtracting {split_name.upper()} set...")
        split_data = []
        
        for pdir in tqdm(directories):
            pid = os.path.basename(pdir)
            
            volumes = {}
            missing_file = False
            for mod in MODALITIES:
                paths = glob.glob(os.path.join(pdir, f"*_{mod}.nii*"))
                if not paths:
                    missing_file = True
                    break
                volumes[mod] = normalise(nib.load(paths[0]).get_fdata())
                
            seg_paths = glob.glob(os.path.join(pdir, "*_seg.nii*"))
            if missing_file or not seg_paths:
                print(f"  [SKIP] {pid} – missing modality or seg files")
                continue
                
            seg_vol = remap_labels(nib.load(seg_paths[0]).get_fdata())

            patient_slices = extract_2_5d_slices(volumes, seg_vol, split_name)
            split_data.extend(patient_slices)

        # 3. Save to the portable PROCESSED_DIR
        path = os.path.join(PROCESSED_DIR, f"{split_name}.pkl")
        with open(path, "wb") as f:
            pickle.dump(split_data, f)
        print(f"  Saved {len(split_data)} slices → {path}")
        
        del split_data 

    print("\n[preprocessing.py] Done. (Test set remains raw for 3D inference).")

if __name__ == "__main__":
    process_all_patients()