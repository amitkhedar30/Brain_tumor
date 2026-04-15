# ============================================================
#  dataset.py  –  PyTorch Dataset for 2.5D BraTS slices
# ============================================================

import os, pickle
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import albumentations as A

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import PROCESSED_DIR, BATCH_SIZE, IMG_SIZE, IN_CHANNELS

# ── augmentation pipelines ───────────────────────────────────

def get_train_transforms():
    # Removed Brightness/Contrast to prevent 12-channel crashes and preserve MRI intensity scaling.
    return A.Compose([
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.3),
        A.Rotate(limit=15, p=0.4),
        A.ElasticTransform(alpha=1, sigma=10, p=0.3),
        # GaussNoise usually handles multi-channel, but if it throws an error, remove it.
        A.GaussNoise(var_limit=(0.001, 0.01), p=0.3),
    ])

def get_val_transforms():
    return None   # no augmentation for val/test

# ── dataset class (For Train & Val Only) ──────────────────────

class BraTS25DDataset(Dataset):
    """
    Loads pre-extracted 2.5D slices from a pickle file.
    Expects inp: [12, 240, 240], mask: [240, 240].
    """

    def __init__(self, split: str = "train"):
        assert split in ("train", "val") # Removed "test" from here
        pkl_path = os.path.join(PROCESSED_DIR, f"{split}.pkl")

        if not os.path.exists(pkl_path):
            raise FileNotFoundError(
                f"{pkl_path} not found. Run preprocessing.py first."
            )

        with open(pkl_path, "rb") as f:
            self.data = pickle.load(f)   # list of (inp [12,H,W], mask [H,W])

        self.transforms = get_train_transforms() if split == "train" else None
        self.split = split

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        inp, mask = self.data[idx]       # inp: [12, 240, 240] float32, mask: [240, 240] uint8

        if self.transforms:
            # albumentations expects HWC image
            inp_hwc = np.transpose(inp, (1, 2, 0))   # [240, 240, 12]
            augmented = self.transforms(image=inp_hwc, mask=mask)
            inp  = np.transpose(augmented["image"], (2, 0, 1)).astype(np.float32)
            mask = augmented["mask"]

        inp_tensor  = torch.from_numpy(inp).float()
        mask_tensor = torch.from_numpy(mask.astype(np.int64)).long()
        return inp_tensor, mask_tensor


# ── convenience loaders ──────────────────────────────────────

def get_loaders(batch_size: int = BATCH_SIZE, num_workers: int = 2):
    loaders = {}
    
    # We only load train and val here. 
    # Test data will be loaded patient-by-patient in Evaluate.py
    for split in ("train", "val"):
        try:
            ds = BraTS25DDataset(split)
            shuffle = (split == "train")
            loaders[split] = DataLoader(
                ds,
                batch_size=batch_size,
                shuffle=shuffle,
                num_workers=num_workers,
                pin_memory=True,
                drop_last=(split == "train"),
            )
            print(f"  [{split:5s}] {len(ds):>6} slices")
        except FileNotFoundError as e:
            print(f"  [{split:5s}] SKIPPED – {e}")
            
    return loaders


if __name__ == "__main__":
    print("Testing dataset …")
    loaders = get_loaders(batch_size=4)
    if "train" in loaders:
        inp, mask = next(iter(loaders["train"]))
        print(f"  Input shape : {inp.shape}")    # Should be [4, 12, 240, 240]
        print(f"  Mask  shape : {mask.shape}")   # Should be [4, 240, 240]
        print(f"  Unique labels: {mask.unique().tolist()}")