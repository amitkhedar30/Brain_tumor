# ============================================================
#  config.py  –  Central configuration for BraTS 2.5D Project
# ============================================================

import os

# ── Paths (Relative & Portable) ──────────────────────────────
BASE_DIR        = os.path.dirname(os.path.abspath(__file__))
DATA_DIR        = os.path.join(BASE_DIR, "data")

# This is now relative to DAC-202/ so it works on both Windows and Linux
RAW_DATA_DIR    = os.path.join(BASE_DIR, "archive (4)", "BraTS2020_TrainingData", "MICCAI_BraTS2020_TrainingData")

PROCESSED_DIR   = os.path.join(DATA_DIR, "processed")
RESULTS_DIR     = os.path.join(BASE_DIR, "results")
CHECKPOINT_DIR  = os.path.join(BASE_DIR, "checkpoints")

# Ensure all directories exist automatically
for d in [DATA_DIR, PROCESSED_DIR, RESULTS_DIR, CHECKPOINT_DIR]:
    os.makedirs(d, exist_ok=True)

# ── Dataset ──────────────────────────────────────────────────
MODALITIES      = ["t1", "t1ce", "t2", "flair"]   # 4 MRI channels
INPUT_MODALITY  = "all"      
IMG_SIZE        = 240        
SLICE_THICKNESS = 1          # 2.5D: 1 slice above + current + 1 below
IN_CHANNELS     = 12         # 3 slices * 4 modalities = 12 input channels

# BraTS label mapping (re-mapped to 0,1,2,3 internally)
LABEL_MAP = {0: "Background", 1: "NCR/NET", 2: "Edema", 4: "Enhancing Tumor"}
NUM_CLASSES = 4              

# ── Training (Tuned for Stability) ───────────────────────────
TRAIN_SPLIT     = 0.70
VAL_SPLIT       = 0.15
TEST_SPLIT      = 0.15
BATCH_SIZE      = 8          
NUM_EPOCHS      = 50

# Lowered LR to prevent "greedy" early convergence
LEARNING_RATE   = 5e-5       
WEIGHT_DECAY    = 1e-5

# Increased patience to give Transformers time to learn global context
EARLY_STOP_PATIENCE = 20     
SEED            = 42

# ── Models ───────────────────────────────────────────────────
MODELS = ["unet", "resunet", "transunet"]

# ── Metrics ──────────────────────────────────────────────────
METRIC_CLASSES  = ["WT", "TC", "ET"]