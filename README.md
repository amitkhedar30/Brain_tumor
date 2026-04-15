# BraTS 2.5D Brain Tumour Segmentation
## 3-Model Comparative Study: UNet | ResUNet | TransUNet

---

## Project Structure

```
brats_project/
│
├── config.py           ← All hyperparameters & paths (edit this first)
├── preprocessing.py    ← BraTS NIfTI → 2.5D slice extraction
├── dataset.py          ← PyTorch Dataset + DataLoaders
├── models.py           ← UNet, ResUNet, TransUNet architectures
├── losses.py           ← Dice Loss + Combined Loss
├── metrics.py          ← Dice, IoU, HD95 for WT/TC/ET regions
├── train.py            ← Training loop with early stopping
├── evaluate.py         ← Load checkpoints → test metrics
├── compare.py          ← Tables + bar charts + radar chart
├── visualise.py        ← Original | GT | Prediction visual output
├── run_all.py          ← Master script (runs everything in order)
├── requirements.txt
│
├── data/
│   ├── raw/            ← PUT BraTS NIfTI folders here
│   └── processed/      ← Auto-generated 2.5D slices (pkl files)
│
├── checkpoints/        ← Best model weights saved here
└── results/            ← Metrics, plots, visual outputs saved here
```

---

## Setup

```bash
pip install -r requirements.txt
```

---

## Step 1 – Get the BraTS Dataset

1. Register at https://www.kaggle.com/datasets/awsaf49/brats20-dataset-training-validation
2. Download and extract into `data/raw/`
3. Structure should look like:
   ```
   data/raw/
     BraTS20_Training_001/
       BraTS20_Training_001_flair.nii.gz
       BraTS20_Training_001_seg.nii.gz
       ...
   ```

---

## Step 2 – Run the Full Pipeline

```bash
# Option A: Run everything at once
python run_all.py

# Option B: Step by step
python preprocessing.py                  # extract 2.5D slices
python train.py --model unet             # train baseline
python train.py --model resunet          # train strong contender
python train.py --model transunet        # train state-of-the-art
python evaluate.py                       # compute test metrics
python compare.py                        # generate comparison plots
python visualise.py --model transunet    # generate visual outputs
```

---

## What Each File Produces

| Script | Output |
|---|---|
| `preprocessing.py` | `data/processed/train.pkl`, `val.pkl`, `test.pkl` |
| `train.py` | `checkpoints/<model>_best.pth`, `results/<model>_history.json` |
| `evaluate.py` | `results/all_models_test_results.json` |
| `compare.py` | `comparison_table.txt`, `dice_comparison.png`, `hd95_comparison.png`, `radar_chart.png`, `training_curves_*.png` |
| `visualise.py` | `results/visual_<model>.png` |

---

## 2.5D Explained

For each axial slice `z`, we stack 3 consecutive slices:
```
Input = [slice z-1, slice z, slice z+1]  → shape [3, 128, 128]
```
This gives the model **depth context** (like a 3D approach) while keeping the memory footprint of a 2D model.

---

## Models

| Model | Key Feature | Targets |
|---|---|---|
| **UNet** | Classic encoder-decoder | Baseline |
| **ResUNet** | Residual blocks everywhere | Low-contrast boundaries |
| **TransUNet** | CNN + Transformer bottleneck | Global context, complex tumour shapes |

---

## Expected Dice Scores (well-tuned)

| Region | UNet | ResUNet | TransUNet |
|---|---|---|---|
| Whole Tumour (WT) | 0.82–0.86 | 0.85–0.88 | 0.87–0.90 |
| Tumour Core (TC) | 0.73–0.78 | 0.76–0.82 | 0.79–0.85 |
| Enhancing Tumour (ET) | 0.65–0.72 | 0.70–0.76 | 0.73–0.80 |

---

## Key Hyperparameters (config.py)

| Parameter | Default | Notes |
|---|---|---|
| `IMG_SIZE` | 128 | Increase to 224 if GPU allows |
| `BATCH_SIZE` | 8 | Reduce to 4 for limited VRAM |
| `NUM_EPOCHS` | 50 | 100 for best results |
| `LEARNING_RATE` | 1e-4 | Adam with cosine annealing |
| `SLICE_THICKNESS` | 1 | ±1 slice → 3 channels |
