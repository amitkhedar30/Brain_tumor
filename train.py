# ============================================================
#  train.py  –  Final Optimized Training Engine
# ============================================================

import os, argparse, json, time
import torch
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR

# All parameters are now synced with your portable config.py
from config import (BASE_DIR, NUM_CLASSES, LEARNING_RATE, WEIGHT_DECAY,
                    NUM_EPOCHS, EARLY_STOP_PATIENCE, CHECKPOINT_DIR, RESULTS_DIR)
from dataset import get_loaders
from models  import get_model
from losses  import CombinedLoss
from metrics import MetricTracker

# Keeps mathematical batch size stable
ACCUMULATION_STEPS = 4 

def train_one_epoch(model, loader, optimizer, loss_fn, device, scaler):
    model.train()
    total_loss = total_dice = total_focal = 0.0
    optimizer.zero_grad() 
    
    for i, (inp, mask) in enumerate(loader):
        inp, mask = inp.to(device), mask.to(device)
        
        with torch.cuda.amp.autocast():
            logits = model(inp)
            loss, dl, fl = loss_fn(logits, mask)
            loss = loss / ACCUMULATION_STEPS
            
        scaler.scale(loss).backward()
        
        if (i + 1) % ACCUMULATION_STEPS == 0 or (i + 1) == len(loader):
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

        total_loss += (loss.item() * ACCUMULATION_STEPS)
        total_dice += dl
        total_focal += fl
        
    return total_loss / len(loader), total_dice / len(loader), total_focal / len(loader)


@torch.no_grad()
def evaluate(model, loader, loss_fn, device):
    model.eval()
    tracker    = MetricTracker()
    total_loss = 0.0
    
    for inp, mask in loader:
        inp, mask = inp.to(device), mask.to(device)
        with torch.cuda.amp.autocast():
            logits = model(inp)
            loss, _, _ = loss_fn(logits, mask)
            
        total_loss += loss.item()
        tracker.update(logits, mask)
        
    return total_loss / len(loader), tracker.compute()


def train(model_name: str):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n{'='*55}")
    print(f"  Training : {model_name.upper()}  on {device}")
    print(f"{'='*55}")

    # ── [FIX] Data Leakage Sanity Check ────────────────────
    test_list_path = os.path.join(BASE_DIR, "test_patients.txt")
    if os.path.exists(test_list_path):
        print(f"  [OK] Found locked test set. Evaluation will be clean.")
    else:
        print(f"  [WARNING] No test_patients.txt found. Run preprocessing.py first!")

    # ── data ───────────────────────────────────────────────
    loaders = get_loaders()
    if "train" not in loaders:
        print("[ERROR] No training data. Run preprocessing.py first.")
        return

    # ── model / optimiser / scheduler / scaler ─────────────
    model     = get_model(model_name).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE,
                            weight_decay=WEIGHT_DECAY)
    
    # Cosine Annealing helps find better minima after the initial descent
    scheduler = CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS, eta_min=1e-6)
    loss_fn   = CombinedLoss(NUM_CLASSES)
    scaler    = torch.cuda.amp.GradScaler()

    # ── training loop ───────────────────────────────────────
    history        = []
    best_val_dice  = -1.0
    patience_count = 0
    ckpt_path      = os.path.join(CHECKPOINT_DIR, f"{model_name}_best.pth")

    for epoch in range(1, NUM_EPOCHS + 1):
        t0 = time.time()
        tr_loss, tr_dl, tr_fl = train_one_epoch(model, loaders["train"], optimizer, loss_fn, device, scaler)
        val_loss, val_metrics = evaluate(model, loaders["val"], loss_fn, device)
        scheduler.step()

        wt_dice = val_metrics["WT"]["dice"]
        elapsed = time.time() - t0

        record = {
            "epoch": epoch,
            "train_loss": round(tr_loss, 5),
            "val_loss":   round(val_loss, 5),
            "val_WT_dice": round(wt_dice, 4),
            "val_TC_dice": round(val_metrics["TC"]["dice"], 4),
            "val_ET_dice": round(val_metrics["ET"]["dice"], 4),
            "lr": round(scheduler.get_last_lr()[0], 7),
        }
        history.append(record)

        print(f"  Ep {epoch:03d}/{NUM_EPOCHS} | Loss {tr_loss:.4f}/{val_loss:.4f} | "
              f"WT={wt_dice:.4f} TC={val_metrics['TC']['dice']:.4f} ET={val_metrics['ET']['dice']:.4f} | {elapsed:.1f}s")

        if wt_dice > best_val_dice:
            best_val_dice = wt_dice
            patience_count = 0
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_metrics": val_metrics,
            }, ckpt_path)
            print(f"         ✓ Saved best checkpoint (WT Dice={best_val_dice:.4f})")
        else:
            patience_count += 1
            if patience_count >= EARLY_STOP_PATIENCE:
                print(f"  Early stopping triggered at epoch {epoch}.")
                break

    hist_path = os.path.join(RESULTS_DIR, f"{model_name}_history.json")
    with open(hist_path, "w") as f:
        json.dump(history, f, indent=2)
    print(f"\n  History saved → {hist_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="unet",
                        choices=["unet", "resunet", "transunet"])
    args = parser.parse_args()
    train(args.model)