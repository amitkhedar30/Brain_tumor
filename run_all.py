# ============================================================
#  run_all.py  –  Run full pipeline: preprocess → train → compare
# ============================================================
"""
Usage:
    python run_all.py                  # full pipeline
    python run_all.py --skip_preprocess
    python run_all.py --models unet resunet
"""

import argparse, subprocess, sys

STEPS = {
    "preprocess": "python preprocessing.py",
    "unet":       "python train.py --model unet",
    "resunet":    "python train.py --model resunet",
    "transunet":  "python train.py --model transunet",
    "evaluate":   "python evaluate.py",
    "compare":    "python compare.py",
    "visualise":  "python visualise.py --model transunet --n 12",
}


def run(cmd: str):
    print(f"\n{'='*55}")
    print(f"  CMD: {cmd}")
    print(f"{'='*55}")
    result = subprocess.run(cmd, shell=True)
    if result.returncode != 0:
        print(f"\n[ERROR] Command failed: {cmd}")
        sys.exit(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip_preprocess", action="store_true")
    parser.add_argument("--models", nargs="+",
                        default=["unet", "resunet", "transunet"])
    args = parser.parse_args()

    if not args.skip_preprocess:
        run(STEPS["preprocess"])

    for m in args.models:
        run(STEPS[m])

    run(STEPS["evaluate"])
    run(STEPS["compare"])
    run(STEPS["visualise"])

    print("\n✓ Full pipeline complete! Check the results/ directory.")
