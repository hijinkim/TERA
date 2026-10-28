#!/usr/bin/env python3
"""Train the detectors used in the paper (YOLOv8s, 640x640).

  clean : Clean-YOLO, trained on the original pre-transmission images (100 epochs).
          This is the frozen detector used by TERA and all receiver-side restorers.
  degft : DegFT-YOLO, initialized from Clean-YOLO and fine-tuned for 50 epochs on the
          same online AquaTX degradation mixture as TERA (seen codecs only).

  python tools/train_detector.py --mode clean --dataset duo --image-dir data/DUO/images/train
  python tools/train_detector.py --mode degft --dataset duo --image-dir data/DUO/images/train \
      --weights runs/detector/clean_duo/weights/best.pt
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
ROOT = Path(__file__).resolve().parents[1]


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--mode", choices=("clean", "degft"), required=True)
    p.add_argument("--dataset", choices=("duo", "ruod"), required=True)
    p.add_argument("--image-dir", required=True, help="original training images (train + val split)")
    p.add_argument("--weights", default="yolov8s.pt", help="init weights (Clean-YOLO best.pt for --mode degft)")
    p.add_argument("--epochs", type=int, default=None, help="default: 100 (clean) / 50 (degft)")
    p.add_argument("--batch", type=int, default=None,
                   help="default (paper): clean 64 (DUO) / 48 (RUOD); degft 32")
    p.add_argument("--device", default="0")
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--project", default=str(ROOT / "runs" / "detector"))
    p.add_argument("--name", default=None)
    p.add_argument("--mix", default="0.1,0.1,0.1,0.7", help="degft training mixture clean,resize,codec,compound")
    return p.parse_args()


def main():
    args = parse_args()
    from ultralytics import YOLO

    from tera.degft import MIX_ENV, AquaTXDetectionTrainer, prepare_yolo_dataset

    splits = ROOT / "splits" / args.dataset
    data = prepare_yolo_dataset(args.image_dir, {"train": splits / "instances_train.json",
                                                 "val": splits / "instances_val.json"},
                                ROOT / "runs" / "yolo_data" / args.dataset)
    common = dict(data=str(data), imgsz=640, device=args.device, workers=args.workers, seed=args.seed,
                  deterministic=True, project=args.project, name=args.name or f"{args.mode}_{args.dataset}",
                  exist_ok=False, cache=False, lr0=1e-3, flipud=0.0, fliplr=0.5, degrees=0.0, shear=0.0,
                  perspective=0.0, mixup=0.0)
    if args.mode == "clean":
        YOLO(args.weights).train(
            epochs=args.epochs or 100, batch=args.batch or (64 if args.dataset == "duo" else 48), patience=30, warmup_epochs=3,
            hsv_h=0.015, hsv_s=0.5, hsv_v=0.3, translate=0.1, scale=0.3, mosaic=0.5, plots=True, **common)
    else:
        os.environ[MIX_ENV] = args.mix
        # make tera importable inside Ultralytics' DDP subprocesses
        os.environ["PYTHONPATH"] = os.pathsep.join([str(ROOT), *filter(None, os.environ.get("PYTHONPATH", "").split(os.pathsep))])
        YOLO(args.weights).train(
            trainer=AquaTXDetectionTrainer, epochs=args.epochs or 50, batch=args.batch or 32, patience=15,
            warmup_epochs=2, rect=False, hsv_h=0.0, hsv_s=0.0, hsv_v=0.0, translate=0.0, scale=0.0,
            mosaic=0.0, cutmix=0.0, close_mosaic=0, plots=False, **common)


if __name__ == "__main__":
    main()
