#!/usr/bin/env bash
# Full DUO-TX pipeline (paper settings). Edit the paths below.
set -euo pipefail
cd "$(dirname "$0")/.."

IMAGES=data/DUO/images/train        # official DUO training images (train + val split)
OUT=runs/duo
DEVICE=cuda:0
mkdir -p "$OUT"

# 1) Clean-YOLO (frozen detector) and the DegFT-YOLO baseline
python tools/train_detector.py --mode clean --dataset duo --image-dir "$IMAGES" --device 0 --name clean_duo
python tools/train_detector.py --mode degft --dataset duo --image-dir "$IMAGES" --device 0 --name degft_duo \
    --weights runs/detector/clean_duo/weights/best.pt
DETECTOR=runs/detector/clean_duo/weights/best.pt

# 2) Pixel-restorer pretraining (20 epochs)
python tools/train_pixel_restorer.py --dataset duo --image-dir "$IMAGES" --detector "$DETECTOR" \
    --save-dir "$OUT/pixel_restorer" --device "$DEVICE"

# 3) Shared neck statistics (mu, sigma) from the pretrained restorer
python tools/precompute_neck_stats.py --dataset duo --image-dir "$IMAGES" --detector "$DETECTOR" \
    --pixel-restorer "$OUT/pixel_restorer/best.pth" --output "$OUT/neck_stats.pt" --device "$DEVICE"

# 4) Joint TERA training
python tools/train_tera.py --dataset duo --image-dir "$IMAGES" --detector "$DETECTOR" \
    --pixel-restorer "$OUT/pixel_restorer/best.pth" --neck-stats "$OUT/neck_stats.pt" \
    --batch-size 20 --max-steps 6000 --save-dir "$OUT/tera" --device "$DEVICE"
