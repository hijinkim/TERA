#!/usr/bin/env bash
# Full RUOD-TX pipeline (paper settings). Edit the paths below.
set -euo pipefail
cd "$(dirname "$0")/.."

IMAGES=data/RUOD/images/train       # official RUOD training images (train + val split)
OUT=runs/ruod
DEVICE=cuda:0
mkdir -p "$OUT"

# 1) Clean-YOLO (frozen detector) and the DegFT-YOLO baseline
python tools/train_detector.py --mode clean --dataset ruod --image-dir "$IMAGES" --device 0 --name clean_ruod
python tools/train_detector.py --mode degft --dataset ruod --image-dir "$IMAGES" --device 0 --name degft_ruod \
    --weights runs/detector/clean_ruod/weights/best.pt
DETECTOR=runs/detector/clean_ruod/weights/best.pt

# 2) Pixel-restorer pretraining (20 epochs)
python tools/train_pixel_restorer.py --dataset ruod --image-dir "$IMAGES" --detector "$DETECTOR" \
    --save-dir "$OUT/pixel_restorer" --device "$DEVICE"

# 3) Shared neck statistics (mu, sigma) from the pretrained restorer
python tools/precompute_neck_stats.py --dataset ruod --image-dir "$IMAGES" --detector "$DETECTOR" \
    --pixel-restorer "$OUT/pixel_restorer/best.pth" --output "$OUT/neck_stats.pt" --device "$DEVICE"

# 4) Joint TERA training
python tools/train_tera.py --dataset ruod --image-dir "$IMAGES" --detector "$DETECTOR" \
    --pixel-restorer "$OUT/pixel_restorer/best.pth" --neck-stats "$OUT/neck_stats.pt" \
    --batch-size 16 --max-steps 5000 --save-dir "$OUT/tera" --device "$DEVICE"
