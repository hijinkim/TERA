#!/usr/bin/env bash
# Evaluate TERA, Clean-YOLO and DegFT-YOLO (Table 3 protocol) with the checkpoints from scripts/train_*.sh.
set -euo pipefail
cd "$(dirname "$0")/.."

for DS in duo ruod; do
  NAME=$(echo "$DS" | tr '[:lower:]' '[:upper:]')
  python tools/evaluate.py --dataset "$DS" \
      --checkpoint "runs/${DS}/tera/best.pth" \
      --detector "runs/detector/clean_${DS}/weights/best.pt" \
      --degft "runs/detector/degft_${DS}/weights/best.pt" \
      --aquatx-root "data/AquaTX/${NAME}-TX/test" \
      --output "results/${DS}_test.json"
done
