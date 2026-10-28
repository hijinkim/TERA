#!/usr/bin/env python3
"""Evaluate TERA (and baselines) on AquaTX, reproducing paper Table 3.

Methods (all on the same received images):
  raw         Clean-YOLO on the received image (no restoration)
  pixel_only  Clean-YOLO on the pixel-restored image x_r
  tera        Clean-YOLO head on the corrected pyramid Z_c
  degft       degradation-fine-tuned YOLO on the received image (with --degft)

Reported per condition and macro-averaged per group:
  clean (1) | resize_only (3) | codec_only (9) | seen_compound (27) | unseen_avif (9)

Example:
  python tools/evaluate.py --dataset duo \
      --checkpoint runs/duo/tera/best.pth --detector runs/detector/clean_duo/weights/best.pt \
      --degft runs/detector/degft_duo/weights/best.pt --aquatx-root data/AquaTX/DUO-TX/test
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tera.data import AquaTXFrozen, AquaTXOnline, make_loader  # noqa: E402
from tera.detector import FrozenYOLO  # noqa: E402
from tera.metrics import METRIC_KEYS, DetectionMAP, decode_detector_output  # noqa: E402
from tera.model import load_tera  # noqa: E402
from tera.transmission import AquaTXProtocol  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
GROUP_ORDER = ("clean", "resize_only", "codec_only", "seen_compound", "unseen_avif")


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset", choices=("duo", "ruod"), required=True)
    p.add_argument("--checkpoint", required=True, help="TERA checkpoint (tera-v1 format)")
    p.add_argument("--detector", required=True, help="frozen Clean-YOLO weights")
    p.add_argument("--degft", default=None, help="optional DegFT-YOLO weights")
    p.add_argument("--aquatx-root", default=None, help="frozen assets, e.g. data/AquaTX/DUO-TX/test")
    p.add_argument("--online", action="store_true", help="render conditions on the fly instead of frozen assets")
    p.add_argument("--image-dir", default=None, help="original test images (only with --online)")
    p.add_argument("--annotation", default=None, help="defaults to splits/<dataset>/instances_test.json")
    p.add_argument("--groups", default=",".join(GROUP_ORDER))
    p.add_argument("--batch-size", type=int, default=48)
    p.add_argument("--workers", type=int, default=12)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--no-amp", action="store_true", help="disable bfloat16 autocast (paper numbers use bf16)")
    p.add_argument("--output", default=None, help="write per-condition and macro results to this JSON")
    return p.parse_args()


@torch.inference_mode()
def evaluate_condition(model, degft, loader, device, amp):
    methods = ["raw", "pixel_only", "tera"] + (["degft"] if degft is not None else [])
    metrics = {m: DetectionMAP() for m in methods}
    for batch in loader:
        degraded = batch["degraded"].to(device, non_blocking=True)
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=amp):
            outputs = model(degraded, return_all=True)
            if degft is not None:
                outputs["degft"] = degft.forward_detect(degft.forward_detect_inputs(degraded))
        for m in methods:
            boxes, scores = decode_detector_output(outputs[m], degft if m == "degft" else model.detector)
            metrics[m].update(boxes, scores, batch)
    return {m: metric.compute() for m, metric in metrics.items()}


def macro(rows: list[dict], method: str) -> dict:
    return {k: sum(r[method][k] for r in rows) / len(rows) for k in METRIC_KEYS}


def main():
    args = parse_args()
    device = torch.device(args.device)
    amp = not args.no_amp and device.type == "cuda"
    annotation = args.annotation or ROOT / "splits" / args.dataset / "instances_test.json"
    if args.online and not args.image_dir:
        raise SystemExit("--online requires --image-dir")
    if not args.online and not args.aquatx_root:
        raise SystemExit("pass --aquatx-root (frozen assets) or --online --image-dir")

    model = load_tera(args.checkpoint, args.detector, device)
    degft = FrozenYOLO(args.degft).to(device).eval() if args.degft else None
    protocol = AquaTXProtocol()
    groups = protocol.eval_groups()
    selected = [g for g in args.groups.split(",") if g]

    dataset = None
    results = {"conditions": {}, "groups": {}}
    for group in selected:
        rows = []
        for condition in groups[group]:
            if args.online:
                if dataset is None:
                    dataset = AquaTXOnline(args.image_dir, annotation, protocol=protocol, condition=condition)
                dataset.set_condition(condition)
            else:
                if dataset is None:
                    dataset = AquaTXFrozen(args.aquatx_root, annotation, condition)
                dataset.set_condition(condition)
            model.detector.verify_class_names(dataset.names)
            loader = make_loader(dataset, args.batch_size, args.workers)
            row = evaluate_condition(model, degft, loader, device, amp)
            rows.append(row)
            results["conditions"][condition.id] = row
            print(f"[{group}] {condition.id:34s} " + "  ".join(f"{m}={row[m]['map_50_95'] * 100:5.1f}" for m in row),
                  flush=True)
        results["groups"][group] = {m: macro(rows, m) for m in rows[0]}

    methods = list(next(iter(results["groups"].values())))
    print("\nmAP50:95 (%)  " + "".join(f"{g}({len(groups[g])})".rjust(18) for g in selected))
    for m in methods:
        print(f"{m:12s}  " + "".join(f"{results['groups'][g][m]['map_50_95'] * 100:18.1f}" for g in selected))

    if args.output:
        out = Path(args.output)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
        print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
