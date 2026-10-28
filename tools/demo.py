#!/usr/bin/env python3
"""Run TERA on images and draw the detections.

The input is treated as an already-received image. Pass --simulate to first apply
an AquaTX transmission condition to a clean image, e.g. --simulate 256 jpeg 0.3.

  python tools/demo.py --checkpoint runs/duo/tera/best.pth --detector runs/detector/clean_duo/weights/best.pt \
      --input examples/ --output outputs/demo --simulate 256 jpeg 0.3 --compare
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tera.metrics import decode_detector_output, postprocess  # noqa: E402
from tera.model import load_tera  # noqa: E402
from tera.transmission import AquaTXProtocol, canonical_geometry, centered_canvas, make_condition, render_pair  # noqa: E402

COLORS = [(56, 56, 255), (151, 157, 255), (31, 112, 255), (29, 178, 255)]


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--detector", required=True)
    p.add_argument("--input", required=True, help="image file or directory")
    p.add_argument("--output", default="outputs/demo")
    p.add_argument("--simulate", nargs=3, metavar=("LONG_SIDE", "CODEC", "BPP"), default=None)
    p.add_argument("--compare", action="store_true", help="also save the frozen detector without TERA")
    p.add_argument("--conf", type=float, default=0.25)
    p.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


def draw(image_rgb, detections, names, scale, left, top):
    canvas = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
    for box, score, label in zip(detections["boxes"], detections["scores"], detections["labels"]):
        x1, y1, x2, y2 = ((box.numpy() - np.array([left, top, left, top])) / scale).round().astype(int)
        color = COLORS[int(label) % len(COLORS)]
        cv2.rectangle(canvas, (x1, y1), (x2, y2), color, 2)
        cv2.putText(canvas, f"{names[int(label)]} {score:.2f}", (x1, max(0, y1 - 4)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)
    return canvas


def main():
    args = parse_args()
    device = torch.device(args.device)
    model = load_tera(args.checkpoint, args.detector, device)
    protocol = AquaTXProtocol()
    names = model.detector.names
    source = Path(args.input)
    paths = sorted(p for p in source.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}) \
        if source.is_dir() else [source]
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    for path in paths:
        rgb = cv2.cvtColor(cv2.imread(str(path), cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
        h, w = rgb.shape[:2]
        content_w, content_h, left, top = canonical_geometry(w, h, protocol.canvas)
        if args.simulate:
            side, codec, bpp = int(args.simulate[0]), args.simulate[1], float(args.simulate[2])
            role = "codec_only" if side == protocol.canvas else "compound"
            condition = make_condition(role, side, codec, bpp, "unseen" if codec == "avif" else "seen")
            canvas, _ = render_pair(rgb, condition, protocol)
        else:
            content = cv2.resize(rgb, (content_w, content_h), interpolation=cv2.INTER_LINEAR)
            canvas = centered_canvas(content, protocol.canvas, left, top, protocol.fill)
        received = canvas[top:top + content_h, left:left + content_w]
        scale = content_w / w
        x = torch.from_numpy(canvas.transpose(2, 0, 1).copy()).float().div(255).unsqueeze(0).to(device)
        with torch.inference_mode(), torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
            outputs = model(x, return_all=args.compare)
        variants = {"tera": outputs["tera"]}
        if args.compare:
            variants["frozen_detector"] = outputs["raw"]
        counts = {}
        for key, raw in variants.items():
            boxes, scores = decode_detector_output(raw, model.detector)
            det = postprocess(boxes, scores, protocol.canvas, conf=args.conf)[0]
            original = cv2.resize(received, (w, h), interpolation=cv2.INTER_LINEAR)
            cv2.imwrite(str(out_dir / f"{path.stem}_{key}.jpg"), draw(original, det, names, scale, left, top))
            counts[key] = len(det["boxes"])
        print(f"{path.name}: " + ", ".join(f"{k}={v} boxes" for k, v in counts.items()))


if __name__ == "__main__":
    main()
