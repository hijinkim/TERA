#!/usr/bin/env python3
"""Receiver-side compute of TERA vs. the frozen detector (paper Sec. 5.1, computational profiling).

Reports parameters, detector backbone/neck passes, GMACs/GFLOPs, latency and peak GPU memory
for a single 640x640 image (batch 1, bfloat16 autocast, median over --iters after warm-up).

  python tools/measure_compute.py --checkpoint runs/duo/tera/best.pth --detector runs/detector/clean_duo/weights/best.pt
"""

from __future__ import annotations

import argparse
import statistics
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tera.detector import FrozenYOLO  # noqa: E402
from tera.model import load_tera  # noqa: E402


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--detector", required=True)
    p.add_argument("--degft", default=None, help="optional DegFT-YOLO weights")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--iters", type=int, default=200)
    p.add_argument("--warmup", type=int, default=20)
    return p.parse_args()


class Graph(torch.nn.Module):
    """Holds the modules as children so the MAC hooks see every call."""

    def __init__(self, kind: str, tera, detector=None) -> None:
        super().__init__()
        self.kind = kind
        self.tera = tera
        self.detector = detector if detector is not None else tera.detector

    def forward(self, x):
        if self.kind == "detector":
            return self.detector.forward_detect(self.detector.forward_detect_inputs(x))
        if self.kind == "pixel_only":
            restored = self.tera.pixel_restorer(x, task_features=self.detector.forward_backbone(x))["restored"]
            return self.detector.forward_detect(self.detector.forward_detect_inputs(restored))
        return self.tera(x)["tera"]


@torch.inference_mode()
def gmacs(graph, x) -> float:
    """Conv2d/Linear multiply-accumulates, counted once per actual call."""
    total = [0.0]

    def conv(m, _, out):
        total[0] += (m.in_channels // m.groups) * out.shape[1] * m.kernel_size[0] * m.kernel_size[1] * out.shape[-2] * out.shape[-1]

    def linear(m, _, out):
        total[0] += m.in_features * m.out_features * (out.numel() // out.shape[-1])

    handles = [m.register_forward_hook(conv if isinstance(m, torch.nn.Conv2d) else linear)
               for m in set(graph.modules()) if isinstance(m, (torch.nn.Conv2d, torch.nn.Linear))]
    graph(x)
    for h in handles:
        h.remove()
    return total[0] / 1e9


@torch.inference_mode()
def latency(graph, x, device, iters, warmup):
    def run():
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
            graph(x)
    for _ in range(warmup):
        run()
    torch.cuda.synchronize(device)
    torch.cuda.reset_peak_memory_stats(device)
    times = []
    for _ in range(iters):
        torch.cuda.synchronize(device)
        start = time.perf_counter()
        run()
        torch.cuda.synchronize(device)
        times.append((time.perf_counter() - start) * 1000)
    return statistics.median(times), torch.cuda.max_memory_allocated(device) / 1024 ** 2


def main():
    args = parse_args()
    device = torch.device(args.device)
    tera = load_tera(args.checkpoint, args.detector, device)
    n = lambda m: sum(p.numel() for p in m.parameters()) / 1e6  # noqa: E731
    det_params, pix_params, evi_params = n(tera.detector), n(tera.pixel_restorer), n(tera.evidence_restorer)
    print(f"parameters: detector {det_params:.2f}M | pixel restorer {pix_params:.2f}M | "
          f"evidence restorer {evi_params:.2f}M")

    methods = [("Frozen detector", Graph("detector", tera), 0.0, 1)]
    if args.degft:
        methods.append(("DegFT-YOLO", Graph("detector", tera, FrozenYOLO(args.degft).to(device).eval()), 0.0, 1))
    methods += [("Pixel-only", Graph("pixel_only", tera), pix_params, 1),
                ("TERA", Graph("tera", tera), pix_params + evi_params, 2)]

    x = torch.rand(1, 3, 640, 640, device=device)
    print(f"\n{'Method':18s}{'+Params(M)':>11s}{'Passes':>7s}{'GMACs':>8s}{'GFLOPs':>8s}{'ms/img':>8s}{'img/s':>8s}{'PeakMB':>8s}")
    for name, graph, added, passes in methods:
        ms, peak = latency(graph, x, device, args.iters, args.warmup)
        g = gmacs(graph, x)
        print(f"{name:18s}{added:>11.2f}{passes:>7d}{g:>8.2f}{2 * g:>8.2f}{ms:>8.2f}{1000 / ms:>8.1f}{peak:>8.0f}")
    print(f"\nbatch 1, 640x640, bf16 autocast, median of {args.iters} iterations on {torch.cuda.get_device_name(device)}; "
          "GFLOPs = 2 x GMACs; Passes = detector backbone/neck passes.")


if __name__ == "__main__":
    main()
