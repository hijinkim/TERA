#!/usr/bin/env python3
"""Stage 1: pretrain the task-guided pixel restorer in front of the frozen detector.

Schedule used for the released checkpoints (one per dataset; 20 epochs, batch 4, AdamW 2e-4, cosine):
  epochs 1-5   warmup: no detection loss, feature loss x0.2, spatial gate teacher-forced with
               the detector-feature damage map of the (received, pre-transmission) pair
  epochs 6-15  detection + detection no-harm losses ramped in with a cosine schedule
  epochs 16-20 full objective
The checkpoint with the best 27-condition validation mAP (pixel-only path, first 100
validation images, evaluated every 5 epochs) is kept and initializes joint TERA training
on the same dataset (DUO-TX and RUOD-TX are trained independently). No transmission metadata is
used: every target is derived from the image pair, the boxes and the frozen detector.

  python tools/train_pixel_restorer.py --dataset duo --image-dir data/DUO/images/train \
      --detector runs/detector/clean_duo/weights/best.pt --save-dir runs/pixel_restorer_duo
"""

from __future__ import annotations

import argparse
import csv
import random
import sys
from pathlib import Path

import numpy as np
import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tera.data import AquaTXOnline, make_loader, transmission_free_mask, yolo_targets  # noqa: E402
from tera.detector import FrozenYOLO  # noqa: E402
from tera.metrics import DetectionMAP, decode_detector_output  # noqa: E402
from tera.model import CHECKPOINT_FORMAT  # noqa: E402
from tera.pixel_restorer import P3_LAYER, P4_LAYER, PixelRestorer, PixelRestorerConfig  # noqa: E402
from tera.pretrain_losses import (  # noqa: E402
    PixelRestorerObjective,
    PretrainWeights,
    detection_scale,
    spatial_task_target,
)
from tera.transmission import AquaTXProtocol, SamplingMix  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset", choices=("duo", "ruod"), default="duo")
    p.add_argument("--image-dir", required=True)
    p.add_argument("--train-annotation", default=None)
    p.add_argument("--val-annotation", default=None)
    p.add_argument("--detector", required=True, help="frozen Clean-YOLO weights")
    p.add_argument("--save-dir", required=True)
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--warmup-epochs", type=int, default=5)
    p.add_argument("--ramp-start", type=int, default=6)
    p.add_argument("--ramp-end", type=int, default=15)
    p.add_argument("--teacher-epochs", type=int, default=5, help="epochs with a teacher-forced spatial gate")
    p.add_argument("--warmup-feature-scale", type=float, default=0.2)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--mix", type=float, nargs=4, default=(0.10, 0.10, 0.10, 0.70))
    p.add_argument("--eval-interval", type=int, default=5)
    p.add_argument("--eval-max-images", type=int, default=100)
    p.add_argument("--eval-batch-size", type=int, default=16)
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--resume", default=None, help="last.pth of an interrupted run")
    return p.parse_args()


def detection_loss(criterion, raw, targets):
    value = criterion(raw, targets)
    value = value[0] if isinstance(value, (tuple, list)) else value
    return value.sum() if value.ndim else value


@torch.no_grad()
def validate(model, detector, dataset, conditions, args, device):
    model.eval()
    total = 0.0
    for condition in conditions:
        dataset.set_condition(condition)
        metric = DetectionMAP()
        for batch in make_loader(dataset, args.eval_batch_size, args.workers):
            degraded = batch["degraded"].to(device, non_blocking=True)
            restored = model(degraded, task_features=detector.forward_backbone(degraded))["restored"]
            boxes, scores = decode_detector_output(detector.forward_rest(detector.forward_backbone(restored)), detector)
            metric.update(boxes, scores, batch)
        total += metric.compute()["map_50_95"]
    model.train()
    return total / len(conditions)


def save(path, model, optimizer, scheduler, epoch, best, val_map):
    torch.save({"format": CHECKPOINT_FORMAT, "pixel_restorer_config": model.config.to_dict(),
                "pixel_restorer": model.state_dict(), "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(), "epoch": epoch, "best": best,
                "meta": {"stage": "pixel-restorer pretraining", "epoch": epoch, "val_map": val_map}}, path)


def main():
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    device = torch.device(args.device)
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    splits = ROOT / "splits" / args.dataset
    protocol = AquaTXProtocol()

    train_set = AquaTXOnline(args.image_dir, args.train_annotation or splits / "instances_train.json",
                             protocol=protocol, mix=SamplingMix(*args.mix), augment=True,
                             horizontal_flip_probability=0.5, vertical_flip_probability=0.5, seed=args.seed)
    val_set = AquaTXOnline(args.image_dir, args.val_annotation or splits / "instances_val.json",
                           protocol=protocol, condition=protocol.compound[0], max_images=args.eval_max_images,
                           seed=args.seed)
    loader = make_loader(train_set, args.batch_size, args.workers, shuffle=True, drop_last=True)

    detector = FrozenYOLO(args.detector).to(device).eval()
    detector.verify_class_names(train_set.names)
    criterion = detector.get_criterion()
    channels = {k: v.shape[1] for k, v in detector.forward_backbone(torch.zeros(1, 3, 128, 128, device=device)).items()}
    model = PixelRestorer(PixelRestorerConfig(p3_channels=channels[P3_LAYER], p4_channels=channels[P4_LAYER])).to(device)
    objective = PixelRestorerObjective(PretrainWeights())
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)
    start_epoch, best = 1, -1.0
    if args.resume:
        state = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(state["pixel_restorer"])
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        start_epoch, best = state["epoch"] + 1, state["best"]
    print(f"pixel restorer parameters: {sum(p.numel() for p in model.parameters()):,}")

    log = save_dir / "train.csv"
    keys = ("total", "pixel", "gradient", "feature", "detection", "damage", "identity", "feature_no_harm", "detection_no_harm", "spatial_gate", "spatial_no_harm", "guidance_l1",
            "guidance_direction", "guidance_improvement", "guidance_clean_zero", "delta_saturation")
    if start_epoch == 1:
        with log.open("w", newline="") as handle:
            csv.writer(handle).writerow(["epoch", "lr", "det_scale", *keys, "grad_norm", "val_map", "best"])

    for epoch in range(start_epoch, args.epochs + 1):
        model.train()
        warmup = epoch <= args.warmup_epochs
        teacher = epoch <= args.teacher_epochs
        det_scale = detection_scale(epoch, args.warmup_epochs, args.ramp_start, args.ramp_end)
        sums = dict.fromkeys(keys, 0.0)
        grad_sum = 0.0
        for batch in tqdm(loader, desc=f"epoch {epoch:02d}", leave=False):
            degraded = batch["degraded"].to(device, non_blocking=True)
            target = batch["target"].to(device, non_blocking=True)
            clean_mask = transmission_free_mask(degraded, target)
            targets = yolo_targets(batch, device)
            optimizer.zero_grad(set_to_none=True)
            with torch.no_grad():
                paired = detector.forward_backbone(torch.cat((target, degraded)))
                clean_feat = {k: v[: len(target)] for k, v in paired.items()}
                degraded_feat = {k: v[len(target):] for k, v in paired.items()}
                spatial_target = spatial_task_target(clean_feat, degraded_feat)
            spatial_gate_override = spatial_target if teacher else None
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
                output = model(degraded, task_features=degraded_feat, spatial_gate_override=spatial_gate_override)
                restored_feat = detector.forward_backbone(output["restored"])
                if warmup:
                    det, raw_det = output["restored"].new_zeros(()), None
                else:
                    det = detection_loss(criterion, detector.forward_rest(restored_feat), targets)
                    with torch.no_grad():
                        raw_det = detection_loss(criterion, detector.forward_rest(degraded_feat), targets).detach()
                losses = objective(output, degraded, target, clean_feat, degraded_feat, restored_feat,
                                   targets["batch_idx"], targets["bboxes"], clean_mask, det, raw_det,
                                   det_scale=det_scale, warmup=warmup, warmup_feature_scale=args.warmup_feature_scale)
            if not torch.isfinite(losses["total"]):
                raise FloatingPointError(f"non-finite loss at epoch {epoch}")
            losses["total"].backward()
            grad_sum += float(torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=args.grad_clip))
            optimizer.step()
            for k in keys:
                sums[k] += float(losses[k].detach())
        scheduler.step()

        val_map = None
        if epoch % args.eval_interval == 0 or epoch == args.epochs:
            val_map = validate(model, detector, val_set, protocol.compound, args, device)
            if val_map > best:
                best = val_map
                save(save_dir / "best.pth", model, optimizer, scheduler, epoch, best, val_map)
        save(save_dir / "last.pth", model, optimizer, scheduler, epoch, best, val_map)
        n = max(len(loader), 1)
        with log.open("a", newline="") as handle:
            csv.writer(handle).writerow([epoch, scheduler.get_last_lr()[0], det_scale,
                                         *(sums[k] / n for k in keys), grad_sum / n,
                                         "" if val_map is None else val_map, best])
        print(f"epoch {epoch:02d} total={sums['total'] / n:.4f} det_scale={det_scale:.3f}"
              + ("" if val_map is None else f" val27={val_map:.4f} best={best:.4f}"))


if __name__ == "__main__":
    main()
