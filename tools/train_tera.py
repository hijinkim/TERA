#!/usr/bin/env python3
"""Joint TERA training (paper Sec. 4.3 / 5.1): pixel + evidence restorer, detector frozen.

L = L_GT + lambda_KD * L_KD + lambda_id * L_id + lambda_res * L_res        (Eq. 11)

* L_GT : YOLOv8 box + cls + DFL loss of the corrected prediction vs. ground truth.
* L_KD : pre-transmission teacher distillation restricted to the foreground set F
         from the task-aligned assigner (Eq. 12; BCE on cls, lambda_dfl=0.1 SmoothL1 on DFL).
* L_id : sum_l ||r_ref^l||_1 on the transmission-free inputs of the batch (Eq. 13).
* L_res: sum_l ||r^l||_1 on the transmission-degraded inputs of the batch (Eq. 14).
r^l is the normalized residual of Eq. 9; ||.||_1 is averaged over elements and summed
over N3/N4. Transmission-free inputs are identified from the image pair itself
(received == pre-transmission), so no transmission metadata is used.

The pixel restorer is initialized from a pretrained checkpoint and fine-tuned
jointly; the evidence restorer starts from zero (identity) residual heads.

Paper settings:
  DUO-TX : --dataset duo  --batch-size 20 --max-steps 6000
  RUOD-TX: --dataset ruod --batch-size 16 --max-steps 5000
"""

from __future__ import annotations

import argparse
import csv
import random
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tera.data import AquaTXOnline, make_loader, transmission_free_mask, yolo_targets  # noqa: E402
from tera.detector import FrozenYOLO  # noqa: E402
from tera.evidence_restorer import EvidenceRestorer, EvidenceRestorerConfig  # noqa: E402
from tera.metrics import DetectionMAP, decode_detector_output  # noqa: E402
from tera.model import load_pixel_restorer, save_tera_checkpoint  # noqa: E402
from tera.transmission import AquaTXProtocol, SamplingMix  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset", choices=("duo", "ruod"), required=True)
    p.add_argument("--image-dir", required=True, help="original training images (train + val split)")
    p.add_argument("--train-annotation", default=None, help="defaults to splits/<dataset>/instances_train.json")
    p.add_argument("--val-annotation", default=None, help="defaults to splits/<dataset>/instances_val.json")
    p.add_argument("--detector", required=True, help="frozen Clean-YOLO weights")
    p.add_argument("--pixel-restorer", required=True, help="pretrained pixel restorer (tera-v1 checkpoint)")
    p.add_argument("--neck-stats", required=True, help="output of tools/precompute_neck_stats.py")
    p.add_argument("--save-dir", required=True)
    # evidence restorer
    p.add_argument("--width", type=int, default=96)
    p.add_argument("--blocks-per-level", type=int, default=2)
    p.add_argument("--norm-mode", choices=("shared", "independent"), default="shared")
    p.add_argument("--input-mode", choices=("full", "signed", "rd", "r_only"), default="full")
    p.add_argument("--no-cross-scale", action="store_true")
    p.add_argument("--unbounded", action="store_true")
    p.add_argument("--freeze-pixel-restorer", action="store_true", help="train only the evidence restorer")
    # objective
    p.add_argument("--lambda-gt", type=float, default=1.0)
    p.add_argument("--lambda-kd", type=float, default=1.0)
    p.add_argument("--lambda-dfl", type=float, default=0.1)
    p.add_argument("--lambda-id", type=float, default=0.5)
    p.add_argument("--lambda-res", type=float, default=1e-3)
    # optimization
    p.add_argument("--batch-size", type=int, default=20)
    p.add_argument("--max-steps", type=int, default=6000)
    p.add_argument("--lr", type=float, default=1e-3, help="evidence restorer")
    p.add_argument("--lr-pixel", type=float, default=3e-4, help="pixel restorer")
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--grad-clip", type=float, default=5.0)
    p.add_argument("--horizontal-flip-prob", type=float, default=0.5)
    p.add_argument("--mix", type=float, nargs=4, default=(0.10, 0.10, 0.10, 0.70),
                   metavar=("CLEAN", "RESIZE", "CODEC", "COMPOUND"))
    # validation / checkpoint selection (validation split only)
    p.add_argument("--eval-interval", type=int, default=500)
    p.add_argument("--eval-max-images", type=int, default=100)
    p.add_argument("--eval-batch-size", type=int, default=64)
    p.add_argument("--log-interval", type=int, default=100)
    p.add_argument("--workers", type=int, default=12)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default="cuda:0")
    return p.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def foreground_kd(student: dict, teacher: dict, fg_mask: torch.Tensor, lambda_dfl: float) -> torch.Tensor:
    """Eq. 12: teacher distillation on the assigner's foreground anchors only."""
    if not fg_mask.any():
        return student["scores"].new_zeros(())
    s_cls, t_cls = student["scores"].permute(0, 2, 1), teacher["scores"].permute(0, 2, 1).sigmoid().detach()
    s_dfl, t_dfl = student["boxes"].permute(0, 2, 1), teacher["boxes"].permute(0, 2, 1).detach()
    cls = F.binary_cross_entropy_with_logits(s_cls[fg_mask], t_cls[fg_mask])
    dfl = F.smooth_l1_loss(s_dfl[fg_mask], t_dfl[fg_mask], beta=1.0)
    return cls + lambda_dfl * dfl


def sentinel_conditions(protocol: AquaTXProtocol):
    """Three seen-compound conditions (severe / medium / mild) used for checkpoint selection."""
    keep = {(256, 0.3), (384, 0.6), (512, 1.0)}
    return [c for c in protocol.compound if (c.long_side, c.target_bpp) in keep]


@torch.inference_mode()
def validate(detector, pixel, evidence, dataset, conditions, args, device):
    """Macro mAP50:95 of the pixel-only and TERA paths over the sentinel conditions."""
    totals = {"pixel_only": 0.0, "tera": 0.0}
    for condition in conditions:
        dataset.set_condition(condition)
        metrics = {k: DetectionMAP() for k in totals}
        for batch in make_loader(dataset, args.eval_batch_size, args.workers):
            degraded = batch["degraded"].to(device, non_blocking=True)
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
                backbone = detector.forward_backbone(degraded)
                z_d = detector.forward_detect_inputs_from_backbone(backbone)
                z_r = detector.forward_detect_inputs(pixel(degraded, task_features=backbone)["restored"])
                z_c = evidence(z_d, z_r)["features"]
                # one Detect call on the concatenated pyramids, as in the paper runs
                joined = detector.forward_detect({l: torch.cat((z_r[l], z_c[l])) for l in z_r})
                boxes, scores = decode_detector_output(joined, detector)
            n = len(degraded)
            metrics["pixel_only"].update(boxes[:n], scores[:n], batch)
            metrics["tera"].update(boxes[n:], scores[n:], batch)
        for k in totals:
            totals[k] += metrics[k].compute()["map_50_95"] / len(conditions)
    return totals


def main():
    args = parse_args()
    set_seed(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    splits = ROOT / "splits" / args.dataset
    protocol = AquaTXProtocol()

    train_set = AquaTXOnline(args.image_dir, args.train_annotation or splits / "instances_train.json",
                             protocol=protocol, mix=SamplingMix(*args.mix), augment=True,
                             horizontal_flip_probability=args.horizontal_flip_prob, seed=args.seed)
    val_set = AquaTXOnline(args.image_dir, args.val_annotation or splits / "instances_val.json",
                           protocol=protocol, condition=protocol.clean, max_images=args.eval_max_images,
                           seed=args.seed)
    loader = make_loader(train_set, args.batch_size, args.workers, shuffle=True, drop_last=True)

    detector = FrozenYOLO(args.detector).to(device).eval()
    detector.verify_class_names(train_set.names)
    criterion = detector.get_criterion()
    pixel = load_pixel_restorer(args.pixel_restorer, device)
    e2e = not args.freeze_pixel_restorer
    pixel.requires_grad_(e2e).train(e2e)

    stats = torch.load(args.neck_stats, map_location="cpu", weights_only=False)
    layers = detector.detect_input_layers()
    probe = detector.forward_detect_inputs(torch.zeros(1, 3, 128, 128, device=device))
    config = EvidenceRestorerConfig(
        n3_layer=layers[0], n4_layer=layers[1], n5_layer=layers[2],
        n3_channels=probe[layers[0]].shape[1], n4_channels=probe[layers[1]].shape[1],
        width=args.width, blocks_per_level=args.blocks_per_level, norm_mode=args.norm_mode,
        input_mode=args.input_mode, use_cross_scale=not args.no_cross_scale, bounded=not args.unbounded)
    evidence = EvidenceRestorer(config, stats["stats"]).to(device).train()

    groups = [{"params": evidence.parameters(), "lr": args.lr}]
    if e2e:
        groups.append({"params": [p for p in pixel.parameters() if p.requires_grad], "lr": args.lr_pixel})
    optimizer = AdamW(groups, lr=args.lr, weight_decay=args.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.max_steps, eta_min=args.lr * 0.05)
    trainable = [p for g in optimizer.param_groups for p in g["params"]]
    print(f"device={device} batch={args.batch_size} steps={args.max_steps} "
          f"evidence_params={sum(p.numel() for p in evidence.parameters()):,} "
          f"pixel_params={sum(p.numel() for p in pixel.parameters()):,} (trainable={e2e})")

    def corrected_prediction(degraded):
        with torch.no_grad():
            backbone = detector.forward_backbone(degraded)
            z_d = detector.forward_detect_inputs_from_backbone(backbone)
        with torch.set_grad_enabled(e2e):
            z_r = detector.forward_detect_inputs(pixel(degraded, task_features=backbone)["restored"])
        out = evidence(z_d, z_r)
        z_c = evidence.correct(z_r, out) if e2e else out["features"]  # keep the path to R_theta when e2e
        return detector.forward_detect(z_c), out

    log_path = save_dir / "train.csv"
    with log_path.open("w", newline="") as handle:
        csv.writer(handle).writerow(["step", "lr", "total", "gt", "kd", "id", "res", "grad_norm",
                                     "val_pixel_only", "val_tera", "best"])
    conditions = sentinel_conditions(protocol)
    best = -float("inf")
    stream = iter(())
    for step in tqdm(range(1, args.max_steps + 1), desc="tera"):
        try:
            batch = next(stream)
        except StopIteration:
            stream = iter(loader)
            batch = next(stream)
        degraded = batch["degraded"].to(device, non_blocking=True)
        target = batch["target"].to(device, non_blocking=True)
        clean_mask = transmission_free_mask(degraded, target)
        targets = yolo_targets(batch, device)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
            with torch.no_grad():
                teacher = detector.forward_detect(detector.forward_detect_inputs(target))
            student, out = corrected_prediction(degraded)
            assigned, gt_vector, _ = criterion.get_assigned_targets_and_loss(criterion.parse_output(student), targets)
            loss_gt = gt_vector.sum()
            loss_kd = foreground_kd(student, teacher, assigned[0], args.lambda_dfl)
            r3, r4 = out["norm_n3"], out["norm_n4"]  # r^l, Eq. 9
            loss_id = r3.new_zeros(())
            loss_res = r3.new_zeros(())
            if clean_mask.any():  # Eq. 13
                loss_id = r3[clean_mask].abs().mean() + r4[clean_mask].abs().mean()
            if (~clean_mask).any():  # Eq. 14
                loss_res = r3[~clean_mask].abs().mean() + r4[~clean_mask].abs().mean()
            total = (args.lambda_gt * loss_gt + args.lambda_kd * loss_kd
                     + args.lambda_id * loss_id + args.lambda_res * loss_res)
        if not torch.isfinite(total):  # rare e2e spike: skip the step
            optimizer.zero_grad(set_to_none=True)
            scheduler.step()
            continue
        total.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(trainable, args.grad_clip)
        optimizer.step()
        scheduler.step()

        val = None
        if step % args.eval_interval == 0 or step == args.max_steps:
            evidence.eval()
            pixel.eval()
            val = validate(detector, pixel, evidence, val_set, conditions, args, device)
            evidence.train()
            pixel.train(e2e)
            meta = {"dataset": args.dataset, "step": step, "val_sentinel_map": val["tera"]}
            save_tera_checkpoint(save_dir / "last.pth", pixel, evidence, **meta)
            if val["tera"] > best:
                best = val["tera"]
                save_tera_checkpoint(save_dir / "best.pth", pixel, evidence, **meta)
            tqdm.write(f"step {step:5d} val pixel_only={val['pixel_only']:.4f} tera={val['tera']:.4f} best={best:.4f}")
        if step % args.log_interval == 0 or val is not None:
            with log_path.open("a", newline="") as handle:
                csv.writer(handle).writerow([
                    step, scheduler.get_last_lr()[0], float(total), float(loss_gt), float(loss_kd), float(loss_id),
                    float(loss_res), float(grad_norm), "" if val is None else val["pixel_only"],
                    "" if val is None else val["tera"], best if best > -float("inf") else ""])
    print(f"done. best val (sentinel) mAP50:95 = {best:.4f} -> {save_dir / 'best.pth'}")


if __name__ == "__main__":
    main()
