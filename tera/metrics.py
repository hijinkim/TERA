"""YOLOv8 raw-output decoding, NMS, and COCO mAP (torchmetrics / pycocotools backend)."""

from __future__ import annotations

import torch
from torchvision.ops import batched_nms

CONF_THRESHOLD = 0.001
NMS_IOU = 0.7
MAX_DETECTIONS = 300


def decode_raw_yolov8(raw: dict, strides, reg_max: int):
    """Raw Detect output -> (boxes xyxy [B, A, 4] in canvas pixels, class scores [B, A, K])."""
    distribution, class_logits, feature_maps = raw["boxes"], raw["scores"], raw["feats"]
    device, batch_size = distribution.device, distribution.shape[0]
    anchors, stride_per_anchor = [], []
    for stride, feature_map in zip(strides, feature_maps):
        height, width = feature_map.shape[-2:]
        yy, xx = torch.meshgrid(torch.arange(height, device=device), torch.arange(width, device=device), indexing="ij")
        anchors.append(torch.stack((xx.flatten() + 0.5, yy.flatten() + 0.5), dim=1))
        stride_per_anchor.append(torch.full((height * width,), float(stride), device=device, dtype=distribution.dtype))
    anchors, stride_per_anchor = torch.cat(anchors), torch.cat(stride_per_anchor)
    projection = torch.arange(reg_max, device=device, dtype=distribution.dtype)
    distances = distribution.view(batch_size, 4, reg_max, -1).softmax(dim=2)
    distances = (distances * projection.view(1, 1, -1, 1)).sum(dim=2)
    cx, cy = anchors[:, 0], anchors[:, 1]
    boxes = torch.stack(((cx - distances[:, 0]) * stride_per_anchor, (cy - distances[:, 1]) * stride_per_anchor,
                         (cx + distances[:, 2]) * stride_per_anchor, (cy + distances[:, 3]) * stride_per_anchor), dim=-1)
    return boxes, class_logits.sigmoid().transpose(1, 2)


def decode_detector_output(raw: dict, detector):
    return decode_raw_yolov8(raw, [float(s) for s in detector.detect.stride], int(detector.detect.reg_max))


def postprocess(boxes: torch.Tensor, scores: torch.Tensor, image_size: int = 640,
                conf: float = CONF_THRESHOLD, iou: float = NMS_IOU, max_det: int = MAX_DETECTIONS) -> list[dict]:
    """Per-image class-aware NMS. Returns [{"boxes", "scores", "labels"}] on CPU."""
    detections = []
    for sample_boxes, sample_scores in zip(boxes, scores):
        confidence, labels = sample_scores.float().max(dim=-1)
        selected = confidence > conf
        sample_boxes = sample_boxes[selected].float().clamp(0, image_size)
        confidence, labels = confidence[selected], labels[selected]
        keep = batched_nms(sample_boxes, confidence, labels, iou)[:max_det]
        detections.append({"boxes": sample_boxes[keep].cpu(), "scores": confidence[keep].cpu(),
                           "labels": labels[keep].long().cpu()})
    return detections


def xywhn_to_xyxy(boxes: torch.Tensor, image_size: int) -> torch.Tensor:
    x, y, w, h = boxes.unbind(dim=1)
    return torch.stack(((x - w / 2) * image_size, (y - h / 2) * image_size,
                        (x + w / 2) * image_size, (y + h / 2) * image_size), dim=1)


class DetectionMAP:
    """COCO-style mAP accumulator over letterboxed 640x640 predictions."""

    def __init__(self, image_size: int = 640) -> None:
        from torchmetrics.detection import MeanAveragePrecision

        self.image_size = image_size
        self.metric = MeanAveragePrecision(box_format="xyxy", iou_type="bbox",
                                           max_detection_thresholds=[1, 10, MAX_DETECTIONS],
                                           extended_summary=True)

    def update(self, boxes: torch.Tensor, scores: torch.Tensor, batch: dict) -> None:
        predictions = postprocess(boxes, scores, self.image_size)
        targets = []
        for i in range(len(predictions)):
            mask = batch["batch_idx"].long() == i
            targets.append({"boxes": xywhn_to_xyxy(batch["bboxes"][mask], self.image_size),
                            "labels": batch["cls"][mask].long()})
        self.metric.update(predictions, targets)

    def compute(self) -> dict:
        result = self.metric.compute()
        precision = result["precision"]  # [T, R, K, A, M]
        recall = result["recall"]        # [T, K, A, M]

        def mean_valid(values):
            valid = values[values > -1]
            return float(valid.mean()) if valid.numel() else -1.0

        def ap(iou_index=None, area_index=0):
            values = precision[:, :, :, area_index, -1]
            return mean_valid(values if iou_index is None else values[iou_index])

        def ar(area_index, maxdet_index=-1):
            return mean_valid(recall[:, :, area_index, maxdet_index])

        def class_ap(k, iou_index=None):
            values = precision[:, :, k, 0, -1]
            return mean_valid(values if iou_index is None else values[iou_index])

        class_ids = [int(c) for c in result["classes"].tolist()]
        return {
            "map_50_95": ap(), "map_50": ap(0), "map_75": ap(5),
            "map_small": ap(None, 1), "map_medium": ap(None, 2), "map_large": ap(None, 3),
            "mar_1": ar(0, 0), "mar_10": ar(0, 1), "mar_max": ar(0, 2),
            "mar_small": ar(1), "mar_medium": ar(2), "mar_large": ar(3),
            "ap_per_class": {str(c): class_ap(k) for k, c in enumerate(class_ids)},
            "ap50_per_class": {str(c): class_ap(k, 0) for k, c in enumerate(class_ids)},
        }


METRIC_KEYS = ("map_50_95", "map_50", "map_75", "map_small", "map_medium", "map_large",
               "mar_1", "mar_10", "mar_max", "mar_small", "mar_medium", "mar_large")
