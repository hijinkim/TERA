"""Frozen YOLOv8 wrapper exposing the intermediate pyramids TERA needs.

* forward_backbone      -> backbone P3/P4/P5 (layers 4/6/9); P3/P4 guide the pixel restorer (E_d).
* forward_detect_inputs -> late neck pyramid N3/N4/N5 consumed by the Detect head.
* forward_detect        -> raw (pre-NMS) Detect output from an arbitrary pyramid.

All parameters and BatchNorm statistics stay frozen, but gradients still flow
to the *input*, so restorers placed in front of / inside the detector can be
trained through it.
"""

from __future__ import annotations

from types import SimpleNamespace

import torch
import torch.nn as nn

BACKBONE_LAYERS = (4, 6, 9)  # P3, P4, P5
_BATCH_NORM = (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)


class FrozenYOLO(nn.Module):
    def __init__(self, checkpoint: str) -> None:
        super().__init__()
        from ultralytics import YOLO

        yolo = YOLO(checkpoint)
        self.model = yolo.model
        self.layers = self.model.model
        self.save_list = self.model.save
        self.names = yolo.names
        self.detect = self.layers[-1]
        self.requires_grad_(False)
        self._criterion = None
        self._set_raw_mode()

    # Ultralytics' Detect returns raw training-format tensors only in train mode;
    # keep Detect in train mode and every BatchNorm in eval mode.
    def _set_raw_mode(self) -> None:
        self.detect.train()
        for module in self.modules():
            if isinstance(module, _BATCH_NORM):
                module.eval()

    def train(self, mode: bool = True) -> "FrozenYOLO":
        super().train(mode)
        self._set_raw_mode()
        return self

    def eval(self) -> "FrozenYOLO":
        return self.train(False)

    def get_criterion(self):
        """Ultralytics v8DetectionLoss (box + cls + DFL) for the frozen model."""
        if self._criterion is None:
            from ultralytics.utils.loss import v8DetectionLoss

            defaults = {"box": 7.5, "cls": 0.5, "dfl": 1.5}
            args = getattr(self.model, "args", None)
            if args is None:
                args = SimpleNamespace(**defaults)
            elif isinstance(args, dict):
                args = SimpleNamespace(**{**defaults, **args})
            else:
                for key, value in defaults.items():
                    if not hasattr(args, key):
                        setattr(args, key, value)
            self.model.args = args
            self._criterion = v8DetectionLoss(self.model)
        return self._criterion

    def verify_class_names(self, expected: dict[int, str]) -> None:
        actual = {int(k): str(v) for k, v in self.names.items()}
        expected = {int(k): str(v) for k, v in expected.items()}
        if actual != expected:
            raise ValueError(f"class order mismatch: dataset={expected}, detector={actual}")

    def detect_input_layers(self) -> tuple[int, ...]:
        sources = self.detect.f
        return (int(sources),) if isinstance(sources, int) else tuple(int(s) for s in sources)

    def _run(self, current, saved: dict, start: int, stop: int) -> object:
        for index in range(start, stop):
            layer = self.layers[index]
            if hasattr(layer, "f") and layer.f != -1:
                current = (saved[layer.f] if isinstance(layer.f, int)
                           else [current if s == -1 else saved[s] for s in layer.f])
            current = layer(current)
            if index in self.save_list:
                saved[index] = current
        return current

    def forward_backbone(self, image: torch.Tensor) -> dict[int, torch.Tensor]:
        saved: dict[int, torch.Tensor] = {}
        intercepted: dict[int, torch.Tensor] = {}
        current = image
        for index in range(max(BACKBONE_LAYERS) + 1):
            current = self._run(current, saved, index, index + 1)
            if index in BACKBONE_LAYERS:
                intercepted[index] = current
        return intercepted

    def _neck_outputs(self, saved: dict) -> dict[int, torch.Tensor]:
        sources = self.detect_input_layers()
        missing = [s for s in sources if s not in saved]
        if missing:
            raise RuntimeError(f"Detect input layers were not saved: {missing}")
        return {s: saved[s] for s in sources}

    def forward_detect_inputs(self, image: torch.Tensor) -> dict[int, torch.Tensor]:
        self._set_raw_mode()
        saved: dict[int, torch.Tensor] = {}
        self._run(image, saved, 0, len(self.layers) - 1)
        return self._neck_outputs(saved)

    def forward_detect_inputs_from_backbone(self, backbone: dict[int, torch.Tensor]) -> dict[int, torch.Tensor]:
        """Run only the neck, reusing backbone features from forward_backbone."""
        self._set_raw_mode()
        saved = dict(backbone)
        self._run(saved[max(BACKBONE_LAYERS)], saved, max(BACKBONE_LAYERS) + 1, len(self.layers) - 1)
        return self._neck_outputs(saved)

    def forward_rest(self, backbone: dict[int, torch.Tensor]):
        """Neck + Detect head from backbone features (raw Detect output)."""
        self._set_raw_mode()
        saved = dict(backbone)
        return self._run(saved[max(BACKBONE_LAYERS)], saved, max(BACKBONE_LAYERS) + 1, len(self.layers))

    def forward_detect(self, features: dict[int, torch.Tensor]):
        """Frozen Detect head on an externally supplied N3/N4/N5 pyramid."""
        self._set_raw_mode()
        return self.detect([features[s] for s in self.detect_input_layers()])
