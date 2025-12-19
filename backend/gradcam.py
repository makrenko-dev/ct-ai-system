import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


def load_effnet_b0(pth_path):
    from torchvision.models import efficientnet_b0

    state = torch.load(pth_path, map_location="cpu")
    if isinstance(state, dict):
        state = state.get("state_dict", state)

    model = efficientnet_b0(weights=None)

    in_features = model.classifier[1].in_features
    num_classes = state["classifier.1.weight"].shape[0]
    model.classifier[1] = nn.Linear(in_features, num_classes)

    clean = {}
    for k, v in state.items():
        if k.startswith("module."):
            k = k[7:]
        clean[k] = v

    model.load_state_dict(clean, strict=True)
    model = model.float()
    model.eval()
    return model


def preprocess_crop(crop_bgr):
    crop_rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
    img = cv2.resize(crop_rgb, (224, 224))

    img = img.astype("float32") / 255.0
    img = (img - [0.485, 0.456, 0.406]) / [0.229, 0.224, 0.225]

    t = torch.from_numpy(img).float().permute(2, 0, 1).unsqueeze(0)
    return t


class GradCAM:
    def __init__(self, model, target_layer="features.7"):
        self.model = model
        self.layer = dict(model.named_modules())[target_layer]

        self.activations = None
        self.gradients = None

        self.layer.register_forward_hook(self._forward_hook)
        self.layer.register_full_backward_hook(self._backward_hook)

    def _forward_hook(self, m, i, o):
        self.activations = o

    def _backward_hook(self, m, gin, gout):
        self.gradients = gout[0]

    def generate(self, x):
        logits = self.model(x)
        class_idx = logits.argmax(dim=1).item()

        self.model.zero_grad()
        logits[:, class_idx].backward()

        A = self.activations
        G = self.gradients

        weights = G.mean(dim=(2, 3), keepdim=True)
        cam = (weights * A).sum(dim=1, keepdim=True)
        cam = F.relu(cam)

        cam = F.interpolate(
            cam,
            size=x.shape[-2:],
            mode="bilinear",
            align_corners=False
        )

        cam = cam[0, 0]
        cam -= cam.min()
        cam /= cam.max().clamp(min=1e-6)

        print("[GradCAM]", float(cam.min()), float(cam.max()), float(cam.mean()))

        return cam.detach().cpu().numpy()
