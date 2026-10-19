"""Model A: ConvNeXt-T. Model B: CLIP ViT-B/16 image encoder + language-anchored head (N1).

All models take float pixels in [0, 255] (B,3,H,W) and normalise internally, so the exact
same module is used for training, validation and the exported TorchScript package.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
CLIP_STD = (0.26862954, 0.26130258, 0.27577711)


def logit(p, eps=1e-4):
    p = torch.as_tensor(p, dtype=torch.float32).clamp(eps, 1 - eps)
    return torch.log(p / (1 - p))


class Normalize(nn.Module):
    def __init__(self, mean, std):
        super().__init__()
        self.register_buffer("mean", torch.tensor(mean).view(1, 3, 1, 1) * 255)
        self.register_buffer("std", torch.tensor(std).view(1, 3, 1, 1) * 255)

    def forward(self, x):
        return (x - self.mean) / self.std


class ConvNeXtPAR(nn.Module):
    def __init__(self, num_attr=40, pos_rate=None, dropout=0.2, pretrained=True):
        super().__init__()
        from torchvision.models import ConvNeXt_Tiny_Weights, convnext_tiny
        net = convnext_tiny(weights=ConvNeXt_Tiny_Weights.IMAGENET1K_V1 if pretrained else None)
        self.norm = Normalize(IMAGENET_MEAN, IMAGENET_STD)
        self.features = net.features
        self.avgpool = net.avgpool
        self.pre = nn.Sequential(net.classifier[0], net.classifier[1])  # LayerNorm2d, Flatten
        self.drop = nn.Dropout(dropout)
        self.fc = nn.Linear(768, num_attr)
        nn.init.normal_(self.fc.weight, std=0.01)
        with torch.no_grad():
            self.fc.bias.copy_(logit(pos_rate) if pos_rate is not None else torch.zeros(num_attr))

    def forward(self, x):
        x = self.features(self.norm(x))
        return self.fc(self.drop(self.pre(self.avgpool(x))))

    def param_groups(self, lr, head_lr, wd, **_):
        head = list(self.fc.parameters())
        hid = {id(p) for p in head}
        body = [p for p in self.parameters() if id(p) not in hid]
        return [{"params": body, "lr": lr, "weight_decay": wd},
                {"params": head, "lr": head_lr, "weight_decay": wd}]


def resize_clip_pos_embed(visual: nn.Module, h: int, w: int):
    """Bicubic-interpolate the ViT positional embedding grid to (h/patch, w/patch)."""
    pe = visual.positional_embedding.data  # (1 + gh*gw, C)
    gh0, gw0 = visual.grid_size
    ph, pw = visual.conv1.kernel_size
    gh, gw = h // ph, w // pw
    if (gh, gw) == (gh0, gw0):
        return
    cls, grid = pe[:1], pe[1:]
    grid = grid.reshape(1, gh0, gw0, -1).permute(0, 3, 1, 2)
    grid = F.interpolate(grid.float(), size=(gh, gw), mode="bicubic", align_corners=False)
    grid = grid.permute(0, 2, 3, 1).reshape(gh * gw, -1).to(pe.dtype)
    visual.positional_embedding = nn.Parameter(torch.cat([cls, grid], 0))
    visual.grid_size = (gh, gw)
    visual.image_size = (h, w)


class LanguageHead(nn.Module):
    """logits_k = s * cos(f, w_k) + b_k  (N1). w_k from CLIP text embeddings or random."""

    def __init__(self, dim, num_attr, text_weights=None, bias=None, init_scale=1 / 0.07, max_scale=100.0):
        super().__init__()
        if text_weights is None:
            w = torch.randn(num_attr, dim) / math.sqrt(dim)
        else:
            w = torch.as_tensor(text_weights, dtype=torch.float32)
        self.weight = nn.Parameter(F.normalize(w, dim=-1))
        self.log_scale = nn.Parameter(torch.tensor(math.log(init_scale)))
        self.max_log_scale = math.log(max_scale)
        self.bias = nn.Parameter(torch.zeros(num_attr) if bias is None else torch.as_tensor(bias, dtype=torch.float32))

    def forward(self, f):
        s = self.log_scale.clamp(max=self.max_log_scale).exp()
        return s * F.normalize(f, dim=-1) @ F.normalize(self.weight, dim=-1).t() + self.bias


class CLIPPAR(nn.Module):
    def __init__(self, num_attr=40, h=256, w=128, arch="ViT-B-16", pretrained="openai",
                 text_weights=None, head_bias=None, init_scale=1 / 0.07, load_pretrained=True):
        super().__init__()
        import open_clip
        model = open_clip.create_model(arch, pretrained=pretrained if load_pretrained else None)
        self.visual = model.visual
        resize_clip_pos_embed(self.visual, h, w)
        self.norm = Normalize(CLIP_MEAN, CLIP_STD)
        dim = self.visual.output_dim
        self.head = LanguageHead(dim, num_attr, text_weights, head_bias, init_scale)

    def features(self, x):
        return self.visual(self.norm(x))

    def forward(self, x):
        return self.head(self.features(x))

    def param_groups(self, lr, head_lr, wd, layer_decay=0.7, **_):
        """Layer-wise LR decay: last block gets lr, each block below * layer_decay."""
        v = self.visual
        blocks = v.transformer.resblocks
        n = len(blocks)
        groups = {}

        def add(p, depth, name):
            no_wd = p.ndim < 2 or "positional_embedding" in name or "class_embedding" in name
            scale = layer_decay ** (n + 1 - depth)
            key = (depth, no_wd)
            groups.setdefault(key, {"params": [], "lr": lr * scale, "weight_decay": 0.0 if no_wd else wd})
            groups[key]["params"].append(p)

        for name, p in v.named_parameters():
            if name.startswith("transformer.resblocks."):
                depth = int(name.split(".")[2]) + 1
            elif name.startswith(("ln_post", "proj")) or name == "proj":
                depth = n + 1
            else:  # conv1, class/positional embedding, ln_pre
                depth = 0
            add(p, depth, name)
        out = list(groups.values())
        out.append({"params": list(self.head.parameters()), "lr": head_lr, "weight_decay": 0.0})
        return out


def build_model(cfg, pos_rate=None, text_weights=None, head_bias=None, load_pretrained=True):
    if cfg["model"] == "convnext_t":
        return ConvNeXtPAR(40, pos_rate, cfg.get("dropout", 0.2), pretrained=load_pretrained)
    if cfg["model"] == "clip":
        return CLIPPAR(40, cfg["h"], cfg["w"], cfg.get("arch", "ViT-B-16"), cfg.get("pretrained", "openai"),
                       text_weights, head_bias, cfg.get("init_scale", 1 / 0.07), load_pretrained=load_pretrained)
    raise ValueError(cfg["model"])
