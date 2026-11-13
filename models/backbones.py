"""Frozen self-supervised encoders. Both return 1,024 L2-normalized 384-d patch tokens (a 32x32 grid)."""
from __future__ import annotations

import hashlib
from pathlib import Path

from utils import runtime  # noqa: F401  (deterministic environment before torch)
import numpy as np
import torch
import torch.nn.functional as F

from utils import IntegrityError

MEAN, STD = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
DINOV2_SHA256 = "b938bf1bc15cd2ec0feacfe3a1bb553fe8ea9ca46a7e1d8d00217f29aef60cd9"   # dinov2_vits14_pretrain.pth
DINOV3_HF_ID = "timm/vit_small_patch16_dinov3.lvd1689m"
DINOV3_SHA256 = "2a1ec16ae28ffa07bc0ead0241ee7df9fc26451fe6f9f839b7b3afa0a906b040"   # model.safetensors


def sha256_file(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


class _Encoder:
    input_size: int

    def __init__(self, device):
        self.device = device
        self.mean = torch.tensor(MEAN, device=device).view(1, 3, 1, 1)
        self.std = torch.tensor(STD, device=device).view(1, 3, 1, 1)

    def tokens(self, x):
        raise NotImplementedError

    @torch.inference_mode()
    def extract(self, image: np.ndarray) -> np.ndarray:
        s = self.input_size
        if image.shape != (3, s, s) or image.dtype != np.float32 or not np.isfinite(image).all():
            raise IntegrityError(f"Expected a finite float32 image of shape (3,{s},{s}).")
        x = torch.from_numpy(image).unsqueeze(0).to(self.device)
        features = F.normalize(self.tokens((x - self.mean) / self.std).float(), p=2, dim=-1)[0].cpu().numpy().copy()
        if features.shape != (1024, 384) or not np.isfinite(features).all():
            raise IntegrityError("Unexpected patch-feature shape or non-finite features.")
        return features


class DINOv2(_Encoder):
    """DINOv2 ViT-S/14 at 448 x 448. `repo` is the official GitHub repo or a local clone of it;
    `checkpoint` is an optional local dinov2_vits14_pretrain.pth (otherwise downloaded by torch.hub)."""
    input_size = 448

    def __init__(self, device="cuda", repo="facebookresearch/dinov2", checkpoint=None):
        super().__init__(device)
        local = Path(repo).is_dir()
        self.model = torch.hub.load(repo, "dinov2_vits14", source="local" if local else "github",
                                    pretrained=checkpoint is None)
        if checkpoint is None:
            checkpoint = Path(torch.hub.get_dir()) / "checkpoints/dinov2_vits14_pretrain.pth"
        else:
            self.model.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=True), strict=True)
        digest = sha256_file(checkpoint) if Path(checkpoint).is_file() else None
        if digest is not None and digest != DINOV2_SHA256:
            print(f"warning: DINOv2 checkpoint SHA-256 {digest} differs from the paper's {DINOV2_SHA256}")
        self.model.eval().requires_grad_(False).to(device)
        self.identity = {"backbone": "dinov2_vits14", "input_size": 448, "checkpoint_sha256": digest}

    def tokens(self, x):
        return self.model.forward_features(x)["x_norm_patchtokens"]


class DINOv3(_Encoder):
    """DINOv3 ViT-S/16 at 512 x 512 (timm). Class and register tokens are dropped."""
    input_size = 512

    def __init__(self, device="cuda"):
        super().__init__(device)
        import timm
        from huggingface_hub import hf_hub_download
        from safetensors.torch import load_file
        path = hf_hub_download(DINOV3_HF_ID, "model.safetensors")
        digest = sha256_file(path)
        if digest != DINOV3_SHA256:
            print(f"warning: DINOv3 checkpoint SHA-256 {digest} differs from the paper's {DINOV3_SHA256}")
        self.model = timm.create_model("vit_small_patch16_dinov3", pretrained=False)
        self.model.load_state_dict(load_file(path), strict=True)
        self.model.eval().requires_grad_(False).to(device)
        self.prefix = self.model.num_prefix_tokens
        self.identity = {"backbone": "dinov3_vits16", "input_size": 512, "hf_id": DINOV3_HF_ID,
                         "checkpoint_sha256": digest, "prefix_tokens_dropped": self.prefix}

    def tokens(self, x):
        return self.model.forward_features(x)[:, self.prefix:]


def build(name: str, device: str = "cuda", **kwargs) -> _Encoder:
    if name == "dinov2":
        return DINOv2(device, **kwargs)
    if name == "dinov3":
        return DINOv3(device)
    raise ValueError(name)
