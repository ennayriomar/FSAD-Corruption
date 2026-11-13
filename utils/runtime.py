"""Deterministic FP32 execution. Import this module before torch (the environment variables must be set first)."""
from __future__ import annotations

import os
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
os.environ["XFORMERS_DISABLED"] = "1"
for _name in ["OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"]:
    os.environ[_name] = "4"

from utils import IntegrityError


def configure(device: str = "cuda") -> dict:
    """FP32 only: no TF32, no mixed precision, deterministic kernels and the math attention backend."""
    import random
    import numpy as np
    import torch
    random.seed(20261104)
    np.random.seed(20261104)
    torch.manual_seed(20261104)
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True, warn_only=False)
    torch.set_float32_matmul_precision("highest")
    torch.backends.cuda.enable_flash_sdp(False)
    torch.backends.cuda.enable_mem_efficient_sdp(False)
    torch.backends.cuda.enable_math_sdp(True)
    torch.backends.cuda.enable_cudnn_sdp(False)
    if device == "cuda" and not torch.cuda.is_available():
        raise IntegrityError("CUDA is unavailable; refusing to fall back to CPU silently (use --device cpu).")
    info = {"torch": torch.__version__, "cuda_build": torch.version.cuda,
            "cudnn_version": torch.backends.cudnn.version(), "inference_dtype": "float32",
            "tf32": False, "deterministic_algorithms": True, "sdpa_backend": "math", "backend": device}
    if device == "cuda":
        prop = torch.cuda.get_device_properties(0)
        info.update(name=prop.name, compute_capability=[prop.major, prop.minor])
    return info
