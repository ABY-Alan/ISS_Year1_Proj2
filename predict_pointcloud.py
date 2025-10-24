import numpy as np
import joblib
import torch
import os
import contextlib
from point_overlock_light import PointOverLoCKLight, PointFrameDataset

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DEVICE = torch.device(os.getenv("TORCH_DEVICE", DEVICE.type))

try:
    AMP_DTYPE = (
        torch.bfloat16
        if (DEVICE.type == "cuda" and torch.cuda.is_bf16_supported())
        else torch.float16
    )
except Exception:
    AMP_DTYPE = torch.float16

torch.set_float32_matmul_precision("high")
if DEVICE.type == "cuda":
    torch.backends.cudnn.benchmark = True


def _load_model(model_path: str):
    ckpt = torch.load(model_path, map_location=DEVICE, weights_only=False)
    labels = ckpt["labels"]
    args = ckpt["args"]
    num_classes = len(labels)

    model = PointOverLoCKLight(
        num_classes=num_classes,
        use_overview=not args.get("no_overview", False),
        use_gate=(not args.get("no_overview", False))
        and (not args.get("no_gate", False)),
        in_channels=5,
        k=args.get("k", 10),
        feat_dim=args.get("feat_dim", 256),
        ctx_dim=args.get("ctx_dim", 256),
        dropout=args.get("dropout", 0.4),
    )

    dummy = torch.zeros(2, 54, 5, dtype=torch.float32, device=DEVICE)
    model.train()
    _ = model(dummy)

    model.load_state_dict(ckpt["model"])
    model.to(DEVICE, non_blocking=True).eval()
    return model, labels, args


def _preprocess_sample(arr: np.ndarray, xyz_min, xyz_max, va_med, va_iqr, target_n=54):
    arr = arr.astype(np.float32)
    xyz = arr[:, :3]
    va = arr[:, 3:5]

    xyz_min = np.asarray(xyz_min, dtype=np.float32)
    xyz_max = np.asarray(xyz_max, dtype=np.float32)
    va_med = np.asarray(va_med, dtype=np.float32)
    va_iqr = np.asarray(va_iqr, dtype=np.float32)

    xyz_n = 2.0 * (xyz - xyz_min) / (xyz_max - xyz_min + 1e-8) - 1.0
    va_n = (va - va_med) / (va_iqr + 1e-6)

    out = np.concatenate([xyz_n, va_n], axis=1)

    n = out.shape[0]
    if n < target_n:
        idx = np.random.choice(n, target_n - n, replace=True)
        out = np.concatenate([out, out[idx]], axis=0)
    elif n > target_n:
        out = out[np.random.choice(n, target_n, replace=False)]

    return out
