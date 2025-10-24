from fastapi import FastAPI
from pydantic import BaseModel
from predict_pointcloud import _load_model, _preprocess_sample
import numpy as np
import torch

app = FastAPI()


# ---------- 启动时加载模型 ----------
@app.on_event("startup")
def load_model_once():
    global MODEL, LABELS, ARGS, DEVICE
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    MODEL, LABELS, ARGS = _load_model("runs_overlock_light/best.pt")
    print(f"✅ 模型加载完成，类别: {LABELS}")


# ---------- 输入数据模型 ----------
class PredictIn(BaseModel):
    points: list[list[float]]


# ---------- 推理接口 ----------
@app.post("/predict")
def predict(body: PredictIn):
    arr = np.array(body.points, dtype=np.float32)

    # 归一化参数（和训练保持一致）
    xyz_min = [-1.0, -1.0, 0.0]
    xyz_max = [1.0, 1.0, 2.0]
    va_med = [0.0, 3.1588]
    va_iqr = [0.208334, 4.687345]

    arr_norm = _preprocess_sample(arr, xyz_min, xyz_max, va_med, va_iqr, target_n=54)

    x = torch.tensor(arr_norm, dtype=torch.float32, device=DEVICE).unsqueeze(0)
    with torch.no_grad():
        out = MODEL(x)
    pred_idx = out.argmax(1).item()
    return {"label": LABELS[pred_idx]}


# ---------- 健康检查 ----------
@app.get("/health")
def health():
    return {"ok": True}
