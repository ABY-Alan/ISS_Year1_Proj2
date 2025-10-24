# ISS_Year1_Proj2 - PointCloud Gesture Recognition API

This project provides a **FastAPI-based REST service** for predicting human actions 
from 3D mmWave radar point-cloud data.

## 🌟 Features
- PyTorch-based lightweight DGCNN backbone (OverLoCK Light)
- Supports both **CPU and CUDA inference**
- Mixed precision acceleration (bfloat16 / float16)
- Ready for **Railway deployment**

## 🚀 Run locally

```bash
pip install -r requirements.txt
python -m uvicorn app:app --reload --port 8080
