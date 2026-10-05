import io
from pathlib import Path

import numpy as np
import torch
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image, ImageOps
from transformers import AutoModelForImageClassification

from knowledge import INFO

ROOT = Path(__file__).resolve().parent
STATIC = ROOT.parent / "static"

model = AutoModelForImageClassification.from_pretrained(ROOT / "model").eval()
assert len(INFO) == model.config.num_labels, "knowledge base out of sync with model"

app = FastAPI(title="LeafLens")


def preprocess(img: Image.Image) -> torch.Tensor:
    # Matches the model's preprocessor: resize shortest edge 256, center crop 224, normalise to [-1, 1]
    img = ImageOps.exif_transpose(img).convert("RGB")
    w, h = img.size
    s = 256 / min(w, h)
    img = img.resize((max(256, round(w * s)), max(256, round(h * s))), Image.BILINEAR)
    w, h = img.size
    l, t = (w - 224) // 2, (h - 224) // 2
    img = img.crop((l, t, l + 224, t + 224))
    x = torch.from_numpy(np.asarray(img).copy()).permute(2, 0, 1).float() / 255
    return ((x - 0.5) / 0.5).unsqueeze(0)


def describe(i: int, conf: float) -> dict:
    crop, cond, kind, sev, about, tips = INFO[i]
    return {"crop": crop, "condition": cond, "kind": kind, "severity": sev,
            "about": about, "treatment": tips, "confidence": conf}


@app.post("/api/predict")
async def predict(file: UploadFile = File(...)):
    if not (file.content_type or "").startswith("image/"):
        raise HTTPException(400, "Please upload an image file.")
    try:
        img = Image.open(io.BytesIO(await file.read()))
    except Exception:
        raise HTTPException(400, "Could not read that image.")
    with torch.no_grad():
        probs = model(pixel_values=preprocess(img)).logits.softmax(-1)[0]
    top = probs.topk(3)
    results = [describe(int(i), float(p)) for p, i in zip(top.values, top.indices)]
    return {"top": results[0], "alternatives": results[1:]}


@app.get("/api/classes")
def classes():
    return sorted({(c[0]) for c in INFO})


@app.get("/")
def index():
    return FileResponse(STATIC / "index.html")


app.mount("/static", StaticFiles(directory=STATIC), name="static")
