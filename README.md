# LeafLens — crop disease detection

Upload a leaf photo, get a diagnosis plus treatment advice.

- Model: `linkanjarad/mobilenet_v2_1.0_224-plant-disease-identification` (MobileNetV2 fine-tuned on the PlantVillage Kaggle dataset, 38 classes). Saved locally in `backend/model/`.
- Backend: FastAPI (`backend/app.py`), treatment info in `backend/knowledge.py`.
- Frontend: `static/index.html`.

## Run

```
python -m venv .venv
.venv\Scripts\pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
.venv\Scripts\pip install transformers pillow fastapi uvicorn python-multipart numpy
cd backend
..\.venv\Scripts\python -m uvicorn app:app --port 8000
```

Open http://localhost:8000
