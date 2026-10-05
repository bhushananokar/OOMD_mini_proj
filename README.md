# LeafLens — crop disease detection

Upload a leaf photo, get a diagnosis plus treatment advice.

## Run

```
python -m venv .venv
.venv\Scripts\pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
.venv\Scripts\pip install transformers pillow fastapi uvicorn python-multipart numpy
cd backend
..\.venv\Scripts\python -m uvicorn app:app --port 8000
```

Open http://localhost:8000
