
# CLINICALMIND – Therapeutic Session Prediction Microservice

This microservice predicts recommended counts of therapeutic sessions based on disease and patient features. It includes:
- Synthetic dataset generator
- Classification + regression (hybrid) model trainer
- FastAPI service with endpoints for training, prediction, and health checks

## Structure
- `app/main.py` – FastAPI app
- `ml/train.py` – Data generation and model training
- `ml/model.py` – Model loading/prediction utilities
- `data/` – Generated synthetic dataset and saved models
- `tests/` – Simple smoke tests

## Quick Start

### 1) Install dependencies
```powershell
python -m venv .venv; .\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

### 2) Train models
```powershell
python -m ml.train --samples 2000 --save
```

### 3) Run the API
```powershell
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

### 4) Try predictions
```powershell
Invoke-RestMethod -Method Post -Uri http://localhost:8000/predict -ContentType 'application/json' -Body '{
  "disease": "diabetes",
  "symptoms": "fatigue, essoufflement, douleur thoracique",
  "age": 55,
  "severity": 3,
  "duration_days": 30,
  "hospitalizations_last_year": 1,
  "functional_score": 40,
  "bmi": 29
}'
```

## API
- `GET /health` – service status
- `POST /train` – triggers training (optional params)
- `POST /predict` – returns predicted `session_count` and disease class probabilities

## Notes
- Dataset is synthetic and for demonstration only.
- The classifier uses numeric + TF-IDF of `symptoms` to predict disease class; the regressor estimates session counts from numeric globals + disease.
