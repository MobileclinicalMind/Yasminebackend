import argparse
import sys
from pathlib import Path
from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel, Field
from typing import Optional
import requests

# Add project root to sys.path
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from ml.model import models_exist, load_models, predict

# --------------------------
# FastAPI App
# --------------------------
app = FastAPI(title="CLINICALMIND – Therapeutic Session Prediction")

class TrainRequest(BaseModel):
    samples: int = Field(default=1500, ge=100, le=100000)
    seed: int = Field(default=42)
    save: bool = Field(default=True)
    export: bool = Field(default=False)

class PredictRequest(BaseModel):
    disease: Optional[str] = Field(default=None)
    symptoms: Optional[str] = Field(default=None)
    age: float = Field(..., ge=0, le=120)
    severity: float = Field(..., ge=1, le=5)
    duration_days: int = Field(..., ge=1, le=365)
    hospitalizations_last_year: int = Field(..., ge=0, le=5)
    functional_score: int = Field(..., ge=0, le=100)
    bmi: float = Field(..., ge=10, le=60)

@app.get('/health')
async def health():
    return {"status": "ok", "models_ready": models_exist()}

@app.post('/train')
async def train(req: TrainRequest):
    import subprocess, sys
    cmd = [sys.executable, '-m', 'ml.train', '--samples', str(req.samples), '--seed', str(req.seed)]
    if req.save:
        cmd.append('--save')
    if req.export:
        cmd.append('--export')
    try:
        subprocess.check_call(cmd)
    except subprocess.CalledProcessError as e:
        raise HTTPException(status_code=500, detail=f'Training failed: {e}')
    return {"status": "trained", "models_ready": models_exist()}


# --------------------------
# Patient Data Fetching
# --------------------------
def get_patient_data(patient_id: int) -> dict:
    base_url = "http://localhost:8080/patients"

    # Base patient info
    resp = requests.get(f"{base_url}/{patient_id}")
    if resp.status_code != 200:
        raise HTTPException(status_code=404, detail=f"Patient {patient_id} not found")
    patient = resp.json()

    # Symptoms
    resp_symptoms = requests.get(f"{base_url}/{patient_id}/symptoms")
    patient['symptoms'] = resp_symptoms.json() if resp_symptoms.status_code == 200 else []

    # Severity
    resp_severity = requests.get(f"{base_url}/{patient_id}/severity")
    patient['severity'] = resp_severity.json() if resp_severity.status_code == 200 else 1

    # Duration days
    resp_durationdays = requests.get(f"{base_url}/{patient_id}/duration-days")
    patient['duration_days'] = resp_durationdays.json() if resp_durationdays.status_code == 200 else 7

    # Hospitalizations last year
    resp_hospitalizations = requests.get(f"{base_url}/{patient_id}/hospitalizations")
    patient['hospitalizations_last_year'] = resp_hospitalizations.json() if resp_hospitalizations.status_code == 200 else 0

    # Functional score
    resp_functional = requests.get(f"{base_url}/{patient_id}/functional-score")
    patient['functional_score'] = resp_functional.json() if resp_functional.status_code == 200 else 50

    # BMI
    resp_bmi = requests.get(f"{base_url}/{patient_id}/bmi")
    patient['bmi'] = resp_bmi.json() if resp_bmi.status_code == 200 else 25.0

    # Symptoms description
    resp_symptoms_desc = requests.get(f"{base_url}/{patient_id}/symptoms-description")
    patient['symptoms_description'] = resp_symptoms_desc.json() if resp_symptoms_desc.status_code == 200 else ""

    return patient


# --------------------------
# Predict Endpoint
# --------------------------
@app.post("/predict-from-patient")
async def predict_from_patient(patient_id: int = Query(..., description="ID du patient")):
    if not models_exist():
        raise HTTPException(status_code=400, detail="Models are not trained yet.")

    patient_data = get_patient_data(patient_id)
    predict_payload = PredictRequest(
        disease=patient_data.get('maladie'),
        symptoms=", ".join(patient_data.get('symptoms', [])),
        age=patient_data.get('age'),
        severity=patient_data.get('severity'),
        duration_days=patient_data.get('duration_days'),
        hospitalizations_last_year=patient_data.get('hospitalizations_last_year'),
        functional_score=patient_data.get('functional_score'),
        bmi=patient_data.get('bmi')
    )

    models = load_models()
    result = predict(models, predict_payload.dict())

    return {
        "predicted_disease": result.get("predicted_disease", "N/A"),
        "recommended_session_count": result.get("recommended_session_count", "N/A"),
        "confidence": result.get("confidence", None),
        "explanation": result.get("explanation", "No explanation available."),
        "raw_output": result
    }


# --------------------------
# CLI Script
# --------------------------
def ensure_models():
    if models_exist():
        return
    from ml import train as tr
    df = tr.generate_synthetic(600, seed=2025)
    clf, reg, enc, scaler, vec = tr.train_models(df)
    tr.save_artifacts(clf, reg, enc, scaler, vec)

def main():
    parser = argparse.ArgumentParser(description="Prédire le nombre de séances thérapeutiques")
    parser.add_argument('--disease', type=str, default=None, help='Maladie connue (optionnel)')
    parser.add_argument('--symptoms', type=str, default='', help='Texte libre des symptômes')
    parser.add_argument('--age', type=float, required=True, help='Âge (0-120)')
    parser.add_argument('--severity', type=float, required=True, help='Sévérité (1-5)')
    parser.add_argument('--duration_days', type=int, required=True, help='Durée des symptômes en jours (1-365)')
    parser.add_argument('--hospitalizations_last_year', type=int, required=True, help='Hospitalisations sur 12 mois (0-5)')
    parser.add_argument('--functional_score', type=int, required=True, help='Score fonctionnel (0-100)')
    parser.add_argument('--bmi', type=float, required=True, help='IMC (10-60)')
    args = parser.parse_args()

    ensure_models()
    models = load_models()
    payload = {
        'disease': args.disease,
        'symptoms': args.symptoms,
        'age': args.age,
        'severity': args.severity,
        'duration_days': args.duration_days,
        'hospitalizations_last_year': args.hospitalizations_last_year,
        'functional_score': args.functional_score,
        'bmi': args.bmi,
    }
    res = predict(models, payload)
    print('Maladie prédite:', res['predicted_disease'])
    print('Nombre de séances prédit:', res['recommended_session_count'])

if __name__ == "__main__":
    main()
