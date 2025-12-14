import os
import subprocess
import sys

BASE_DIR = os.path.dirname(os.path.dirname(__file__))


def test_training_and_prediction_flow():
    # Train
    cmd = [sys.executable, '-m', 'ml.train', '--samples', '300', '--seed', '123', '--save']
    subprocess.check_call(cmd, cwd=BASE_DIR)

    # Verify artifacts
    data_dir = os.path.join(BASE_DIR, 'data')
    assert os.path.exists(os.path.join(data_dir, 'disease_classifier.joblib'))
    assert os.path.exists(os.path.join(data_dir, 'session_regressor.joblib'))
    assert os.path.exists(os.path.join(data_dir, 'label_encoder.joblib'))
    assert os.path.exists(os.path.join(data_dir, 'feature_scaler.joblib'))
    assert os.path.exists(os.path.join(data_dir, 'symptoms_vectorizer.joblib'))

    # Simple predict via Python import
    from ml.model import load_models, predict
    models = load_models()
    payload = {
        "disease": "diabetes",
        "symptoms": "fatigue, essoufflement, douleur thoracique",
        "age": 55,
        "severity": 3,
        "duration_days": 30,
        "hospitalizations_last_year": 1,
        "functional_score": 40,
        "bmi": 28.5
    }
    res = predict(models, payload)
    assert 'recommended_session_count' in res and res['recommended_session_count'] >= 0
