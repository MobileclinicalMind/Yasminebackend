import os
from typing import Dict, Any
import joblib

BASE_DIR = os.path.dirname(os.path.dirname(__file__))
DATA_DIR = os.path.join(BASE_DIR, 'data')
CLASSIFIER_PATH = os.path.join(DATA_DIR, 'disease_classifier.joblib')
REGRESSOR_PATH = os.path.join(DATA_DIR, 'session_regressor.joblib')
ENCODER_PATH = os.path.join(DATA_DIR, 'label_encoder.joblib')
SCALER_PATH = os.path.join(DATA_DIR, 'feature_scaler.joblib')
VECTORIZER_PATH = os.path.join(DATA_DIR, 'symptoms_vectorizer.joblib')


def models_exist() -> bool:
    return all(os.path.exists(p) for p in [CLASSIFIER_PATH, REGRESSOR_PATH, ENCODER_PATH, SCALER_PATH, VECTORIZER_PATH])


def load_models():
    clf = joblib.load(CLASSIFIER_PATH)
    reg = joblib.load(REGRESSOR_PATH)
    enc = joblib.load(ENCODER_PATH)
    scl = joblib.load(SCALER_PATH)
    vec = joblib.load(VECTORIZER_PATH)
    return clf, reg, enc, scl, vec


def predict(models, payload: Dict[str, Any]) -> Dict[str, Any]:
    clf, reg, enc, scl, vec = models
    # Order features consistently
    disease = payload.get('disease')
    age = float(payload.get('age'))
    severity = float(payload.get('severity'))
    duration_days = float(payload.get('duration_days'))
    hospitalizations = float(payload.get('hospitalizations_last_year'))
    functional_score = float(payload.get('functional_score'))
    bmi = float(payload.get('bmi'))
    symptoms_text = str(payload.get('symptoms') or '')

    # Transform features
    import numpy as np
    # Disease one-hot via label encoder classes; build vector as probability prior
    # For prediction, we rely on classifier for class probs using numeric features
    num_cols = np.array([[age, severity, duration_days, hospitalizations, functional_score, bmi]])
    X_scaled = scl.transform(num_cols)
    X_sym = vec.transform([symptoms_text]).toarray()
    X_cls = np.hstack([X_scaled, X_sym])

    # Class probabilities and predicted class
    probs = clf.predict_proba(X_cls)[0]
    classes = enc.classes_.tolist()
    class_probs = {cls: float(p) for cls, p in zip(classes, probs)}

    # Use input disease if provided; otherwise top class
    predicted_disease = disease if disease in classes else classes[int(probs.argmax())]

    # Regressor estimates session counts conditioned on numeric features + class index
    class_index = classes.index(predicted_disease)
    X_reg = np.array([[age, severity, duration_days, hospitalizations, functional_score, bmi, class_index]])
    session_count = float(reg.predict(X_reg)[0])
    session_count = max(0.0, round(session_count))

    return {
        'predicted_disease': predicted_disease,
        'class_probabilities': class_probs,
        'recommended_session_count': int(session_count)
    }
