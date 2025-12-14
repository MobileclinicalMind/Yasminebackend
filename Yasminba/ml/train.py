import os
import argparse
import random
from typing import Tuple

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestRegressor
import joblib

BASE_DIR = os.path.dirname(os.path.dirname(__file__))
DATA_DIR = os.path.join(BASE_DIR, 'data')
CLASSIFIER_PATH = os.path.join(DATA_DIR, 'disease_classifier.joblib')
REGRESSOR_PATH = os.path.join(DATA_DIR, 'session_regressor.joblib')
ENCODER_PATH = os.path.join(DATA_DIR, 'label_encoder.joblib')
SCALER_PATH = os.path.join(DATA_DIR, 'feature_scaler.joblib')
VECTORIZER_PATH = os.path.join(DATA_DIR, 'symptoms_vectorizer.joblib')
DATASET_PATH = os.path.join(DATA_DIR, 'synthetic_dataset.csv')

DISEASES = [
    'diabetes', 'hypertension', 'asthma', 'depression', 'arthritis',
    'heart_failure', 'copd', 'migraine', 'eczema', 'covid19'
]


SYMPTOMS_BY_DISEASE = {
    'diabetes': ["fatigue", "soif excessive", "polyurie", "vision trouble"],
    'hypertension': ["maux de tête", "vertiges", "acouphènes", "fatigue"],
    'asthma': ["essoufflement", "toux", "sifflements", "oppression thoracique"],
    'depression': ["tristesse", "insomnie", "anxiété", "perte d'intérêt"],
    'arthritis': ["douleur articulaire", "raideur", "gonflement", "limitation"],
    'heart_failure': ["dyspnée", "œdèmes", "fatigue", "prise de poids"],
    'copd': ["toux chronique", "expectoration", "dyspnée", "infections"],
    'migraine': ["céphalées", "photophobie", "nausées", "aura"],
    'eczema': ["prurit", "rougeur", "sécheresse", "lésions"],
    'covid19': ["fièvre", "toux sèche", "anosmie", "myalgies"],
}


def generate_synthetic(n: int, seed: int = 42) -> pd.DataFrame:
    random.seed(seed)
    np.random.seed(seed)
    rows = []
    for _ in range(n):
        disease = random.choice(DISEASES)
        age = int(np.clip(np.random.normal(50, 18), 1, 100))
        severity = int(np.clip(np.random.normal(3, 1.2), 1, 5))
        # Global, objective features
        duration_days = int(np.clip(np.random.exponential(scale=30), 1, 365))
        hospitalizations = int(np.clip(np.random.poisson(0.5), 0, 5))
        functional_score = int(np.clip(np.random.normal(60, 20), 0, 100))
        bmi = float(np.clip(np.random.normal(27, 5), 18, 40))

        # Symptoms text synthesized per disease
        symptoms_list = SYMPTOMS_BY_DISEASE[disease]
        k = np.random.randint(2, len(symptoms_list)+1)
        chosen = np.random.choice(symptoms_list, size=k, replace=False)
        symptoms = ", ".join(chosen)

        # Base sessions per disease
        base = {
            'diabetes': 10,
            'hypertension': 8,
            'asthma': 7,
            'depression': 12,
            'arthritis': 9,
            'heart_failure': 14,
            'copd': 11,
            'migraine': 6,
            'eczema': 5,
            'covid19': 8,
        }[disease]

        # Modulate by severity, age and comorbidities
        sessions = base + 1.6 * (severity - 3) + 0.06 * (age - 50)
        sessions += 0.3 * (duration_days/30.0 - 1) + 0.5 * (hospitalizations)
        sessions += 0.5 * ((100 - functional_score)/20.0) + 0.2 * (bmi - 25)
        sessions += np.random.normal(0, 1.8)  # noise
        sessions = max(0, round(sessions))

        rows.append({
            'disease': disease,
            'age': age,
            'severity': severity,
            'symptoms': symptoms,
            'duration_days': duration_days,
            'hospitalizations_last_year': hospitalizations,
            'functional_score': functional_score,
            'bmi': round(bmi, 1),
            'session_count': sessions
        })
    return pd.DataFrame(rows)


def train_models(df: pd.DataFrame) -> Tuple[LogisticRegression, RandomForestRegressor, LabelEncoder, StandardScaler, TfidfVectorizer]:
    enc = LabelEncoder()
    y_cls = enc.fit_transform(df['disease'])

    # Numeric global features
    num_cols = ['age', 'severity', 'duration_days', 'hospitalizations_last_year', 'functional_score', 'bmi']
    X_num = df[num_cols].values.astype(float)
    scaler = StandardScaler()
    X_num_scaled = scaler.fit_transform(X_num)

    # Symptoms text vectorization
    vectorizer = TfidfVectorizer(max_features=300, ngram_range=(1,2))
    X_sym = vectorizer.fit_transform(df['symptoms']).toarray()

    # Concatenate for classifier
    X_cls = np.hstack([X_num_scaled, X_sym])

    # Classifier: predict disease from numeric + text features
    clf = LogisticRegression(max_iter=1000)
    X_train, X_test, y_train, y_test = train_test_split(X_cls, y_cls, test_size=0.2, random_state=42)
    clf.fit(X_train, y_train)

    # Regressor: predict session_count from numeric features + disease class index
    class_index = y_cls.reshape(-1, 1)
    X_reg = np.hstack([X_num, class_index])
    y_reg = df['session_count'].values.astype(float)
    Xr_train, Xr_test, yr_train, yr_test = train_test_split(X_reg, y_reg, test_size=0.2, random_state=42)
    reg = RandomForestRegressor(n_estimators=200, random_state=42)
    reg.fit(Xr_train, yr_train)

    return clf, reg, enc, scaler, vectorizer


def save_artifacts(clf, reg, enc, scaler, vectorizer):
    os.makedirs(DATA_DIR, exist_ok=True)
    joblib.dump(clf, CLASSIFIER_PATH)
    joblib.dump(reg, REGRESSOR_PATH)
    joblib.dump(enc, ENCODER_PATH)
    joblib.dump(scaler, SCALER_PATH)
    joblib.dump(vectorizer, VECTORIZER_PATH)


def main():
    parser = argparse.ArgumentParser(description='Train synthetic therapeutic session models')
    parser.add_argument('--samples', type=int, default=1500)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--save', action='store_true', help='Save artifacts to data/')
    parser.add_argument('--export', action='store_true', help='Save synthetic dataset to CSV')
    args = parser.parse_args()

    df = generate_synthetic(args.samples, args.seed)
    clf, reg, enc, scaler, vectorizer = train_models(df)

    if args.save:
        save_artifacts(clf, reg, enc, scaler, vectorizer)
    if args.export:
        os.makedirs(DATA_DIR, exist_ok=True)
        df.to_csv(DATASET_PATH, index=False)

    print('Training complete.')
    if args.save:
        print(f'Artifacts saved to {DATA_DIR}')
    if args.export:
        print(f'Dataset saved to {DATASET_PATH}')


if __name__ == '__main__':
    main()
