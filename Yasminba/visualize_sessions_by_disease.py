import os
import sys
from pathlib import Path
from typing import List, Dict

import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))


def ensure_models():
    from ml.model import models_exist
    if models_exist():
        return
    from ml import train as tr
    df = tr.generate_synthetic(800, seed=2025)
    clf, reg, enc, scaler, vec = tr.train_models(df)
    tr.save_artifacts(clf, reg, enc, scaler, vec)


def predict_sessions_by_disease() -> Dict[str, int]:
    from ml.model import load_models, predict
    from ml.train import DISEASES, SYMPTOMS_BY_DISEASE

    models = load_models()
    results: Dict[str, int] = {}

    # Profil patient type (fixe) pour comparer entre maladies
    defaults = {
        'age': 55,
        'severity': 3,
        'duration_days': 30,
        'hospitalizations_last_year': 0,
        'functional_score': 60,
        'bmi': 27.0,
    }

    for disease in DISEASES:
        # Composer un texte de symptômes plausible pour la maladie
        sym_list = SYMPTOMS_BY_DISEASE.get(disease, [])
        symptoms = ", ".join(sym_list[:3]) if sym_list else ""
        payload = {
            'disease': disease,
            'symptoms': symptoms,
            **defaults,
        }
        res = predict(models, payload)
        results[disease] = int(res['recommended_session_count'])

    return results


def plot_results(results: Dict[str, int]) -> str:
    data_dir = ROOT / 'data'
    data_dir.mkdir(parents=True, exist_ok=True)
    out_path = data_dir / 'sessions_by_disease.png'

    diseases = list(results.keys())
    values = [results[d] for d in diseases]

    plt.figure(figsize=(11, 5))
    bars = plt.bar(diseases, values, color="#2a9d8f")
    plt.title("Séances recommandées par maladie (profil patient type)")
    plt.ylabel("Nombre de séances (prédit)")
    plt.xticks(rotation=30, ha='right')
    # Annoter les barres
    for b, v in zip(bars, values):
        plt.text(b.get_x() + b.get_width()/2, b.get_height() + 0.2, str(v), ha='center', va='bottom', fontsize=9)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()
    return str(out_path)


def main():
    ensure_models()
    results = predict_sessions_by_disease()
    print("Résultats sessions par maladie:", results)
    path = plot_results(results)
    print("Graphique enregistré:", path)
    # Ouvrir automatiquement sur Windows
    try:
        if sys.platform.startswith('win') and os.path.exists(path):
            os.startfile(path)  # type: ignore[attr-defined]
    except Exception as e:
        print("(Info) Impossible d'ouvrir automatiquement l'image:", e)


if __name__ == '__main__':
    main()
