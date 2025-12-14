import sys
import os
from pathlib import Path

# Chemins
ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / 'data'


def ensure_models():
    from ml.model import models_exist
    if models_exist():
        return
    # Entraîner rapidement si les modèles n'existent pas
    from ml import train as tr
    df = tr.generate_synthetic(500, seed=2025)
    clf, reg, enc, scaler, vec = tr.train_models(df)
    tr.save_artifacts(clf, reg, enc, scaler, vec)


def run_prediction():
    from ml.model import load_models, predict
    models = load_models()
    payload = {
        'disease': 'hypertension',
        'symptoms': 'maux de tête, vertiges, fatigue',
        'age': 60,
        'severity': 3,
        'duration_days': 45,
        'hospitalizations_last_year': 0,
        'functional_score': 55,
        'bmi': 27.2,
    }
    res = predict(models, payload)
    print('— Prédiction —')
    print('Maladie prédite:', res['predicted_disease'])
    print('Nombre de séances prédit:', res['recommended_session_count'])
    return res


def run_evaluation_and_plots():
    from ml.evaluate import evaluate_and_plot
    res = evaluate_and_plot(samples=800, seed=2025)
    print('\n— Évaluation —')
    print('MAE (séances):', res['mae'])
    print('Confusion matrix PNG :', res['confusion_matrix_path'])
    print('Histogramme erreur PNG:', res['error_histogram_path'])

    # Ouvrir les images sous Windows
    try:
        if sys.platform.startswith('win'):
            if os.path.exists(res['confusion_matrix_path']):
                os.startfile(res['confusion_matrix_path'])  # type: ignore[attr-defined]
            if os.path.exists(res['error_histogram_path']):
                os.startfile(res['error_histogram_path'])  # type: ignore[attr-defined]
    except Exception as e:
        print("(Info) Impossible d'ouvrir automatiquement les images:", e)

    return res


def main():
    ensure_models()
    pred = run_prediction()
    _ = run_evaluation_and_plots()


if __name__ == '__main__':
    # S'assurer que le projet est importable si lancé ailleurs
    sys.path.insert(0, str(ROOT))
    main()
