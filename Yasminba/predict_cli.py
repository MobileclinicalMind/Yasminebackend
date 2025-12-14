import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))


def ensure_models():
    from ml.model import models_exist
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

    from ml.model import load_models, predict
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


if __name__ == '__main__':
    main()
