import os
from typing import Tuple
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split
import joblib

BASE_DIR = os.path.dirname(os.path.dirname(__file__))
DATA_DIR = os.path.join(BASE_DIR, 'data')
CLASSIFIER_PATH = os.path.join(DATA_DIR, 'disease_classifier.joblib')
REGRESSOR_PATH = os.path.join(DATA_DIR, 'session_regressor.joblib')
ENCODER_PATH = os.path.join(DATA_DIR, 'label_encoder.joblib')
SCALER_PATH = os.path.join(DATA_DIR, 'feature_scaler.joblib')
VECTORIZER_PATH = os.path.join(DATA_DIR, 'symptoms_vectorizer.joblib')
DATASET_PATH = os.path.join(DATA_DIR, 'synthetic_dataset.csv')

def load_artifacts():
    clf = joblib.load(CLASSIFIER_PATH)
    reg = joblib.load(REGRESSOR_PATH)
    enc = joblib.load(ENCODER_PATH)
    scaler = joblib.load(SCALER_PATH)
    vec = joblib.load(VECTORIZER_PATH)
    return clf, reg, enc, scaler, vec

def evaluate_from_csv(test_size: float = 0.2, seed: int = 42):
    if not os.path.exists(DATASET_PATH):
        raise FileNotFoundError(f"Dataset not found at {DATASET_PATH}. Run training with --export.")
    df = pd.read_csv(DATASET_PATH)
    required_cols = {'disease','age','severity','symptoms','duration_days','hospitalizations_last_year','functional_score','bmi','session_count'}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Dataset missing columns: {missing}. Regenerate dataset.")

    clf, reg, enc, scaler, vec = load_artifacts()

    # Prepare classifier inputs
    num_cols = ['age', 'severity', 'duration_days', 'hospitalizations_last_year', 'functional_score', 'bmi']
    X_num = df[num_cols].values.astype(float)
    X_num_scaled = scaler.transform(X_num)
    X_sym = vec.transform(df['symptoms']).toarray()
    X_cls = np.hstack([X_num_scaled, X_sym])
    y_cls = enc.transform(df['disease'])

    Xc_train, Xc_test, yc_train, yc_test = train_test_split(X_cls, y_cls, test_size=test_size, random_state=seed)
    y_pred_cls = clf.predict(Xc_test)
    acc = accuracy_score(yc_test, y_pred_cls)
    cm = confusion_matrix(yc_test, y_pred_cls)

    # Prepare regressor inputs
    class_index_full = y_cls.reshape(-1, 1)
    X_reg_full = np.hstack([X_num, class_index_full])
    y_reg_full = df['session_count'].values.astype(float)
    Xr_train, Xr_test, yr_train, yr_test = train_test_split(X_reg_full, y_reg_full, test_size=test_size, random_state=seed)
    y_pred_reg = reg.predict(Xr_test)
    mae = mean_absolute_error(yr_test, y_pred_reg)
    rmse = mean_squared_error(yr_test, y_pred_reg, squared=False)

    return {
        'classification_accuracy': acc,
        'confusion_matrix': cm.tolist(),
        'regression_mae': mae,
        'regression_rmse': rmse,
        'test_size': test_size,
        'samples': len(df)
    }

if __name__ == '__main__':
    results = evaluate_from_csv()
    print("Accuracy:", results['classification_accuracy'])
    print("MAE:", results['regression_mae'])
    print("RMSE:", results['regression_rmse'])

from .train import generate_synthetic, train_models, DATA_DIR, DATASET_PATH
from .model import load_models, models_exist

os.makedirs(DATA_DIR, exist_ok=True)


def evaluate_and_plot(samples: int = 1000, seed: int = 123):
    # Lazy imports to avoid hard dependency when only using evaluate_from_csv
    import matplotlib.pyplot as plt
    from sklearn.metrics import ConfusionMatrixDisplay
    import numpy as np

    # Generate fresh synthetic dataset for evaluation
    df = generate_synthetic(samples, seed)
    df.to_csv(DATASET_PATH, index=False)

    # Ensure models exist; if not, train and save
    if not models_exist():
        clf, reg, enc, scaler, vectorizer = train_models(df)
        from .train import save_artifacts
        save_artifacts(clf, reg, enc, scaler, vectorizer)

    clf, reg, enc, scaler, vec = load_models()

    # Classification evaluation
    num_cols = ['age', 'severity', 'duration_days', 'hospitalizations_last_year', 'functional_score', 'bmi']
    X_num = df[num_cols].values.astype(float)
    X_num_scaled = scaler.transform(X_num)
    X_sym = vec.transform(df['symptoms']).toarray()
    X_cls = np.hstack([X_num_scaled, X_sym])
    y_true_cls = enc.transform(df['disease'])
    y_pred_cls = clf.predict(X_cls)

    cm = confusion_matrix(y_true_cls, y_pred_cls)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=enc.classes_)
    fig_cm, ax_cm = plt.subplots(figsize=(8, 6))
    disp.plot(ax=ax_cm, xticks_rotation=45, cmap='Blues', colorbar=False)
    fig_cm.tight_layout()
    cm_path = os.path.join(DATA_DIR, 'confusion_matrix.png')
    fig_cm.savefig(cm_path)
    plt.close(fig_cm)

    # Regression evaluation
    class_index = y_true_cls.reshape(-1, 1)
    X_reg = np.hstack([X_num, class_index])
    y_true_reg = df['session_count'].values.astype(float)
    y_pred_reg = reg.predict(X_reg)
    mae = mean_absolute_error(y_true_reg, y_pred_reg)

    fig_err, ax_err = plt.subplots(figsize=(8, 6))
    ax_err.hist(y_true_reg - y_pred_reg, bins=30, color='#2a9d8f', edgecolor='white')
    ax_err.set_title(f'Erreur (vrai - prédit) des séances | MAE={mae:.2f}')
    ax_err.set_xlabel('Erreur (séances)')
    ax_err.set_ylabel('Fréquence')
    fig_err.tight_layout()
    err_path = os.path.join(DATA_DIR, 'session_error_hist.png')
    fig_err.savefig(err_path)
    plt.close(fig_err)

    return {
        'confusion_matrix_path': cm_path,
        'error_histogram_path': err_path,
        'mae': mae,
        'dataset_path': DATASET_PATH
    }


if __name__ == '__main__':
    res = evaluate_and_plot()
    print('Evaluation complete:')
    for k, v in res.items():
        print(f'- {k}: {v}')
