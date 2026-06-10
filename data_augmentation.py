"""
data_augmentation.py
--------------------
Sección 3.4 del TP3: Adquisición de datos y data augmentation.

Compara dos estrategias para mejorar el rendimiento en la cohorte problemática
(client_type = internal_research):

  Estrategia A — Datos reales:
    Incorporar el dataset extra de la cohorte al entrenamiento.

  Estrategia B — SMOTE:
    Generar datos sintéticos de la misma cohorte usando SMOTE,
    en lugar de incorporar el dataset extra.

En ambos casos se re-entrena el baseline (Random Forest con los mismos
hiperparámetros) y se evalúa con k-fold CV global y métricas específicas
de la cohorte internal_research.

Outputs:
  results/augmentation_comparison.csv   — métricas globales por estrategia
  results/augmentation_cohort.csv       — métricas de la cohorte por estrategia
"""

import warnings
warnings.filterwarnings("ignore")

import joblib
import numpy as np
import pandas as pd
import yaml
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score
from sklearn.model_selection import StratifiedKFold

# ── helpers ───────────────────────────────────────────────────────────────────

COLS_TO_DROP = [
    "total_gpu_hours",
    "total_processes",
    "request_week",
    "queue_wait_time",
]

COHORT_COL   = "client_type"
COHORT_VALUE = "internal_research"
EXTRA_PATH   = "data/raw/extra_client_type_internal_research.csv"


def load_config(path: str = "config.yaml") -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def calculate_metrics(y_true, y_pred, prefix: str = "") -> dict:
    return {
        f"{prefix}accuracy":  round(accuracy_score(y_true, y_pred), 4),
        f"{prefix}precision": round(precision_score(y_true, y_pred, zero_division=0), 4),
        f"{prefix}recall":    round(recall_score(y_true, y_pred, zero_division=0), 4),
        f"{prefix}f1":        round(f1_score(y_true, y_pred, zero_division=0), 4),
        f"{prefix}fp":        int(((y_pred == 1) & (y_true == 0)).sum()),
        f"{prefix}fn":        int(((y_pred == 0) & (y_true == 1)).sum()),
        f"{prefix}tp":        int(((y_pred == 1) & (y_true == 1)).sum()),
        f"{prefix}n":         int(len(y_true)),
    }


def oof_predict(model_params: dict, X: np.ndarray, y: np.ndarray,
                n_splits: int, random_state: int) -> np.ndarray:
    """Genera predicciones out-of-fold con k-fold estratificado."""
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    y_pred = np.zeros(len(y), dtype=int)
    for train_idx, val_idx in skf.split(X, y):
        clf = RandomForestClassifier(**model_params)
        clf.fit(X[train_idx], y[train_idx])
        y_pred[val_idx] = clf.predict(X[val_idx])
    return y_pred


def cohort_mask(raw_df: pd.DataFrame) -> np.ndarray:
    """Devuelve máscara booleana de filas pertenecientes a la cohorte problemática."""
    return (raw_df[COHORT_COL] == COHORT_VALUE).to_numpy()


# ── carga de datos ────────────────────────────────────────────────────────────

def load_data(config: dict):
    """
    Carga train transformado (para el modelo) y train raw (para máscaras de cohorte).
    También carga el dataset extra.
    """
    target = config["project"]["target"]

    raw_train = pd.read_csv(config["data"]["train_path"])
    X_train_transformed = pd.read_csv(config["data"]["train_transformed_path"])

    if target in X_train_transformed.columns:
        y_train = X_train_transformed[target].to_numpy()
        X_train = X_train_transformed.drop(columns=[target]).to_numpy()
    else:
        y_train = raw_train[target].to_numpy()
        X_train = X_train_transformed.to_numpy()

    # Dataset extra de la cohorte
    extra_raw = pd.read_csv(EXTRA_PATH)
    extra_raw = extra_raw.drop(columns=[c for c in COLS_TO_DROP if c in extra_raw.columns])

    return raw_train, X_train, y_train, extra_raw, target


def preprocess_extra(extra_raw: pd.DataFrame, target: str, config: dict):
    """
    Aplica el mismo preprocesador fitted sobre train al dataset extra.
    Devuelve X_extra, y_extra.
    """
    preprocessor = joblib.load(config["preprocessing"]["transformer_path"])
    y_extra = extra_raw[target].to_numpy()
    X_extra_raw = extra_raw.drop(columns=[target])
    X_extra = preprocessor.transform(X_extra_raw)
    return X_extra, y_extra


# ── estrategias ───────────────────────────────────────────────────────────────

def strategy_baseline(X_train, y_train, raw_train, model_params, config):
    """Baseline original sin augmentation."""
    n_splits    = config["validation"]["n_splits"]
    random_state = config["project"]["random_state"]

    y_pred = oof_predict(model_params, X_train, y_train, n_splits, random_state)

    mask = cohort_mask(raw_train)
    global_m = calculate_metrics(y_train, y_pred, "global_")
    cohort_m = calculate_metrics(y_train[mask], y_pred[mask], "cohort_")
    return {**global_m, **cohort_m}


def strategy_real_data(X_train, y_train, X_extra, y_extra, raw_train,
                       model_params, config):
    """
    Estrategia A: agrega datos reales de la cohorte al train.
    Se evalúa con OOF sobre el conjunto ampliado.
    Las métricas de cohorte se calculan solo sobre los samples
    del extra (son los nuevos datos de esa cohorte).
    """
    n_splits     = config["validation"]["n_splits"]
    random_state = config["project"]["random_state"]

    X_aug = np.vstack([X_train, X_extra])
    y_aug = np.concatenate([y_train, y_extra])

    y_pred = oof_predict(model_params, X_aug, y_aug, n_splits, random_state)

    # Métricas globales sobre todo el conjunto ampliado
    global_m = calculate_metrics(y_aug, y_pred, "global_")

    # Métricas de cohorte sobre los nuevos samples (índices del extra)
    cohort_idx = np.arange(len(y_train), len(y_aug))
    cohort_m = calculate_metrics(y_aug[cohort_idx], y_pred[cohort_idx], "cohort_")

    return {**global_m, **cohort_m}


def strategy_smote(X_train, y_train, raw_train, model_params, config):
    """
    Estrategia B: aplica SMOTE sobre el train para oversamplear la clase
    minoritaria de la cohorte interna.
    Se aplica SMOTE al train completo (no solo a la cohorte) para no
    distorsionar la distribución general.
    Las métricas de cohorte se evalúan sobre los samples originales de
    la cohorte en el train (misma máscara que baseline).
    """
    n_splits     = config["validation"]["n_splits"]
    random_state = config["project"]["random_state"]

    smote = SMOTE(random_state=random_state)
    X_aug, y_aug = smote.fit_resample(X_train, y_train)

    print(f"  SMOTE: {len(y_train)} -> {len(y_aug)} samples "
          f"(+{len(y_aug) - len(y_train)} sintéticos)")
    print(f"  Distribución post-SMOTE: "
          f"0={int((y_aug==0).sum())}  1={int((y_aug==1).sum())}")

    y_pred = oof_predict(model_params, X_aug, y_aug, n_splits, random_state)

    # Métricas globales sobre el conjunto resampleado
    global_m = calculate_metrics(y_aug, y_pred, "global_")

    # Métricas de cohorte: solo sobre los samples originales de la cohorte.
    # SMOTE agrega sintéticos al final, así que los primeros len(y_train)
    # elementos de y_pred corresponden a los originales.
    mask = cohort_mask(raw_train)
    y_pred_original = y_pred[:len(y_train)]
    cohort_m = calculate_metrics(y_train[mask], y_pred_original[mask], "cohort_")

    return {**global_m, **cohort_m}


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    config = load_config()
    target = config["project"]["target"]
    random_state = config["project"]["random_state"]

    # Cargar el modelo baseline para reutilizar sus hiperparámetros
    baseline_model = joblib.load(config["baseline"]["model_path"])
    model_params = {
        **baseline_model.get_params(),
        "random_state": random_state,
    }

    print("=== Cargando datos ===")
    raw_train, X_train, y_train, extra_raw, target = load_data(config)
    print(f"  Train original: {X_train.shape}  |  positivos: {y_train.sum()}")

    print("\n=== Preprocesando dataset extra ===")
    X_extra, y_extra = preprocess_extra(extra_raw, target, config)
    print(f"  Extra: {X_extra.shape}  |  positivos: {y_extra.sum()}")

    results = []

    # ── Baseline ──────────────────────────────────────────────────────────────
    print("\n=== [Baseline] Sin augmentation ===")
    m = strategy_baseline(X_train, y_train, raw_train, model_params, config)
    results.append({"estrategia": "baseline", **m})
    _print_metrics(m)

    # ── Estrategia A: datos reales ─────────────────────────────────────────
    print("\n=== [A] Datos reales (extra_client_type_internal_research) ===")
    m = strategy_real_data(X_train, y_train, X_extra, y_extra,
                           raw_train, model_params, config)
    results.append({"estrategia": "real_data", **m})
    _print_metrics(m)

    # ── Estrategia B: SMOTE ───────────────────────────────────────────────
    print("\n=== [B] SMOTE (datos sintéticos) ===")
    m = strategy_smote(X_train, y_train, raw_train, model_params, config)
    results.append({"estrategia": "smote", **m})
    _print_metrics(m)

    # ── Guardar resultados ────────────────────────────────────────────────
    df_results = pd.DataFrame(results)
    out_path = "results/augmentation_comparison.csv"
    df_results.to_csv(out_path, index=False)
    print(f"\nResultados guardados en: {out_path}")
    print(df_results.to_string(index=False))


def _print_metrics(m: dict):
    print(f"  Global  — Accuracy: {m['global_accuracy']:.4f}  "
          f"Precision: {m['global_precision']:.4f}  "
          f"Recall: {m['global_recall']:.4f}  "
          f"F1: {m['global_f1']:.4f}")
    print(f"  Cohorte — Precision: {m['cohort_precision']:.4f}  "
          f"Recall: {m['cohort_recall']:.4f}  "
          f"F1: {m['cohort_f1']:.4f}  "
          f"(n={m['cohort_n']}  FN={m['cohort_fn']}  TP={m['cohort_tp']})")


if __name__ == "__main__":
    main()