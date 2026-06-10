"""
feature_cleaning.py
-------------------
Análisis de correlaciones y limpieza de variables redundantes.
Lee train.csv y test.csv desde los paths definidos en config.yaml,
elimina las features redundantes y sobreescribe los archivos.

Variables eliminadas y justificación:
  1. total_gpu_hours   -> derivada exacta: peak_gpu_hours + offpeak_gpu_hours (diff = 0)
  2. total_processes   -> derivada exacta: num_primary + num_auxiliary + num_background (diff = 0)
  3. request_week      -> correlación 0.9954 con request_month; aporta información redundante
  4. queue_wait_time   -> varianza cero (todos los valores son 0); no aporta señal

Estas variables introducen multicolinealidad y pueden inflar la importancia de features
en modelos como Random Forest sin agregar capacidad predictiva real.
"""

import numpy as np
import pandas as pd
import yaml

COLS_TO_DROP = [
    "total_gpu_hours",   # = peak_gpu_hours + offpeak_gpu_hours (exactamente)
    "total_processes",   # = num_primary + num_auxiliary + num_background (exactamente)
    "request_week",      # correlación 0.9954 con request_month
    "queue_wait_time",   # varianza cero en todo el dataset
]


def load_config(path: str = "config.yaml") -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def verify_redundancies(df: pd.DataFrame) -> None:
    print("=== Verificación de redundancias ===")
    match1 = np.allclose(df["total_gpu_hours"], df["peak_gpu_hours"] + df["offpeak_gpu_hours"])
    print(f"total_gpu_hours == peak + offpeak:  {match1}")
    match2 = np.allclose(
        df["total_processes"],
        df["num_primary_processes"] + df["num_auxiliary_processes"] + df["num_background_processes"]
    )
    print(f"total_processes == sum(parts):      {match2}")
    corr = df[["request_week", "request_month"]].corr().iloc[0, 1]
    print(f"corr(request_week, request_month):  {corr:.4f}")
    all_zero = (df["queue_wait_time"] == 0).all()
    print(f"queue_wait_time all zeros:          {all_zero}")
    print()


def drop_redundant_features(df: pd.DataFrame, verbose: bool = True) -> pd.DataFrame:
    cols_present = [c for c in COLS_TO_DROP if c in df.columns]
    cols_missing  = [c for c in COLS_TO_DROP if c not in df.columns]
    if verbose:
        print(f"  Columnas eliminadas:  {cols_present}")
        if cols_missing:
            print(f"  Ya no presentes:      {cols_missing}")
        print(f"  Shape antes:  {df.shape}")
    df_clean = df.drop(columns=cols_present)
    if verbose:
        print(f"  Shape después: {df_clean.shape}")
    return df_clean


def get_high_correlations(df: pd.DataFrame, threshold: float = 0.85) -> pd.DataFrame:
    num_df = df.select_dtypes(include="number").drop(columns=["canceled_job"], errors="ignore")
    corr = num_df.corr().abs()
    upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
    pairs = (
        upper.stack()
        .reset_index()
        .rename(columns={"level_0": "feature_a", "level_1": "feature_b", 0: "correlation"})
    )
    return pairs[pairs["correlation"] > threshold].sort_values("correlation", ascending=False)


def main() -> None:
    config = load_config()
    train_path = config["data"]["train_path"]
    test_path  = config["data"]["test_path"]

    print(f"Cargando train: {train_path}")
    train = pd.read_csv(train_path)
    verify_redundancies(train)

    print("Limpiando train...")
    train_clean = drop_redundant_features(train)
    train_clean.to_csv(train_path, index=False)
    print(f"  Guardado en: {train_path}\n")

    print(f"Limpiando test: {test_path}")
    test = pd.read_csv(test_path)
    test_clean = drop_redundant_features(test, verbose=False)
    test_clean.to_csv(test_path, index=False)
    print(f"  Shape: {test.shape} -> {test_clean.shape}")
    print(f"  Guardado en: {test_path}")

    print("\nPares con alta correlación (|r| > 0.85) en train limpio:")
    print(get_high_correlations(train_clean).to_string(index=False))


if __name__ == "__main__":
    main()