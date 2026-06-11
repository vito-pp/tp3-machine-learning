# TP3 — Data-Centered Design

**Materia:** 72.75 Aprendizaje Automático — ITBA 2026 Q1  

**Objetivo:** 

Predecir si una reserva de infraestructura GPU será cancelada antes de su ejecución (`canceled_job`).

![Pipeline](docs/pipeline_flow.png)

---

## Cómo correr el proyecto

### Linux / macOS

```bash

bash setup_linux.sh
```

### Windows (PowerShell)

```powershell
# Solo la primera vez, para habilitar scripts:
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser

.\setup_windows.ps1
```

Ambos scripts crean el entorno virtual, instalan dependencias y corren el pipeline completo hasta `analyze_cohorts.py`.

---

## Estructura del proyecto

```
tp3-machine-learning/
├── data/
│   ├── raw/                          # Dataset original (no subir a git)
│   │   └── gpu_resource_reservations.csv
│   └── processed/                    # Generado por los scripts
│       ├── train.csv
│       ├── test.csv
│       ├── train_transformed.csv
│       └── test_transformed.csv
├── models/                           # Modelos serializados (.joblib)
├── results/                          # Métricas y predicciones
├── docs/                             # Imágenes para el README
│   └── pipeline_flow.png
├── src/                              # Módulos internos
├── config.yaml                       # Paths y parámetros centralizados
├── prepare_data.py
├── feature_cleaning.py
├── preprocess_data.py
├── train_baseline.py
├── analyze_cohorts.py
├── setup_linux.sh
├── setup_windows.ps1
├── requirements.txt
└── README.md
```

---

## Estado actual

### Completado

| # | Paso | Script | Output |
|---|------|--------|--------|
| 1 | Separación train/test (split fijo, seed=42) | `prepare_data.py` | `data/processed/train.csv`, `test.csv` |
| 2 | Limpieza de features redundantes | `feature_cleaning.py` | — |
| 3 | Encoding de categóricas + escalado | `preprocess_data.py` | `train_transformed.csv`, `test_transformed.csv` |
| 4 | Entrenamiento baseline con k-fold CV (k=5) | `train_baseline.py` | `models/baseline.joblib`, `results/baseline_comparison.csv` |
| 5 | OOF predictions + análisis de errores por cohorte | `analyze_cohorts.py` | `results/oof_predictions.csv`, `results/cohort_metrics.csv`, FP/FN |
| 6 | Selección de cohorte problemática | — | `client_type = internal_research` |
| 7 | Datos reales de cohorte nuevos + SMOTE | — | `data_augmentation.py` |

### Pendiente
- [ ] Modelo final + función predictora (`train_final_model.py`)

---

## Feature Cleaning

Se eliminaron 4 variables antes del preprocesamiento, reduciendo de 29 a 25 features:

| Variable eliminada | Motivo |
|--------------------|--------|
| `total_gpu_hours` | Derivada exacta de `peak_gpu_hours + offpeak_gpu_hours` (diff = 0.0) |
| `total_processes` | Derivada exacta de `num_primary + num_auxiliary + num_background` (diff = 0.0) |
| `request_week` | Correlación 0.9954 con `request_month`; se conserva `request_month` por ser más interpretable |
| `queue_wait_time` | Varianza cero: todos los valores son 0 en el dataset |

---

## Baseline elegido

Se compararon Regresión Logística y Random Forest con k-fold CV (k=5). Métricas elegidas: **F1** y **Recall** — el costo de un falso negativo (no detectar una cancelación) es mayor que el de un falso positivo en este contexto operativo.

**Ganador: Random Forest**

```
Accuracy  = 0.8267
Precision = 0.8120
Recall    = 0.6926
F1        = 0.7475
```

---

## Cohorte problemática seleccionada

```
client_type = internal_research
```

| Métrica | Valor |
|---------|-------|
| n (train) | 804 |
| Precision | 0.727 |
| Recall | **0.121** |
| F1 | 0.208 |
| Falsos negativos | 116 |
| Verdaderos positivos | 16 |

El modelo detecta solo el ~12% de las cancelaciones reales de esta cohorte. Con 116 falsos negativos sobre 132 cancelaciones reales, es la cohorte con peor recall del dataset y la mayor cantidad de errores en términos absolutos, lo que justifica priorizar su mejora en la estrategia data-centered.
