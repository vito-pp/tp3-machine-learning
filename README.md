# TP3 -- Data-Centered Design -- Grupo 8

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
├── data_augmentation.py
├── compare_strategies_cv.py
├── train_final_model.py
├── predictor_v1_grupo_8.py
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
| 7 | Comparación de estrategias: baseline vs real data vs SMOTE | `compare_strategies_cv.py` | `results/strategy_comparison_cv.csv`, `strategy_oof_predictions.csv` |
| 8 | Entrenamiento del modelo final con la estrategia seleccionada | `train_final_model.py` | `models/final_model.joblib`, `results/final_test_metrics.csv` |
| 9 | Función predictora final | `predictor_v1_grupo_8.py` | `predictor_v1_Grupo_X(data)` |

### Pendiente
- [ ] Redactar conclusiones finales en el informe.
- [ ] Verificar que el notebook o presentación usen las métricas finales actualizadas.

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

El modelo detecta solo el ~12% de las cancelaciones reales de esta cohorte. Con 116 falsos negativos sobre 132 cancelaciones reales, el error principal está en los falsos negativos: reservas que sí se cancelaron, pero el modelo no detectó. Por este motivo se priorizó esta cohorte dentro de la estrategia data-centered.

---

## Comparación de estrategias data-centered

Después de seleccionar la cohorte problemática, se compararon tres estrategias mediante validación cruzada estratificada sobre el conjunto de entrenamiento:

| Estrategia | Descripción |
|------------|-------------|
| `baseline` | Random Forest original, sin datos adicionales ni oversampling |
| `real_data` | Random Forest entrenado agregando los datos reales adicionales de `internal_research` |
| `smote` | Random Forest entrenado aplicando SMOTE sobre el sub-train de cada fold |

La comparación se realizó con `compare_strategies_cv.py`.

Importante: SMOTE se aplica dentro de cada fold de cross-validation, solo sobre el sub-train. La validación se realiza siempre sobre datos reales, no sintéticos. El conjunto de test no se usa para elegir la estrategia.

Resultados de CV:

| Estrategia | F1 global | Recall global | F1 cohorte | Recall cohorte |
|------------|-----------|---------------|------------|----------------|
| `smote` | **0.7478** | 0.7021 | 0.1316 | 0.0758 |
| `baseline` | 0.7464 | 0.6968 | 0.1447 | 0.0833 |
| `real_data` | 0.7456 | **0.7057** | **0.1806** | **0.1061** |

La estrategia seleccionada fue `smote`, porque obtuvo el mayor F1 global promedio en cross-validation. Sin embargo, la mejora global fue marginal y la estrategia `real_data` fue la que mejoró más específicamente la cohorte problemática `internal_research`.

Conclusión de esta etapa:

> SMOTE fue elegido como estrategia final por F1 global, pero los datos reales adicionales fueron más útiles para mejorar la cohorte problemática. Esto sugiere que la mejora global y la mejora localizada no necesariamente coinciden.

---

## Modelo final

Una vez seleccionada la estrategia ganadora por cross-validation, se entrenó un único modelo final usando:

```
Random Forest + SMOTE
```

Este modelo se entrenó con el conjunto de entrenamiento y se evaluó una sola vez sobre el conjunto de test.

Script:

```
train_final_model.py
```

Outputs:

| Archivo | Descripción |
|---------|-------------|
| `models/final_model.joblib` | Modelo final serializado |
| `models/final_model_metadata.json` | Metadata del modelo final |
| `results/final_test_metrics.csv` | Métricas finales sobre test |
| `results/final_test_cohort_metrics.csv` | Métricas finales sobre la cohorte en test |
| `results/final_test_predictions.csv` | Predicciones finales sobre test |

Métricas finales sobre test:

| Métrica | Valor |
|---------|-------|
| Accuracy | 0.8331 |
| Precision | 0.8215 |
| Recall | 0.7020 |
| F1 | 0.7570 |

Métricas finales en test para `client_type = internal_research`:

| Métrica | Valor |
|---------|-------|
| n | 205 |
| Accuracy | 0.8878 |
| Precision | 0.5000 |
| Recall | 0.0435 |
| F1 | 0.0800 |
| Falsos negativos | 22 |
| Verdaderos positivos | 1 |

El modelo final mejora levemente el rendimiento global respecto del baseline inicial, pero sigue teniendo dificultades para detectar cancelaciones dentro de `internal_research`. Esto queda como una limitación importante del enfoque.

---

## Predictor final

La función predictora final está definida en:

```
predictor_v1_grupo_8.py
```

Función principal:

```python
predictor_v1_Grupo_X(data)
```

La función acepta:

- un `DataFrame` de pandas;
- o una ruta a un archivo `.csv`.

Devuelve:

- `prediction`: clase predicha (`0` o `1`);
- `probability_canceled_job`: probabilidad estimada de cancelación.

Ejemplo de uso:

```python
from predictor_v1_grupo_8 import predictor_v1_Grupo_X

predicciones = predictor_v1_Grupo_8("data/processed/test.csv")
print(predicciones.head())
```

---

## Interpretación final

El trabajo siguió un enfoque data-centered:

1. Se entrenó un baseline.
2. Se analizaron errores mediante predicciones out-of-fold.
3. Se identificó una cohorte problemática.
4. Se pidieron datos reales adicionales de esa cohorte.
5. Se comparó el efecto de datos reales contra SMOTE.
6. Se seleccionó una estrategia final por cross-validation.
7. Se entrenó un modelo final y se evaluó una única vez sobre test.

La decisión final fue usar `SMOTE + Random Forest`, ya que obtuvo el mejor F1 global en validación cruzada. Aun así, el análisis por cohorte muestra que el problema específico de `internal_research` no quedó resuelto completamente. Esto sugiere que la cohorte puede requerir más datos reales, mejores variables explicativas o una estrategia de modelado específica para esa región del dataset.
