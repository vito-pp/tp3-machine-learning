
# TP3 - Data-Centered Design

## Instalación de dependencias

```
python3 -m venv .venv
source .venv/bin/activate

python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## Objetivo

Predecir si una reserva de infraestructura GPU será cancelada antes de su ejecución.

La variable objetivo es:

* `canceled_job = 1`: la reserva fue cancelada.
* `canceled_job = 0`: la reserva no fue cancelada.

## Estado actual

Hasta ahora se completó:

1. Carga y validación del dataset original.
2. Separación fija en train y test.
3. Preprocesamiento de variables numéricas y categóricas.
4. Comparación de modelos baseline con validación cruzada.
5. Selección del mejor baseline.
6. Generación de predicciones out-of-fold sobre train.
7. Cálculo de métricas globales.
8. Análisis de métricas por cohorte.
9. Selección de la cohorte problemática a pedir.

## Baseline elegido

Se compararon:

* Regresión logística.
* Random Forest.

El mejor modelo según F1 promedio fue Random Forest.

```text
Accuracy  = 0.8267
Precision = 0.8120
Recall    = 0.6926
F1        = 0.7475
```

## Cohorte seleccionada

```text
client_type = internal_research
```

Resultados:

```text
n = 804
Precision = 0.727
Recall = 0.121
F1 = 0.208
Falsos negativos = 116
Verdaderos positivos = 16
```

El modelo detecta solamente alrededor del 12% de las cancelaciones reales de esta cohorte.
