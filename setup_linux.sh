#!/bin/bash
# =============================================================================
# TP3 Machine Learning — Setup y ejecución (Linux / macOS)
# =============================================================================
set -e  # salir si cualquier comando falla

echo "============================================"
echo " TP3 — Data-Centered Design"
echo " Setup Linux / macOS"
echo "============================================"

# 1. Crear y activar entorno virtual
echo ""
echo "[1/6] Creando entorno virtual..."
python3 -m venv .venv
source .venv/bin/activate
echo "      Entorno activado: $(which python3)"

# 2. Actualizar pip e instalar dependencias
echo ""
echo "[2/6] Instalando dependencias..."
python3 -m pip install --upgrade pip -q
python3 -m pip install -r requirements.txt -q
echo "      Dependencias instaladas."

# 3. Preparar datos (train/test split)
echo ""
echo "[3/6] Preparando datos (train/test split)..."
python3 prepare_data.py

# 4. Limpiar features redundantes
echo ""
echo "[4/6] Limpieza de features redundantes..."
python3 feature_cleaning.py

# 5. Preprocesar datos
echo ""
echo "[5/6] Preprocesando datos (encoding, scaling)..."
python3 preprocess_data.py

# 6. Correr pipeline completo
echo ""
echo "[6/6] Entrenando baseline y analizando cohortes..."
python3 train_baseline.py
python3 analyze_cohorts.py

echo ""
echo "============================================"
echo " Pipeline base completado."
echo ""
echo " Proximos pasos (ejecutar manualmente):"
echo "   python3 learning_curves.py"
echo "   python3 data_augmentation.py"
echo "   python3 train_final_model.py"
echo "============================================"