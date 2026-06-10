# =============================================================================
# TP3 Machine Learning — Setup y ejecucion (Windows - PowerShell)
# =============================================================================
# Ejecutar desde PowerShell con:
#   Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
#   .\setup_windows.ps1
# =============================================================================

$ErrorActionPreference = "Stop"

Write-Host "============================================" -ForegroundColor Cyan
Write-Host " TP3 - Data-Centered Design" -ForegroundColor Cyan
Write-Host " Setup Windows (PowerShell)" -ForegroundColor Cyan
Write-Host "============================================" -ForegroundColor Cyan

# 1. Crear entorno virtual
Write-Host ""
Write-Host "[1/6] Creando entorno virtual..." -ForegroundColor Yellow
python -m venv .venv
& ".\.venv\Scripts\Activate.ps1"
Write-Host "      Entorno activado." -ForegroundColor Green

# 2. Instalar dependencias
Write-Host ""
Write-Host "[2/6] Instalando dependencias..." -ForegroundColor Yellow
python -m pip install --upgrade pip -q
python -m pip install -r requirements.txt -q
Write-Host "      Dependencias instaladas." -ForegroundColor Green

# 3. Preparar datos
Write-Host ""
Write-Host "[3/6] Preparando datos (train/test split)..." -ForegroundColor Yellow
python prepare_data.py

# 4. Limpiar features redundantes
Write-Host ""
Write-Host "[4/6] Limpieza de features redundantes..." -ForegroundColor Yellow
python feature_cleaning.py

# 5. Preprocesar
Write-Host ""
Write-Host "[5/6] Preprocesando datos (encoding, scaling)..." -ForegroundColor Yellow
python preprocess_data.py

# 6. Pipeline base
Write-Host ""
Write-Host "[6/6] Entrenando baseline y analizando cohortes..." -ForegroundColor Yellow
python train_baseline.py
python analyze_cohorts.py

Write-Host ""
Write-Host "============================================" -ForegroundColor Cyan
Write-Host " Pipeline base completado." -ForegroundColor Green
Write-Host ""
Write-Host " Proximos pasos (ejecutar manualmente):" -ForegroundColor Yellow
Write-Host "   python learning_curves.py"
Write-Host "   python data_augmentation.py"
Write-Host "   python train_final_model.py"
Write-Host "============================================" -ForegroundColor Cyan