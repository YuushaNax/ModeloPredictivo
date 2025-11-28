# ModeloPredictivo — Starter pipeline

Minimal scaffolding to train models that predict:
- `Probabilidad de accidente` (regression, percent)
- `Accidente` (classification, Sí/No)
- `Condicion del Vehiculo` (regression, percent)

Quick usage

1. Create and activate virtualenv (PowerShell):

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

2. If you have an Excel file (like the generator output) convert to CSV or point scripts at the `.xlsx` directly.

3. Generate CSV (optional) or run training:

```powershell
# convert (optional)
python data\make_csv_from_excel.py --input data\raw\datos_entrenamiento_10000.xlsx --output data\raw\datos_entrenamiento_10000.csv

# train (with optional tuning / mlflow)
python -m src.train --data data\raw\datos_entrenamiento_10000.csv --out models

# quick train with Optuna tuning (fast example: 8 trials)
python -m src.train --data data\raw\datos_entrenamiento_10000.csv --out models

# generate EDA plots
python src\visualize.py --data data\raw\datos_entrenamiento_10000.csv --out reports

# run API server
uvicorn src.app:app --reload --port 8000
```

4. Predictions (example):

```powershell
python src\predict.py --sample "Mes=5;Hora=13:45;Distancia Kilometros=12;Tipo de Vehiculo=Moto;Clima=Lluvioso"
```

Files added
- `src/train.py` — training and evaluation script
- `src/predict.py` — load saved models and predict a sample
- `src/utils.py` — helpers to load/prepare data
- `data/make_csv_from_excel.py` — helper to convert xlsx to csv
