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
python -m src.predict --sample "Mes=5;Hora=02:30;Distancia Kilometros=150;Tipo de Vehiculo=Moto;Clima=Lluvioso;Velocidad Promedio=120;Edad Conductor=22;Experiencia Conductor=2;Alcohol en Sangre=1;Visibilidad=Mala;Estado de la Via=Mojada;Iluminacion=Noche;Cinturon=No;Tipo de Carretera=Rural;Dia de la Semana=3" --models_dir models
```

Files added
- `src/train.py` — training and evaluation script
- `src/predict.py` — load saved models and predict a sample
- `src/utils.py` — helpers to load/prepare data
- `data/make_csv_from_excel.py` — helper to convert xlsx to csv
