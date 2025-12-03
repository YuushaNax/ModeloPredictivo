import joblib
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional
import uvicorn

from src.utils import load_and_prepare

app = FastAPI(title='ModeloPredictivo API')


class Sample(BaseModel):
    Mes: Optional[int]
    Hora: Optional[str]
    Distancia_Kilometros: Optional[float]
    Tipo_de_Vehiculo: Optional[str]
    Clima: Optional[str]
    # additional optional fields allowed


@app.on_event('startup')
def load_models():
    global prob_pipe, cond_pipe, acc_pipe
    try:
        prob_pipe = joblib.load('models/prob_pipeline.joblib')
    except Exception:
        prob_pipe = None
    try:
        cond_pipe = joblib.load('models/cond_pipeline.joblib')
    except Exception:
        cond_pipe = None
    try:
        acc_pipe = joblib.load('models/acc_pipeline.joblib')
    except Exception:
        acc_pipe = None


@app.get('/health')
def health():
    return {'status':'ok'}


@app.post('/predict')
def predict(sample: dict):
    # Accept a dict of fields similar to sample string earlier
    try:
        df = load_and_prepare(sample)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

    features = ['Mes', 'Hora_hour', 'Hora_min', 'Distancia Kilometros', 'Tipo de Vehiculo', 'Clima']
    if prob_pipe is None or cond_pipe is None or acc_pipe is None:
        raise HTTPException(status_code=503, detail='Modelos no disponibles. Ejecute el entrenamiento primero.')

    X = df[features]
    p = prob_pipe.predict(X)[0] if prob_pipe is not None else None
    c = cond_pipe.predict(X)[0] if cond_pipe is not None else None
    a = int(acc_pipe.predict(X)[0]) if acc_pipe is not None else None

    # Try probability for accident if calibrated pipeline exposes predict_proba
    prob_acc = None
    try:
        prob_acc = float(acc_pipe.predict_proba(X)[0][1])
    except Exception:
        prob_acc = None

    return {
        'Probabilidad de accidente': f'{p:.1f}%',
        'Condicion del Vehiculo': f'{c:.1f}%',
        'Accidente': a,
        'Prob_accidente_model': prob_acc
    }


if __name__ == '__main__':
    uvicorn.run(app, host='0.0.0.0', port=8000)
