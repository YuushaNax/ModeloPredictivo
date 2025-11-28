import os
import argparse
import joblib
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import mean_squared_error, classification_report, f1_score

from src.utils import load_and_prepare

# Optional imports for LightGBM/Optuna/MLflow/SHAP
try:
    import lightgbm as lgb
except Exception:
    lgb = None
try:
    import optuna
except Exception:
    optuna = None
try:
    import mlflow
except Exception:
    mlflow = None
try:
    import shap
except Exception:
    shap = None


def build_preprocessor(num_cols, cat_cols):
    num_pipe = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])
    cat_pipe = Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('ohe', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])
    preprocessor = ColumnTransformer([
        ('num', num_pipe, num_cols),
        ('cat', cat_pipe, cat_cols)
    ])
    return preprocessor


def train_models(df, out_dir='models'):
    os.makedirs(out_dir, exist_ok=True)

    # Extend feature set with new columns if present
    features = ['Mes', 'Hora_hour', 'Hora_min', 'Distancia Kilometros', 'Tipo de Vehiculo', 'Clima']
    # optional numeric features added by generator
    for c in ['Velocidad Promedio','Edad Conductor','Experiencia Conductor','Dia de la Semana']:
        if c in df.columns and c not in features:
            features.append(c)
    # optional categorical features
    for c in ['Tipo de Carretera','Visibilidad','Estado de la Via','Iluminacion','Cinturon','Alcohol en Sangre']:
        if c in df.columns and c not in features:
            features.append(c)

    num_cols = [c for c in features if c in ['Mes','Hora_hour','Hora_min','Distancia Kilometros','Velocidad Promedio','Edad Conductor','Experiencia Conductor','Dia de la Semana']]
    cat_cols = [c for c in features if c not in num_cols]

    # Drop rows with missing target values
    df = df.copy()
    df = df.dropna(subset=['Probabilidad de accidente (%)','Condicion del Vehiculo (%)','Accidente_bin'])

    X = df[features]
    y_prob = df['Probabilidad de accidente (%)']
    y_cond = df['Condicion del Vehiculo (%)']
    y_acc = df['Accidente_bin']

    # Prepare stratify parameter: only use stratify if every class has at least 2 samples
    stratify_param = None
    try:
        counts = y_acc.value_counts()
        if len(counts) > 1 and counts.min() >= 2:
            stratify_param = y_acc
        else:
            print('Warning: skipping stratified split because one or more classes have fewer than 2 members')
    except Exception:
        stratify_param = None

    X_train, X_test, yp_train, yp_test, yc_train, yc_test, ya_train, ya_test = train_test_split(
        X, y_prob, y_cond, y_acc, test_size=0.2, random_state=42, stratify=stratify_param
    )

    preprocessor = build_preprocessor(num_cols, cat_cols)

    # Probabilidad (regression) - try LightGBM regressor if available
    if lgb is not None:
        prob_model = lgb.LGBMRegressor(n_estimators=200, random_state=42)
    else:
        from sklearn.ensemble import RandomForestRegressor
        prob_model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    prob_pipe = Pipeline([
        ('pre', preprocessor),
        ('model', prob_model)
    ])
    prob_pipe.fit(X_train, yp_train)
    p_pred = prob_pipe.predict(X_test)
    rmse = np.sqrt(mean_squared_error(yp_test, p_pred))
    print(f'Probabilidad RMSE: {rmse:.3f}')
    joblib.dump(prob_pipe, os.path.join(out_dir, 'prob_pipeline.joblib'))

    # Condicion del vehiculo (regression)
    if lgb is not None:
        cond_model = lgb.LGBMRegressor(n_estimators=200, random_state=43)
    else:
        from sklearn.ensemble import RandomForestRegressor
        cond_model = RandomForestRegressor(n_estimators=100, random_state=43, n_jobs=-1)
    cond_pipe = Pipeline([
        ('pre', preprocessor),
        ('model', cond_model)
    ])
    cond_pipe.fit(X_train, yc_train)
    c_pred = cond_pipe.predict(X_test)
    rmse_c = np.sqrt(mean_squared_error(yc_test, c_pred))
    print(f'Condicion RMSE: {rmse_c:.3f}')
    joblib.dump(cond_pipe, os.path.join(out_dir, 'cond_pipeline.joblib'))

    # Accidente (classification) - use LightGBM with optional Optuna tuning and MLflow
    if lgb is not None:
        best_params = None
        if optuna is not None and (hasattr(train_models, 'optuna_trials') and train_models.optuna_trials > 0):
            # optuna tuning on a pipeline that includes preprocessing
            def objective(trial):
                param = {
                    'n_estimators': trial.suggest_int('n_estimators', 50, 300),
                    'learning_rate': trial.suggest_loguniform('learning_rate', 0.01, 0.3),
                    'num_leaves': trial.suggest_int('num_leaves', 16, 128),
                    'min_child_samples': trial.suggest_int('min_child_samples', 5, 100)
                }
                model = lgb.LGBMClassifier(random_state=44, **param)
                pipe = Pipeline([('pre', preprocessor), ('model', model)])
                scores = cross_val_score(pipe, X_train, ya_train, cv=3, scoring='f1', n_jobs=-1)
                return scores.mean()

            study = optuna.create_study(direction='maximize')
            study.optimize(objective, n_trials=train_models.optuna_trials)
            best_params = study.best_params
            print('Optuna best params:', best_params)

        if best_params is None:
            acc_model = lgb.LGBMClassifier(n_estimators=200, random_state=44)
        else:
            acc_model = lgb.LGBMClassifier(random_state=44, **best_params)

        # Build a pipeline with preprocessor + model
        acc_pipe_uncl = Pipeline([('pre', preprocessor), ('model', acc_model)])

        # Fit uncalibrated pipeline (we'll use the underlying model for SHAP)
        acc_pipe_uncl.fit(X_train, ya_train)

        # Calibrate the whole pipeline
        acc_calibrated = CalibratedClassifierCV(acc_pipe_uncl, cv=3, method='sigmoid')
        acc_calibrated.fit(X_train, ya_train)
        a_pred = acc_calibrated.predict(X_test)
        print('Accidente classification report:')
        print(classification_report(ya_test, a_pred, digits=3))
        joblib.dump(acc_calibrated, os.path.join(out_dir, 'acc_pipeline.joblib'))

        # MLflow logging (optional)
        if mlflow is not None:
            try:
                mlflow.set_experiment('ModeloPredictivo')
                with mlflow.start_run():
                    mlflow.log_metric('acc_f1', f1_score(ya_test, a_pred))
                    mlflow.log_metric('prob_rmse', float(rmse))
                    mlflow.log_metric('cond_rmse', float(rmse_c))
                    if best_params is not None:
                        mlflow.log_params(best_params)
                    # save model artifact
                    mlflow.log_artifact(os.path.join(out_dir, 'acc_pipeline.joblib'))

            except Exception as e:
                print('MLflow logging failed:', e)

        # SHAP explainability (optional)
        if shap is not None:
            try:
                # Extract the trained LightGBM model from the pipeline for SHAP
                try:
                    lgb_model = acc_pipe_uncl.named_steps['model']
                    pre = acc_pipe_uncl.named_steps['pre']
                except Exception:
                    lgb_model = acc_model
                    pre = preprocessor

                sample = X_train.sample(min(500, len(X_train)), random_state=42)
                X_sample_trans = pre.transform(sample)
                explainer = shap.TreeExplainer(lgb_model)
                shap_values = explainer.shap_values(X_sample_trans)
                import matplotlib.pyplot as plt
                shap.summary_plot(shap_values, X_sample_trans, show=False)
                plt.tight_layout()
                shap_path = os.path.join(out_dir, 'shap_summary.png')
                plt.savefig(shap_path)
                plt.close()
                if mlflow is not None:
                    mlflow.log_artifact(shap_path)
            except Exception as e:
                print('SHAP step failed:', e)
    else:
        # Fallback to RandomForest classifier if LightGBM not available
        from sklearn.ensemble import RandomForestClassifier
        acc_pipe = Pipeline([
            ('pre', preprocessor),
            ('model', RandomForestClassifier(n_estimators=200, random_state=44, n_jobs=-1))
        ])
        acc_pipe.fit(X_train, ya_train)
        a_pred = acc_pipe.predict(X_test)
        print('Accidente classification report:')
        print(classification_report(ya_test, a_pred, digits=3))
        joblib.dump(acc_pipe, os.path.join(out_dir, 'acc_pipeline.joblib'))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', required=True, help='Path to CSV or XLSX input file')
    parser.add_argument('--out', default='models', help='Output folder for saved models')
    parser.add_argument('--optuna-trials', type=int, default=0, help='Number of Optuna trials for tuning (0=disabled)')
    parser.add_argument('--max-rows', type=int, default=0, help='If >0, limit dataset to this many rows (random sample)')
    args = parser.parse_args()

    df = load_and_prepare(args.data)
    # apply optional max rows
    if getattr(args, 'max_rows', 0) and int(args.max_rows) > 0:
        m = int(args.max_rows)
        df = df.sample(n=m, random_state=42).reset_index(drop=True)

    # configure optional tuning trials
    train_models.optuna_trials = int(getattr(args, 'optuna_trials', 0))
    train_models(df, out_dir=args.out)


if __name__ == '__main__':
    main()
