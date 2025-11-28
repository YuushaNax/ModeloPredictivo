#!/usr/bin/env python
"""Orchestrator CLI to run the full pipeline quickly.

Usage examples:
  python main.py --generate-data
  python main.py --convert-excel data/raw/datos_entrenamiento_10000.xlsx data/raw/datos_entrenamiento_10000.csv
  python main.py --visualize --data data/raw/datos_entrenamiento_10000.xlsx --out reports
  python main.py --train --data data/raw/datos_entrenamiento_10000.xlsx --out models --optuna-trials 20
  python main.py --serve --host 127.0.0.1 --port 8000
  python main.py --all --optuna-trials 10
"""

import argparse
import subprocess
import sys
import os
from pathlib import Path


def run_cmd(cmd, check=True):
    print('Running:', ' '.join(map(str, cmd)))
    return subprocess.run(cmd, check=check)


def generate_data():
    script = Path('data') / 'raw' / 'generator.py'
    if not script.exists():
        raise FileNotFoundError(f'{script} not found')
    return run_cmd([sys.executable, str(script)])


def convert_excel(input_path, output_path):
    script = Path('data') / 'make_csv_from_excel.py'
    if not script.exists():
        raise FileNotFoundError(f'{script} not found')
    return run_cmd([sys.executable, str(script), '--input', str(input_path), '--output', str(output_path)])


def visualize(data, out='reports'):
    return run_cmd([sys.executable, '-m', 'src.visualize', '--data', str(data), '--out', str(out)])


def train(data, out='models', optuna_trials=0):
    cmd = [sys.executable, '-m', 'src.train', '--data', str(data), '--out', str(out)]
    if int(optuna_trials) > 0:
        cmd += ['--optuna-trials', str(int(optuna_trials))]
    return run_cmd(cmd)


def serve(host='0.0.0.0', port=8000):
    # Run uvicorn to serve the FastAPI app
    return run_cmd([sys.executable, '-m', 'uvicorn', 'src.app:app', '--host', str(host), '--port', str(port)])


def parse_args():
    p = argparse.ArgumentParser(description='Run ModeloPredictivo pipeline tasks')
    p.add_argument('--generate-data', action='store_true', help='Run the synthetic data generator')
    p.add_argument('--convert-excel', nargs=2, metavar=('IN','OUT'), help='Convert Excel to CSV')
    p.add_argument('--visualize', action='store_true', help='Run EDA visualizations')
    p.add_argument('--train', action='store_true', help='Train models')
    p.add_argument('--serve', action='store_true', help='Serve FastAPI app')
    p.add_argument('--all', action='store_true', help='Run generate -> visualize -> train')
    p.add_argument('--data', default='datos_entrenamiento_10000.xlsx', help='Path to input data (xlsx or csv)')
    p.add_argument('--out', default='models', help='Output folder for models or reports')
    p.add_argument('--optuna-trials', type=int, default=0, help='Number of Optuna trials to run during training')
    p.add_argument('--host', default='127.0.0.1', help='Host for serving the API')
    p.add_argument('--port', type=int, default=8000, help='Port for serving the API')
    return p.parse_args()


def main():
    args = parse_args()

    try:
        if args.generate_data:
            generate_data()

        if args.convert_excel:
            inp, outp = args.convert_excel
            convert_excel(inp, outp)

        if args.visualize:
            visualize(args.data, out=args.out if args.out else 'reports')

        if args.train:
            train(args.data, out=args.out if args.out else 'models', optuna_trials=args.optuna_trials)

        if args.all:
            print('Running full pipeline: generate -> visualize -> train')
            generate_data()
            # default excel name the generator produces
            data_path = Path(args.data)
            # convert to csv next to excel
            csv_path = data_path.with_suffix('.csv')
            try:
                convert_excel(data_path, csv_path)
            except Exception:
                print('Conversion skipped or failed; continuing with excel')
            visualize(data_path, out='reports')
            train(data_path, out='models', optuna_trials=args.optuna_trials)

        if args.serve:
            serve(host=args.host, port=args.port)

    except subprocess.CalledProcessError as exc:
        print('Command failed:', exc)
        sys.exit(1)
    except Exception as e:
        print('Error:', e)
        sys.exit(2)


if __name__ == '__main__':
    main()
