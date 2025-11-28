import threading
import subprocess
import sys
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import os
import pandas as pd
import joblib
from pathlib import Path
from src.utils import load_and_prepare
import numpy as np


class PipelineGUI(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title('ModeloPredictivo — Pipeline GUI')
        self.geometry('800x600')

        # Notebook with tabs
        self.nb = ttk.Notebook(self)
        self.nb.pack(fill='both', expand=True, padx=8, pady=8)

        # Tab 1: Runner / pipeline
        tab_run = ttk.Frame(self.nb)
        self.nb.add(tab_run, text='Run')

        frm = ttk.Frame(tab_run)
        frm.pack(fill='x', padx=8, pady=8)

        ttk.Label(frm, text='Data file (xlsx/csv):').grid(row=0, column=0, sticky='w')
        self.data_entry = ttk.Entry(frm, width=60)
        self.data_entry.grid(row=0, column=1, sticky='w')
        ttk.Button(frm, text='Browse', command=self.browse_data).grid(row=0, column=2, padx=4)

        # Generate options
        self.generate_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(frm, text='Generate synthetic data', variable=self.generate_var).grid(row=1, column=0, sticky='w')
        ttk.Label(frm, text='Rows:').grid(row=1, column=1, sticky='w')
        self.rows_entry = ttk.Entry(frm, width=10)
        self.rows_entry.insert(0, '10000')
        self.rows_entry.grid(row=1, column=1, sticky='e')

        # Visualize
        self.visualize_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(frm, text='Generate EDA plots', variable=self.visualize_var).grid(row=2, column=0, sticky='w')

        # Train options
        self.train_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(frm, text='Train models', variable=self.train_var).grid(row=3, column=0, sticky='w')
        ttk.Label(frm, text='Optuna trials:').grid(row=3, column=1, sticky='w')
        self.optuna_entry = ttk.Entry(frm, width=10)
        self.optuna_entry.insert(0, '0')
        self.optuna_entry.grid(row=3, column=1, sticky='e')

        ttk.Label(frm, text='Max rows for training (0=all):').grid(row=4, column=0, sticky='w')
        self.maxrows_entry = ttk.Entry(frm, width=10)
        self.maxrows_entry.insert(0, '0')
        self.maxrows_entry.grid(row=4, column=1, sticky='w')

        ttk.Label(frm, text='Models out dir:').grid(row=5, column=0, sticky='w')
        self.out_entry = ttk.Entry(frm, width=40)
        self.out_entry.insert(0, 'models')
        self.out_entry.grid(row=5, column=1, sticky='w')

        # Buttons
        btn_frm = ttk.Frame(tab_run)
        btn_frm.pack(fill='x', padx=8, pady=4)
        ttk.Button(btn_frm, text='Run', command=self.run_pipeline).pack(side='left')
        ttk.Button(btn_frm, text='Stop', command=self.stop_pipeline).pack(side='left')

        # Log output
        self.log = tk.Text(tab_run, wrap='none')
        self.log.pack(fill='both', expand=True, padx=8, pady=8)

        self.process = None
        self.thread = None

        # Tab 2: Results table and file predictions
        tab_results = ttk.Frame(self.nb)
        self.nb.add(tab_results, text='Results')

        res_top = ttk.Frame(tab_results)
        res_top.pack(fill='x', padx=8, pady=8)
        ttk.Label(res_top, text='Input file to predict:').grid(row=0, column=0, sticky='w')
        self.predict_file_entry = ttk.Entry(res_top, width=60)
        self.predict_file_entry.grid(row=0, column=1, sticky='w')
        ttk.Button(res_top, text='Browse', command=self.browse_predict_file).grid(row=0, column=2, padx=4)
        ttk.Button(res_top, text='Load and Predict', command=self.predict_file).grid(row=0, column=3, padx=4)
        ttk.Button(res_top, text='Save predictions to Excel', command=self.save_predictions).grid(row=0, column=4, padx=4)
        ttk.Button(res_top, text='Generate empty template from file', command=self.generate_template_from_file).grid(row=1, column=1, sticky='w', pady=4)

        # Treeview for results
        self.results_tv = ttk.Treeview(tab_results)
        self.results_tv.pack(fill='both', expand=True, padx=8, pady=8)

        # Tab 3: Single case predictor
        tab_single = ttk.Frame(self.nb)
        self.nb.add(tab_single, text='Single Case')

        sc_top = ttk.Frame(tab_single)
        sc_top.pack(fill='x', padx=8, pady=8)
        ttk.Button(sc_top, text='Load columns from file', command=self.load_columns_for_single).pack(side='left')
        ttk.Button(sc_top, text='Predict single case', command=self.predict_single_case).pack(side='left')

        self.single_fields_frame = ttk.Frame(tab_single)
        self.single_fields_frame.pack(fill='both', expand=True, padx=8, pady=8)

        # place to show single case predictions
        self.single_result_label = ttk.Label(tab_single, text='Prediction: -')
        self.single_result_label.pack(padx=8, pady=8)

        # keep track of current loaded columns and last predictions
        self.current_nonresult_columns = []
        self.last_predictions_df = None

    def browse_data(self):
        p = filedialog.askopenfilename(filetypes=[('Excel files','*.xlsx;*.xls'),('CSV','*.csv'),('All','*.*')])
        if p:
            self.data_entry.delete(0,'end')
            self.data_entry.insert(0,p)

    def append_log(self, text):
        self.log.insert('end', text)
        self.log.see('end')

    def run_pipeline(self):
        if self.thread and self.thread.is_alive():
            messagebox.showinfo('Running', 'Pipeline is already running')
            return

        data_file = self.data_entry.get().strip() or 'datos_entrenamiento_10000.xlsx'
        optuna = int(self.optuna_entry.get().strip() or 0)
        maxrows = int(self.maxrows_entry.get().strip() or 0)
        outdir = self.out_entry.get().strip() or 'models'
        do_generate = bool(self.generate_var.get())
        do_visualize = bool(self.visualize_var.get())
        do_train = bool(self.train_var.get())
        rows = int(self.rows_entry.get().strip() or 10000)

        # Build sequence of commands using main.py
        cmds = []
        if do_generate:
            gen_path = os.path.join('data', 'raw', 'generator.py')
            cmds.append([sys.executable, gen_path, '--n', str(rows), '--output', data_file])
        if do_visualize:
            cmds.append([sys.executable, '-m', 'src.visualize', '--data', data_file, '--out', 'reports'])
        if do_train:
            cmd = [sys.executable, '-m', 'src.train', '--data', data_file, '--out', outdir]
            if optuna > 0:
                cmd += ['--optuna-trials', str(optuna)]
            if maxrows > 0:
                cmd += ['--max-rows', str(maxrows)]
            cmds.append(cmd)

        if not cmds:
            messagebox.showinfo('Nothing to do', 'No steps selected')
            return

        # Run in thread
        def target():
            try:
                for c in cmds:
                    self.append_log('\n>>> ' + ' '.join(c) + '\n')
                    self.process = subprocess.Popen(c, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
                    for line in self.process.stdout:
                        self.append_log(line)
                    self.process.wait()
                    if self.process.returncode != 0:
                        self.append_log(f'Command failed with code {self.process.returncode}\n')
                        break
                self.append_log('\nPipeline finished.\n')
            except Exception as e:
                self.append_log(f'Error: {e}\n')
            finally:
                self.process = None

        self.thread = threading.Thread(target=target, daemon=True)
        self.thread.start()

    def stop_pipeline(self):
        if self.process:
            try:
                self.process.terminate()
                self.append_log('\nProcess terminated by user.\n')
            except Exception as e:
                self.append_log(f'Failed to terminate process: {e}\n')
        else:
            messagebox.showinfo('No process', 'No running process to stop')

    # --- Results / prediction helpers ---
    def browse_predict_file(self):
        p = filedialog.askopenfilename(filetypes=[('Excel files','*.xlsx;*.xls'),('CSV','*.csv'),('All','*.*')])
        if p:
            self.predict_file_entry.delete(0,'end')
            self.predict_file_entry.insert(0,p)

    def _load_models(self, models_dir=None):
        models_dir = models_dir or self.out_entry.get().strip() or 'models'
        base = Path(models_dir)
        prob_p = base / 'prob_pipeline.joblib'
        cond_p = base / 'cond_pipeline.joblib'
        acc_p = base / 'acc_pipeline.joblib'
        missing = [str(p.name) for p in (prob_p, cond_p, acc_p) if not p.exists()]
        if missing:
            messagebox.showerror('Models missing', f'Missing model files in {models_dir}: {missing}')
            return None, None, None
        try:
            prob = joblib.load(prob_p)
            cond = joblib.load(cond_p)
            acc = joblib.load(acc_p)
            return prob, cond, acc
        except Exception as e:
            messagebox.showerror('Load error', f'Failed loading models: {e}')
            return None, None, None

    def predict_file(self):
        p = self.predict_file_entry.get().strip()
        if not p:
            messagebox.showinfo('No file', 'Please select a file to predict')
            return
        try:
            if p.lower().endswith('.csv'):
                df = pd.read_csv(p)
            else:
                df = pd.read_excel(p)
        except Exception as e:
            messagebox.showerror('Read error', f'Failed to read file: {e}')
            return

        prob_m, cond_m, acc_m = self._load_models()
        if prob_m is None:
            return

        # Prepare input: drop result columns if present
        result_cols = ['Probabilidad de accidente', 'Accidente', 'Condicion del Vehiculo']
        input_df = df.copy()
        for c in result_cols:
            if c in input_df.columns:
                input_df = input_df.drop(columns=[c])

        # run standard preprocessing used during training (adds Hora_hour/ Hora_min etc.)
        try:
            input_df = load_and_prepare(input_df)
        except Exception as e:
            messagebox.showerror('Preprocess error', f'Failed to preprocess input data: {e}')
            return

        # Ensure the input has all columns expected by the model preprocessor; fill missing ones
        try:
            pre = prob_m.named_steps.get('pre')
            num_cols = list(pre.transformers[0][2])
            cat_cols = list(pre.transformers[1][2])
            expected = num_cols + cat_cols
        except Exception:
            expected = ['Mes', 'Hora_hour', 'Hora_min', 'Distancia Kilometros', 'Tipo de Vehiculo', 'Clima']

        for col in expected:
            if col not in input_df.columns:
                input_df[col] = np.nan

        try:
            prob_pred = prob_m.predict(input_df)
        except Exception as e:
            messagebox.showerror('Model predict error', f'Prob model prediction failed: {e}')
            return

        try:
            cond_pred = cond_m.predict(input_df)
        except Exception as e:
            messagebox.showerror('Model predict error', f'Cond model prediction failed: {e}')
            return

        try:
            acc_pred = acc_m.predict(input_df)
            # attempt to get probabilities
            try:
                acc_proba = acc_m.predict_proba(input_df)[:, 1]
            except Exception:
                acc_proba = None
        except Exception as e:
            messagebox.showerror('Model predict error', f'Acc model prediction failed: {e}')
            return

        # attach results
        out = input_df.reset_index(drop=True).copy()
        # clip regression outputs to sensible 0-100 range and attach
        out['Probabilidad de accidente'] = np.clip(prob_pred, 0.0, 100.0)
        out['Condicion del Vehiculo'] = np.clip(cond_pred, 0.0, 100.0)
        # map classifier output
        mapped = []
        for v in acc_pred:
            if v is None:
                mapped.append(None)
            else:
                mapped.append('Sí' if int(v) == 1 else 'No')
        out['Accidente'] = mapped
        # add probability column if available
        if 'acc_proba' in locals() and acc_proba is not None:
            out['Accidente_proba'] = acc_proba

        # keep last predictions
        self.last_predictions_df = out

        # display in treeview (limit to first 200 rows)
        self._display_results(out.head(200))
        messagebox.showinfo('Predictions', f'Predictions added for {len(out)} rows. Displaying first 200.')

    def _display_results(self, df):
        # clear treeview
        for col in self.results_tv.get_children():
            self.results_tv.delete(col)
        # configure columns
        cols = list(df.columns)
        self.results_tv['columns'] = cols
        self.results_tv['show'] = 'headings'
        for c in cols:
            self.results_tv.heading(c, text=c)
            self.results_tv.column(c, width=120, anchor='w')
        # insert rows
        for _, row in df.iterrows():
            vals = [self._format_cell(row[c]) for c in cols]
            self.results_tv.insert('', 'end', values=vals)

    def _format_cell(self, v):
        if pd.isna(v):
            return ''
        # format percentages with one decimal when column looks like percent
        try:
            if isinstance(v, float):
                return f'{v:.1f}'
        except Exception:
            pass
        return str(v)

    def save_predictions(self):
        if self.last_predictions_df is None:
            messagebox.showinfo('No predictions', 'No predictions to save. Run "Load and Predict" first.')
            return
        p = filedialog.asksaveasfilename(defaultextension='.xlsx', filetypes=[('Excel','*.xlsx')])
        if not p:
            return
        try:
            # write to excel
            self.last_predictions_df.to_excel(p, index=False)
            messagebox.showinfo('Saved', f'Predictions saved to {p}')
        except Exception as e:
            messagebox.showerror('Save error', f'Failed to save file: {e}')

    def generate_template_from_file(self):
        p = self.predict_file_entry.get().strip() or self.data_entry.get().strip()
        if not p:
            messagebox.showinfo('No file', 'Please select a source file to build template from')
            return
        try:
            if p.lower().endswith('.csv'):
                df = pd.read_csv(p)
            else:
                df = pd.read_excel(p)
        except Exception as e:
            messagebox.showerror('Read error', f'Failed to read file: {e}')
            return

        # remove result columns
        result_cols = ['Probabilidad de accidente', 'Accidente', 'Condicion del Vehiculo']
        cols = [c for c in df.columns if c not in result_cols]
        template_df = pd.DataFrame(columns=cols)

        outp = filedialog.asksaveasfilename(defaultextension='.xlsx', filetypes=[('Excel','*.xlsx')], title='Save template as')
        if not outp:
            return
        try:
            template_df.to_excel(outp, index=False)
            messagebox.showinfo('Template', f'Empty template saved to {outp}')
        except Exception as e:
            messagebox.showerror('Save error', f'Failed to save template: {e}')

    def load_columns_for_single(self):
        # use predict_file_entry or ask
        p = self.predict_file_entry.get().strip() or self.data_entry.get().strip()
        if not p:
            p = filedialog.askopenfilename(filetypes=[('Excel files','*.xlsx;*.xls'),('CSV','*.csv')])
            if not p:
                return
        try:
            if p.lower().endswith('.csv'):
                df = pd.read_csv(p, nrows=5)
            else:
                df = pd.read_excel(p, nrows=5)
        except Exception as e:
            messagebox.showerror('Read error', f'Failed to read file: {e}')
            return

        result_cols = ['Probabilidad de accidente', 'Accidente', 'Condicion del Vehiculo']
        cols = [c for c in df.columns if c not in result_cols]
        self.current_nonresult_columns = cols

        # clear existing
        for w in self.single_fields_frame.winfo_children():
            w.destroy()
        self.single_entries = {}
        for i, c in enumerate(cols):
            ttk.Label(self.single_fields_frame, text=c).grid(row=i, column=0, sticky='w')
            e = ttk.Entry(self.single_fields_frame, width=40)
            e.grid(row=i, column=1, sticky='w')
            self.single_entries[c] = e
        messagebox.showinfo('Loaded', f'Loaded {len(cols)} columns for manual entry')

    def predict_single_case(self):
        if not hasattr(self, 'single_entries') or not self.single_entries:
            messagebox.showinfo('No columns', 'Load columns first using "Load columns from file"')
            return
        row = {}
        for c, e in self.single_entries.items():
            v = e.get().strip()
            row[c] = v if v != '' else None
        df = pd.DataFrame([row])

        prob_m, cond_m, acc_m = self._load_models()
        if prob_m is None:
            return

        try:
            df_prep = load_and_prepare(df)
        except Exception as e:
            messagebox.showerror('Preprocess error', f'Failed to preprocess single case: {e}')
            return

        # Ensure expected columns exist
        try:
            pre = prob_m.named_steps.get('pre')
            num_cols = list(pre.transformers[0][2])
            cat_cols = list(pre.transformers[1][2])
            expected = num_cols + cat_cols
        except Exception:
            expected = ['Mes', 'Hora_hour', 'Hora_min', 'Distancia Kilometros', 'Tipo de Vehiculo', 'Clima']
        for col in expected:
            if col not in df_prep.columns:
                df_prep[col] = np.nan

        try:
            prob_pred = float(prob_m.predict(df_prep)[0])
            cond_pred = float(cond_m.predict(df_prep)[0])
            acc_pred = acc_m.predict(df_prep)[0]
            # try to get probability
            try:
                acc_proba = float(acc_m.predict_proba(df_prep)[0, 1])
            except Exception:
                acc_proba = None
            acc_lab = 'Sí' if int(acc_pred) == 1 else 'No'
        except Exception as e:
            messagebox.showerror('Predict error', f'Prediction failed: {e}')
            return

        # clip regressions
        prob_pred = float(np.clip(prob_pred, 0.0, 100.0))
        cond_pred = float(np.clip(cond_pred, 0.0, 100.0))

        text = f'Probabilidad de accidente: {prob_pred:.1f}%    Condicion del Vehiculo: {cond_pred:.1f}%    Accidente: {acc_lab}'
        if acc_proba is not None:
            text += f'    Accidente_proba: {acc_proba*100:.1f}%'
        self.single_result_label.config(text=text)


def main():
    app = PipelineGUI()
    app.mainloop()


if __name__ == '__main__':
    main()
