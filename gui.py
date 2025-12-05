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
import src.visualize as visualize
import src.prediction_viz as prediction_viz
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
        self.nb.add(tab_run, text='Ejecutar')

        frm = ttk.Frame(tab_run)
        frm.pack(fill='x', padx=8, pady=8)

        ttk.Label(frm, text='Archivo de datos (xlsx/csv):').grid(row=0, column=0, sticky='w')
        self.data_entry = ttk.Entry(frm, width=60)
        self.data_entry.grid(row=0, column=1, sticky='w')
        ttk.Button(frm, text='Examinar', command=self.browse_data).grid(row=0, column=2, padx=4)

        # Generate options
        self.generate_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(frm, text='Generar datos sintéticos', variable=self.generate_var).grid(row=1, column=0, sticky='w')
        ttk.Label(frm, text='Filas:').grid(row=1, column=1, sticky='w')
        self.rows_entry = ttk.Entry(frm, width=10)
        self.rows_entry.insert(0, '10000')
        self.rows_entry.grid(row=1, column=1, sticky='e')

        # Visualize
        self.visualize_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(frm, text='Generar gráficas EDA', variable=self.visualize_var).grid(row=2, column=0, sticky='w')

        # Train options
        self.train_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(frm, text='Entrenar modelos', variable=self.train_var).grid(row=3, column=0, sticky='w')
        ttk.Label(frm, text='Ensayos Optuna:').grid(row=3, column=1, sticky='w')
        self.optuna_entry = ttk.Entry(frm, width=10)
        self.optuna_entry.insert(0, '0')
        self.optuna_entry.grid(row=3, column=1, sticky='e')

        ttk.Label(frm, text='Máx filas para entrenar (0=todas):').grid(row=4, column=0, sticky='w')
        self.maxrows_entry = ttk.Entry(frm, width=10)
        self.maxrows_entry.insert(0, '0')
        self.maxrows_entry.grid(row=4, column=1, sticky='w')

        ttk.Label(frm, text='Directorio salida modelos:').grid(row=5, column=0, sticky='w')
        self.out_entry = ttk.Entry(frm, width=40)
        self.out_entry.insert(0, 'models')
        self.out_entry.grid(row=5, column=1, sticky='w')

        # Buttons
        btn_frm = ttk.Frame(tab_run)
        btn_frm.pack(fill='x', padx=8, pady=4)
        ttk.Button(btn_frm, text='Ejecutar', command=self.run_pipeline).pack(side='left')
        ttk.Button(btn_frm, text='Detener', command=self.stop_pipeline).pack(side='left')

        # Log output
        self.log = tk.Text(tab_run, wrap='none')
        self.log.pack(fill='both', expand=True, padx=8, pady=8)

        self.process = None
        self.thread = None

        # Tab 2: Results table and file predictions
        tab_results = ttk.Frame(self.nb)
        self.nb.add(tab_results, text='Resultados')

        res_top = ttk.Frame(tab_results)
        res_top.pack(fill='x', padx=8, pady=8)
        ttk.Label(res_top, text='Archivo de entrada para predecir:').grid(row=0, column=0, sticky='w')
        self.predict_file_entry = ttk.Entry(res_top, width=60)
        self.predict_file_entry.grid(row=0, column=1, sticky='w')
        ttk.Button(res_top, text='Examinar', command=self.browse_predict_file).grid(row=0, column=2, padx=4)
        ttk.Button(res_top, text='Cargar y Predecir', command=self.predict_file).grid(row=0, column=3, padx=4)
        ttk.Button(res_top, text='Guardar predicciones (Excel)', command=self.save_predictions).grid(row=0, column=4, padx=4)
        ttk.Button(res_top, text='Generar plantilla vacía', command=self.generate_template_from_file).grid(row=1, column=1, sticky='w', pady=4)

        # Area for analysis with scrollbars (both vertical and horizontal)
        results_container = ttk.Frame(tab_results)
        results_container.pack(fill='both', expand=True, padx=8, pady=8)
        
        # Create canvas with both scrollbars
        self.results_canvas = tk.Canvas(results_container, bg='white')
        vsb = ttk.Scrollbar(results_container, orient='vertical', command=self.results_canvas.yview)
        hsb = ttk.Scrollbar(results_container, orient='horizontal', command=self.results_canvas.xview)
        self.results_canvas.configure(yscrollcommand=vsb.set, xscrollcommand=hsb.set)
        
        # Grid layout for scrollbars
        self.results_canvas.grid(row=0, column=0, sticky='nsew')
        vsb.grid(row=0, column=1, sticky='ns')
        hsb.grid(row=1, column=0, sticky='ew')
        results_container.grid_rowconfigure(0, weight=1)
        results_container.grid_columnconfigure(0, weight=1)
        
        # Create frame inside canvas
        self.results_area = ttk.Frame(self.results_canvas)
        self.results_canvas.create_window((0, 0), window=self.results_area, anchor='nw')
        
        # Update scroll region when frame changes
        def on_frame_configure(event=None):
            self.results_canvas.configure(scrollregion=self.results_canvas.bbox('all'))
        
        self.results_area.bind('<Configure>', on_frame_configure)
        
        # Enable mouse wheel scrolling for vertical
        def _on_mousewheel(event):
            self.results_canvas.yview_scroll(int(-1*(event.delta/120)), 'units')
        
        self.results_canvas.bind_all('<MouseWheel>', _on_mousewheel)
        
        self._report_images = []

        # Tab 3: Single case predictor
        tab_single = ttk.Frame(self.nb)
        self.nb.add(tab_single, text='Caso Único')

        sc_top = ttk.Frame(tab_single)
        sc_top.pack(fill='x', padx=8, pady=8)
        ttk.Button(sc_top, text='Cargar columnas desde archivo', command=self.load_columns_for_single).pack(side='left')
        ttk.Button(sc_top, text='Predecir caso único', command=self.predict_single_case).pack(side='left')

        self.single_fields_frame = ttk.Frame(tab_single)
        self.single_fields_frame.pack(fill='both', expand=True, padx=8, pady=8)

        # place to show single case predictions
        self.single_result_label = tk.Label(tab_single, text='Predicción: -', font=('Courier', 10), justify='left', bg='#f0f0f0', padx=10, pady=10)
        self.single_result_label.pack(padx=8, pady=8, fill='both')

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
            messagebox.showinfo('En ejecución', 'El pipeline ya está en ejecución')
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
            messagebox.showinfo('Nada para hacer', 'No se seleccionaron pasos')
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
                self.append_log('\nPipeline finalizado.\n')
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
                self.append_log('\nProceso terminado por el usuario.\n')
            except Exception as e:
                self.append_log(f'No se pudo terminar el proceso: {e}\n')
        else:
            messagebox.showinfo('Sin proceso', 'No hay procesos en ejecución para detener')

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
            messagebox.showinfo('Sin archivo', 'Por favor seleccione un archivo para predecir')
            return
        try:
            if p.lower().endswith('.csv'):
                df = pd.read_csv(p)
            else:
                df = pd.read_excel(p)
        except Exception as e:
            messagebox.showerror('Error de lectura', f'No se pudo leer el archivo: {e}')
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
            messagebox.showerror('Error de preprocesado', f'No se pudo preprocesar los datos de entrada: {e}')
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
            messagebox.showerror('Error de modelo', f'La predicción (probabilidad) falló: {e}')
            return

        try:
            cond_pred = cond_m.predict(input_df)
        except Exception as e:
            messagebox.showerror('Error de modelo', f'La predicción (condición) falló: {e}')
            return

        try:
            acc_pred = acc_m.predict(input_df)
            # attempt to get probabilities
            try:
                acc_proba = acc_m.predict_proba(input_df)[:, 1]
            except Exception:
                acc_proba = None
        except Exception as e:
            messagebox.showerror('Error de modelo', f'La predicción (accidente) falló: {e}')
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

        # Always generate fresh EDA plots for the predicted dataset
        rep_out = os.path.join('reports', 'gui_temp')
        try:
            # Remove any old images in the temp folder
            if os.path.exists(rep_out):
                for f in os.listdir(rep_out):
                    fp = os.path.join(rep_out, f)
                    if os.path.isfile(fp):
                        try:
                            os.remove(fp)
                        except Exception:
                            pass
            else:
                os.makedirs(rep_out, exist_ok=True)
            visualize.make_plots(out, outdir=rep_out)
        except Exception as e:
            messagebox.showerror('Error EDA', f'No se pudieron generar las gráficas EDA: {e}')
        self._display_analysis(input_df, out, reports_dir=rep_out)
        messagebox.showinfo('Predicciones', f'Se añadieron predicciones para {len(out)} filas. Análisis mostrado.')

    def _display_results(self, df):
        # legacy: replaced by analysis view; keep for compatibility
        return

    def _display_analysis(self, input_df, predictions_df, reports_dir='reports'):
        # clear previous area
        for w in self.results_area.winfo_children():
            w.destroy()
        self._report_images = []

        # Metrics summary (dataset-level and prediction-level)
        try:
            n = len(input_df)
            mean_prob = float(predictions_df['Probabilidad de accidente'].mean()) if 'Probabilidad de accidente' in predictions_df.columns else float('nan')
            mean_cond = float(predictions_df['Condicion del Vehiculo'].mean()) if 'Condicion del Vehiculo' in predictions_df.columns else float('nan')
            # predicted accident rate
            acc_rate = None
            if 'Accidente' in predictions_df.columns:
                acc_vals = predictions_df['Accidente'].apply(lambda x: 1 if str(x).strip().lower() in ['sí','si','1','true'] else 0)
                acc_rate = float(acc_vals.mean())
        except Exception:
            n = len(input_df)
            mean_prob = mean_cond = acc_rate = None

        # Header for metrics
        header = ttk.Label(self.results_area, text='📊 RESUMEN DE MÉTRICAS', font=('Helvetica', 12, 'bold'))
        header.pack(fill='x', padx=6, pady=(12, 6))

        metrics_frame = ttk.Frame(self.results_area)
        metrics_frame.pack(fill='x', padx=6, pady=6)
        ttk.Label(metrics_frame, text=f'Filas: {n}').grid(row=0, column=0, sticky='w')
        ttk.Label(metrics_frame, text=f'Media Prob. Predicha: {mean_prob:.2f}%' if mean_prob==mean_prob else 'Media Prob. Predicha: -').grid(row=0, column=1, sticky='w', padx=8)
        ttk.Label(metrics_frame, text=f'Media Condición Predicha: {mean_cond:.2f}%' if mean_cond==mean_cond else 'Media Condición Predicha: -').grid(row=0, column=2, sticky='w', padx=8)
        ttk.Label(metrics_frame, text=f'Tasa Accidente predicha: {acc_rate:.3f}' if acc_rate is not None else 'Tasa Accidente predicha: -').grid(row=0, column=3, sticky='w', padx=8)

        # Generar visualizaciones mejoradas de predicciones
        try:
            prediction_viz.generate_all_prediction_visualizations(predictions_df, output_dir=reports_dir)
        except Exception as e:
            print(f"Error generando visualizaciones de predicción: {e}")

        # Collect all visualization files with their display names
        imgs = [
            ('prediction_summary.png', '📈 Resumen de Predicciones'),
            ('detailed_metrics.png', '📊 Métricas Detalladas'),
            ('prediction_quality.png', '✓ Calidad de Predicciones'),
            ('hist_probabilidad.png', '📉 Distribución de Probabilidades'),
            ('count_accidente.png', '⚠️ Conteo de Accidentes'),
            ('corr_numeric.png', '🔗 Correlaciones Numéricas'),
            ('box_prob_by_vehicle.png', '🚗 Probabilidad por Vehículo'),
        ]

        # load and insert images with headers
        for img_name, header_text in imgs:
            p = os.path.join(reports_dir, img_name)
            if os.path.exists(p):
                try:
                    # Add header for this graphic
                    img_header = ttk.Label(self.results_area, text=header_text, font=('Helvetica', 11, 'bold'))
                    img_header.pack(fill='x', padx=6, pady=(12, 6))
                    
                    # Load and display image
                    img = tk.PhotoImage(file=p)
                    self._report_images.append(img)
                    lbl = ttk.Label(self.results_area, image=img)
                    lbl.pack(fill='x', padx=6, pady=6)
                except Exception:
                    # skip images that cannot be loaded
                    continue

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
            messagebox.showinfo('Sin predicciones', 'No hay predicciones para guardar. Ejecute "Cargar y Predecir" primero.')
            return
        p = filedialog.asksaveasfilename(defaultextension='.xlsx', filetypes=[('Excel','*.xlsx')])
        if not p:
            return
        try:
            # write to excel
            self.last_predictions_df.to_excel(p, index=False)
            messagebox.showinfo('Guardado', f'Predicciones guardadas en {p}')
        except Exception as e:
            messagebox.showerror('Error al guardar', f'No se pudo guardar el archivo: {e}')

    def generate_template_from_file(self):
        p = self.predict_file_entry.get().strip() or self.data_entry.get().strip()
        if not p:
            messagebox.showinfo('Sin archivo', 'Por favor seleccione un archivo fuente para crear la plantilla')
            return
        try:
            if p.lower().endswith('.csv'):
                df = pd.read_csv(p)
            else:
                df = pd.read_excel(p)
        except Exception as e:
            messagebox.showerror('Error de lectura', f'No se pudo leer el archivo: {e}')
            return

        # remove result columns
        result_cols = ['Probabilidad de accidente', 'Accidente', 'Condicion del Vehiculo']
        cols = [c for c in df.columns if c not in result_cols]
        template_df = pd.DataFrame(columns=cols)

        outp = filedialog.asksaveasfilename(defaultextension='.xlsx', filetypes=[('Excel','*.xlsx')], title='Guardar plantilla como')
        if not outp:
            return
        try:
            template_df.to_excel(outp, index=False)
            messagebox.showinfo('Plantilla', f'Plantilla vacía guardada en {outp}')
        except Exception as e:
            messagebox.showerror('Error al guardar', f'No se pudo guardar la plantilla: {e}')

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
            messagebox.showerror('Error de lectura', f'No se pudo leer el archivo: {e}')
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
        messagebox.showinfo('Cargado', f'Se cargaron {len(cols)} columnas para entrada manual')

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
            messagebox.showerror('Error de preprocesado', f'No se pudo preprocesar el caso único: {e}')
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
            messagebox.showerror('Error de predicción', f'La predicción falló: {e}')
            return

        # clip regressions
        prob_pred = float(np.clip(prob_pred, 0.0, 100.0))
        cond_pred = float(np.clip(cond_pred, 0.0, 100.0))

        # Crear una representación visual mejorada
        risk_level = "ALTO ⚠️" if prob_pred > 60 else "MEDIO ⚠" if prob_pred > 30 else "BAJO ✓"
        vehicle_condition = "BUENA ✓" if cond_pred > 80 else "REGULAR ⚠" if cond_pred > 50 else "MALA ⚠️"
        
        text = (f'╔════════════════════════════════════════════════════════════════╗\n'
                f'║                    PREDICCIÓN DEL CASO ÚNICO                    ║\n'
                f'╠════════════════════════════════════════════════════════════════╣\n'
                f'║ Riesgo de Accidente:           {prob_pred:6.1f}%    [{risk_level:^15}] ║\n'
                f'║ Condición del Vehículo:        {cond_pred:6.1f}%    [{vehicle_condition:^15}] ║\n'
                f'║ Accidente Predicho:            {acc_lab:^8}                      ║\n')
        
        if acc_proba is not None:
            text += f'║ Probabilidad de Accidente:     {acc_proba*100:6.1f}%                        ║\n'
        
        text += f'╚════════════════════════════════════════════════════════════════╝'
        
        self.single_result_label.config(text=text)


def main():
    app = PipelineGUI()
    app.mainloop()


if __name__ == '__main__':
    main()
