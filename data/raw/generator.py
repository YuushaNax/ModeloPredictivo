import pandas as pd
import numpy as np
import argparse


def generate(n=10000, out_path="datos_entrenamiento_10000.xlsx"):
    rng = np.random.default_rng(42)

    # Datos base
    mes = rng.integers(1, 13, size=n)
    horas_h = rng.integers(0, 24, size=n)
    horas_m = rng.integers(0, 60, size=n)
    hora = [f"{h}:{m:02d}" for h, m in zip(horas_h, horas_m)]
    dist = rng.integers(1, 301, size=n)

    vehiculos = ["Bus", "Carro", "Moto", "Camión", "Van"]
    veh = rng.choice(vehiculos, size=n)

    climas = ["Soleado", "Lluvioso", "Nublado", "Nevado", "Viento"]
    cl = rng.choice(climas, size=n)

    # Nuevas columnas sugeridas
    dias = rng.integers(1, 8, size=n)  # 1=Lunes .. 7=Domingo
    carreteras = rng.choice(["Urbana", "Rural", "Autopista"], size=n)
    velocidad = rng.normal(60, 15, size=n).clip(10, 180)  # km/h
    edad = rng.integers(18, 81, size=n)
    experiencia = (edad - 18) * rng.random(size=n)  # años de experiencia aproxim
    alcohol = rng.choice([0, 1], size=n, p=[0.95, 0.05])
    visibilidad = rng.choice(["Buena", "Mala", "Niebla"], size=n, p=[0.75, 0.2, 0.05])
    estado_via = rng.choice(["Seca", "Mojada", "Helada"], size=n, p=[0.8, 0.18, 0.02])
    iluminacion = rng.choice(["Dia", "Noche"], size=n, p=[0.7, 0.3])
    cinturon = rng.choice(["Sí", "No"], size=n, p=[0.9, 0.1])

    # Condición del vehículo (porcentaje)
    cond = 50 + rng.random(size=n) * 50  # entre 50% y 100%

    # -----------------------------
    # LÓGICA DE PROBABILIDAD REALISTA
    # -----------------------------
    prob_list = []
    for i in range(n):
        base = 5  # riesgo base mínimo

        # --- Clima ---
        if cl[i] == "Soleado":
            base += 2
        elif cl[i] == "Nublado":
            base += 5
        elif cl[i] == "Viento":
            base += 7
        elif cl[i] == "Lluvioso":
            base += 15
        elif cl[i] == "Nevado":
            base += 25

        # --- Distancia ---
        if dist[i] < 30:
            base += 2
        elif dist[i] < 100:
            base += 8
        else:
            base += 18

        # --- Vehículo ---
        if veh[i] == "Moto":
            base += 15
        elif veh[i] == "Camión":
            base += 8
        elif veh[i] == "Carro":
            base += 6
        elif veh[i] == "Van":
            base += 5
        elif veh[i] == "Bus":
            base += 3

        # --- Condición del vehículo ---
        if cond[i] < 60:
            base += 10
        elif cond[i] < 80:
            base += 5
        else:
            base -= 5

        # --- Velocidad ---
        if velocidad[i] > 120:
            base += 20
        elif velocidad[i] > 80:
            base += 10
        elif velocidad[i] > 50:
            base += 5

        # --- Edad y experiencia ---
        if edad[i] < 25:
            base += 8
        if edad[i] > 70:
            base += 7
        if experiencia[i] > 20:
            base -= 5

        # --- Alcohol ---
        if alcohol[i] == 1:
            base += 30

        # --- Visibilidad ---
        if visibilidad[i] == 'Mala':
            base += 10
        elif visibilidad[i] == 'Niebla':
            base += 20

        # --- Estado de la via ---
        if estado_via[i] == 'Mojada':
            base += 8
        elif estado_via[i] == 'Helada':
            base += 20

        # --- Iluminacion ---
        if iluminacion[i] == 'Noche':
            base += 5

        # --- Tipo de carretera ---
        if carreteras[i] == 'Autopista':
            base += 7
        elif carreteras[i] == 'Rural':
            base += 5

        # --- Cinturon --- (not directly risk of accident occurrence, but correlated with severity)
        if cinturon[i] == 'No':
            base += 3

        # Asegurar un rango válido
        prob = np.clip(base + rng.normal(0, 3), 1, 99)
        prob_list.append(prob)

    # -----------------------------
    # GENERAR COLUMNA "Accidente"
    # -----------------------------
    acc = ["Sí" if rng.random() < (prob_list[i] / 100) else "No" for i in range(n)]

    # -----------------------------
    # CREAR DATAFRAME
    # -----------------------------
    df = pd.DataFrame({
        "Mes": mes,
        "Hora de Salida": hora,
        "Distancia Kilometros": dist,
        "Tipo de Vehiculo": veh,
        "Clima": cl,
        "Probabilidad de accidente": [f"{p:.1f}%" for p in prob_list],
        "Accidente": acc,
        "Condicion del Vehiculo": [f"{c:.1f}%" for c in cond],
        # Nuevas columnas
        "Dia de la Semana": dias,
        "Tipo de Carretera": carreteras,
        "Velocidad Promedio": [f"{v:.1f}" for v in velocidad],
        "Edad Conductor": edad,
        "Experiencia Conductor": [f"{e:.1f}" for e in experiencia],
        "Alcohol en Sangre": alcohol,
        "Visibilidad": visibilidad,
        "Estado de la Via": estado_via,
        "Iluminacion": iluminacion,
        "Cinturon": cinturon,
    })

    df.to_excel(out_path, index=False)
    print(f"Archivo generado: {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--n', type=int, default=10000, help='Number of rows to generate')
    parser.add_argument('--output', default='datos_entrenamiento_10000.xlsx', help='Output xlsx file')
    args = parser.parse_args()
    generate(n=args.n, out_path=args.output)


if __name__ == '__main__':
    main()
