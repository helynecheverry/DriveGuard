"""Simulación de impacto a partir de datos de acelerómetro.

Una cámara que mira al conductor no puede detectar un choque. En un vehículo
real eso lo hace un acelerómetro (el del celular o el del sistema eCall). En
el laboratorio se simula: se carga un CSV con columnas t, ax, ay, az y se
busca un pico de aceleración por encima de un umbral (4 g por defecto).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

G = 9.80665


def cargar_acelerometro(origen) -> pd.DataFrame:
    """Lee un CSV (ruta o archivo subido) y devuelve t, ax, ay, az, magnitud_g.

    Acepta aceleraciones en g o en m/s² (si la mediana de la magnitud es
    cercana a 9,8 se asume m/s² y se convierte a g).
    """
    df = pd.read_csv(origen)
    df.columns = [c.strip().lower() for c in df.columns]
    alias = {"tiempo": "t", "time": "t", "timestamp": "t", "x": "ax", "y": "ay", "z": "az"}
    df = df.rename(columns={c: alias[c] for c in df.columns if c in alias})
    faltantes = {"t", "ax", "ay", "az"} - set(df.columns)
    if faltantes:
        raise ValueError(f"Faltan columnas en el CSV: {sorted(faltantes)}")
    df = df[["t", "ax", "ay", "az"]].astype(float).dropna()
    mag = np.sqrt(df.ax ** 2 + df.ay ** 2 + df.az ** 2)
    if mag.median() > 4:            # datos en m/s²
        df[["ax", "ay", "az"]] = df[["ax", "ay", "az"]] / G
        mag = mag / G
    df["magnitud_g"] = mag
    return df.reset_index(drop=True)


def detectar_impacto(df: pd.DataFrame, umbral_g: float = 4.0) -> dict:
    """Devuelve si hubo impacto, el pico en g y el instante del pico."""
    if df.empty:
        return {"impacto": False, "pico_g": 0.0, "t_pico": None, "umbral_g": umbral_g}
    i = int(df["magnitud_g"].idxmax())
    pico = float(df.loc[i, "magnitud_g"])
    return {"impacto": pico >= umbral_g, "pico_g": round(pico, 2),
            "t_pico": round(float(df.loc[i, "t"]), 3), "umbral_g": umbral_g}


def generar_csv_ejemplo(ruta: str | Path, choque: bool = True, segundos: float = 20.0,
                        hz: int = 50, t_choque: float = 12.0, semilla: int = 7) -> Path:
    """Crea un CSV sintético de conducción (con o sin choque) para pruebas."""
    rng = np.random.default_rng(semilla)
    t = np.arange(0, segundos, 1 / hz)
    ax = 0.05 * np.sin(2 * np.pi * 0.3 * t) + rng.normal(0, 0.03, t.size)
    ay = 0.04 * np.sin(2 * np.pi * 0.2 * t) + rng.normal(0, 0.03, t.size)
    az = 1.0 + rng.normal(0, 0.04, t.size)
    for tb in (4.0, 8.5, 15.0):   # baches
        m = np.abs(t - tb) < 0.1
        az[m] += 0.6
    if choque:
        dt = t - t_choque
        pulso = np.where(dt >= 0, 7.5 * np.exp(-dt / 0.05), 0.0)
        ax -= pulso
        ay += 0.3 * pulso
    ruta = Path(ruta)
    ruta.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"t": t.round(3), "ax": ax.round(4), "ay": ay.round(4), "az": az.round(4)}).to_csv(ruta, index=False)
    return ruta
