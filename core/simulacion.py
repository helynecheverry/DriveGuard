"""Generador de mediciones sintéticas para probar el motor de riesgo sin cámara.

Se usa en pytest, en el notebook 03 y en scripts/datos_demo.py. Cada
escenario reproduce un patrón típico: parpadeo normal, cansancio
(parpadeos lentos y bostezos), microsueño, conductor que no responde,
distracción y cabeceos.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from core.config import ConfigRiesgo
from core.riesgo import Medicion, MotorRiesgo

# (tipo, inicio_s, duracion_s)
ESCENARIOS: dict[str, dict] = {
    "normal": {"duracion": 60, "eventos": []},
    "cansancio": {"duracion": 100, "eventos":
                  [("parpadeo_lento", t, 0.7) for t in np.arange(15, 95, 2.0)]
                  + [("bostezo", 30, 3.0), ("bostezo", 50, 3.0), ("bostezo", 70, 3.0)]},
    "microsueno": {"duracion": 40, "eventos": [("cierre", 20, 2.5)]},
    "sin_respuesta": {"duracion": 50, "eventos": [("cierre", 20, 22.0)]},
    "distraccion": {"duracion": 40, "eventos": [("giro", 20, 5.0)]},
    "cabeceos": {"duracion": 60, "eventos": [("cabeceo", 20, 1.0), ("cabeceo", 30, 1.0), ("cabeceo", 40, 1.0)]},
    "rostro_ausente": {"duracion": 40, "eventos": [("ausente", 20, 6.0)]},
}


def generar_mediciones(escenario: str = "normal", fps: int = 15, semilla: int = 42,
                       ear_base: float = 0.30, parpadeo_cada_s: float = 4.0, ruido: float = 0.012,
                       dur_parpadeo: float = 0.15, dur_cierre: float | None = None) -> list[Medicion]:
    """Devuelve una lista de Medicion para el escenario indicado.

    ruido: desviación estándar del EAR (iluminación, cámara de baja calidad).
    dur_parpadeo: duración de cada parpadeo normal en segundos.
    dur_cierre: si se indica, reemplaza la duración de los cierres de ojos del escenario.
    """
    if escenario not in ESCENARIOS:
        raise ValueError(f"Escenario desconocido: {escenario}. Opciones: {list(ESCENARIOS)}")
    rng = np.random.default_rng(semilla)
    datos = ESCENARIOS[escenario]
    eventos = [(tipo, ini, dur_cierre if (tipo == "cierre" and dur_cierre) else dur)
               for tipo, ini, dur in datos["eventos"]]
    mediciones = []
    for i in range(int(datos["duracion"] * fps)):
        t = i / fps
        ear = ear_base + rng.normal(0, ruido)
        mar = abs(0.05 + rng.normal(0, 0.02))
        pitch, yaw, rostro = rng.normal(0, 1.0), rng.normal(0, 1.5), True
        # parpadeo normal
        if (t % parpadeo_cada_s) < dur_parpadeo:
            ear = 0.08 + rng.normal(0, 0.01)
        for tipo, ini, dur in eventos:
            if ini <= t < ini + dur:
                if tipo in ("cierre", "parpadeo_lento"):
                    ear = 0.09 + rng.normal(0, 0.01)
                elif tipo == "bostezo":
                    mar = 0.85 + rng.normal(0, 0.03)
                elif tipo == "cabeceo":
                    pitch = 25 + rng.normal(0, 1.0)
                elif tipo == "giro":
                    yaw = 45 + rng.normal(0, 2.0)
                elif tipo == "ausente":
                    rostro = False
        if rostro:
            mediciones.append(Medicion(t=t, rostro=True, ear=float(ear), mar=float(mar),
                                       pitch=float(pitch), yaw=float(yaw)))
        else:
            mediciones.append(Medicion(t=t, rostro=False))
    return mediciones


def ejecutar(mediciones: list[Medicion], config: ConfigRiesgo | None = None,
             reconocer_en: float | None = None) -> pd.DataFrame:
    """Pasa las mediciones por el motor y devuelve un DataFrame con los estados."""
    motor = MotorRiesgo(config)
    filas = []
    for m in mediciones:
        if reconocer_en is not None and abs(m.t - reconocer_en) < 1e-9:
            motor.reconocer(m.t)
        e = motor.actualizar(m)
        fila = e.a_dict()
        fila["motivos"] = ",".join(e.motivos)
        fila["ear_crudo"] = m.ear
        filas.append(fila)
    return pd.DataFrame(filas)
