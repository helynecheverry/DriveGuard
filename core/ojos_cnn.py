"""CNN opcional para clasificar ojos abiertos / cerrados.

Se entrena en notebooks/04_CNN_Ojos_Abiertos_Cerrados.ipynb (MRL Eye Dataset
o capturas propias) y se guarda en models/cnn_ojos.pt. Si OJOS_METODO=cnn y el
archivo existe, el pipeline usa esta red en lugar del umbral de EAR.

Clases: 0 = cerrado, 1 = abierto (misma convención que el MRL Eye Dataset).
Entrada: recorte en escala de grises de 64x64 píxeles.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from core.config import CNN_OJOS_PATH

TAM = 64
CLASES = ["cerrado", "abierto"]


def crear_cnn():
    """Arquitectura pequeña (~25 mil parámetros): rápida en CPU."""
    from torch import nn

    def bloque(c_in, c_out):
        return nn.Sequential(nn.Conv2d(c_in, c_out, 3, padding=1), nn.BatchNorm2d(c_out),
                             nn.ReLU(inplace=True), nn.MaxPool2d(2))

    return nn.Sequential(bloque(1, 16), bloque(16, 32), bloque(32, 64),
                         nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Dropout(0.3), nn.Linear(64, 2))


def preparar(recortes: list[np.ndarray]):
    """Convierte recortes grises (64x64 uint8) en un tensor normalizado N×1×64×64."""
    import torch
    arr = np.stack([r.astype(np.float32) / 255.0 for r in recortes])[:, None, :, :]
    return torch.from_numpy((arr - 0.5) / 0.5)


class ClasificadorOjos:
    """Carga la CNN entrenada y estima la probabilidad de ojo cerrado."""

    def __init__(self, ruta: Path | str = CNN_OJOS_PATH):
        import torch
        ruta = Path(ruta)
        if not ruta.exists():
            raise FileNotFoundError(f"No existe {ruta}. Entrene la CNN con el notebook 04.")
        self._torch = torch
        self.modelo = crear_cnn()
        estado = torch.load(ruta, map_location="cpu")
        self.modelo.load_state_dict(estado["modelo"] if "modelo" in estado else estado)
        self.modelo.eval()

    def prob_cerrado(self, recortes: list[np.ndarray]) -> np.ndarray:
        if not recortes:
            return np.array([])
        with self._torch.no_grad():
            logits = self.modelo(preparar(recortes))
            return self._torch.softmax(logits, dim=1)[:, 0].numpy()
