"""Alarma sonora local del conductor.

Por qué el sonido es local: en streamlit-webrtc cada fotograma se procesa en
un hilo aparte del de Streamlit, y desde ese hilo no se puede usar st.audio
ni modificar la página. Como la aplicación se ejecuta en el mismo portátil
que tiene la cámara, el sonido se reproduce directamente en ese equipo con
pygame (o winsound en Windows). La página solo muestra el aviso visual.

Comportamiento:
- ALTO: sirena continua hasta que baja el nivel o se pulsa "Estoy bien".
- MEDIO: aviso de dos tonos, repetido cada `intervalo_medio_s` segundos.
- BAJO / NORMAL: sin sonido (solo indicador visual).
"""
from __future__ import annotations

import logging
import math
import os
import struct
import sys
import threading
import time
import wave
from pathlib import Path

from core.config import ALARMA_BACKEND, ASSETS_DIR

os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")
log = logging.getLogger("driveguard.alarma")

SONIDOS = {"media": ASSETS_DIR / "alarma_media.wav", "alta": ASSETS_DIR / "alarma_alta.wav"}


# --------------------------------------------------------------------------
# Generación de los archivos de sonido (sin dependencias externas)
# --------------------------------------------------------------------------
def _escribir_wav(ruta: Path, muestras: list[float], fs: int = 44100) -> None:
    ruta.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(ruta), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(fs)
        w.writeframes(b"".join(struct.pack("<h", int(max(-1, min(1, s)) * 32000))
                               for s in muestras))


def _tono(freq: float, dur: float, fs: int = 44100, vol: float = 0.8) -> list[float]:
    n = int(dur * fs)
    rampa = int(0.01 * fs)  # 10 ms de rampa para evitar chasquidos
    out = []
    for i in range(n):
        env = min(1.0, i / rampa, (n - i) / rampa)
        out.append(vol * env * math.sin(2 * math.pi * freq * i / fs))
    return out


def generar_sonidos(forzar: bool = False) -> dict[str, Path]:
    """Crea assets/alarma_media.wav y assets/alarma_alta.wav si no existen."""
    if forzar or not SONIDOS["media"].exists():
        _escribir_wav(SONIDOS["media"], _tono(880, 0.25) + _tono(0, 0.08) + _tono(660, 0.35))
    if forzar or not SONIDOS["alta"].exists():
        muestras: list[float] = []
        for _ in range(4):
            muestras += _tono(1400, 0.16, vol=0.95) + _tono(950, 0.16, vol=0.95)
        _escribir_wav(SONIDOS["alta"], muestras)
    return SONIDOS


# --------------------------------------------------------------------------
# Reproductores
# --------------------------------------------------------------------------
class _ReproductorPygame:
    nombre = "pygame"

    def __init__(self, volumen: float):
        import pygame
        pygame.mixer.init()
        self._pg = pygame
        self._sonidos = {k: pygame.mixer.Sound(str(v)) for k, v in SONIDOS.items()}
        self.volumen(volumen)

    def volumen(self, v: float) -> None:
        for s in self._sonidos.values():
            s.set_volume(max(0.0, min(1.0, v)))

    def reproducir(self, clave: str, bucle: bool) -> None:
        self._sonidos[clave].play(loops=-1 if bucle else 0)

    def detener(self) -> None:
        self._pg.mixer.stop()


class _ReproductorWinsound:
    nombre = "winsound"

    def __init__(self, volumen: float):
        import winsound  # solo existe en Windows
        self._ws = winsound

    def volumen(self, v: float) -> None:  # winsound no controla el volumen
        pass

    def reproducir(self, clave: str, bucle: bool) -> None:
        flags = self._ws.SND_FILENAME | self._ws.SND_ASYNC
        if bucle:
            flags |= self._ws.SND_LOOP
        self._ws.PlaySound(str(SONIDOS[clave]), flags)

    def detener(self) -> None:
        self._ws.PlaySound(None, self._ws.SND_PURGE)


class _ReproductorSilencioso:
    """Respaldo sin audio: registra en consola (útil en servidores o pruebas)."""

    nombre = "ninguno"

    def __init__(self, volumen: float):
        self.historial: list[str] = []

    def volumen(self, v: float) -> None:
        pass

    def reproducir(self, clave: str, bucle: bool) -> None:
        self.historial.append(clave)
        log.warning("ALARMA %s (sin dispositivo de audio)", clave.upper())

    def detener(self) -> None:
        pass


def crear_reproductor(backend: str = ALARMA_BACKEND, volumen: float = 0.8):
    """Elige el reproductor disponible: pygame -> winsound -> silencioso."""
    generar_sonidos()
    if backend == "auto":
        orden = ["pygame", "winsound"] if sys.platform == "win32" else ["pygame"]
    elif backend == "ninguno":
        orden = []
    else:
        orden = [backend]
    for nombre in orden:
        try:
            clase = _ReproductorPygame if nombre == "pygame" else _ReproductorWinsound
            return clase(volumen)
        except Exception as exc:  # sin audio, librería ausente, etc.
            log.info("Reproductor %s no disponible: %s", nombre, exc)
    return _ReproductorSilencioso(volumen)


# --------------------------------------------------------------------------
# Alarma con hilo propio
# --------------------------------------------------------------------------
class Alarma:
    """Controla el sonido según el nivel de riesgo, sin bloquear el video."""

    def __init__(self, backend: str = ALARMA_BACKEND, volumen: float = 0.8,
                 intervalo_medio_s: float = 6.0, activa: bool = True):
        self.reproductor = crear_reproductor(backend, volumen)
        self.intervalo_medio_s = intervalo_medio_s
        self.activa = activa
        self._nivel = "NORMAL"
        self._silenciado = False
        self._sonando: str | None = None
        self._ultimo_medio = 0.0
        self._lock = threading.Lock()
        self._fin = threading.Event()
        self._hilo = threading.Thread(target=self._bucle, name="alarma", daemon=True)
        self._hilo.start()

    @property
    def backend(self) -> str:
        return self.reproductor.nombre

    def actualizar(self, nivel: str, silenciado: bool = False) -> None:
        """Llamar en cada fotograma con el nivel actual (operación barata)."""
        with self._lock:
            self._nivel, self._silenciado = nivel, silenciado

    def configurar(self, volumen: float | None = None, activa: bool | None = None,
                   intervalo_medio_s: float | None = None) -> None:
        if volumen is not None:
            self.reproductor.volumen(volumen)
        if activa is not None:
            self.activa = activa
        if intervalo_medio_s is not None:
            self.intervalo_medio_s = intervalo_medio_s

    def probar(self, clave: str = "alta") -> None:
        """Reproduce una vez el sonido (botón de prueba en Configuración)."""
        self.reproductor.reproducir(clave, bucle=False)

    def detener(self) -> None:
        """Detiene el hilo y el sonido (al terminar la sesión de cámara)."""
        self._fin.set()
        self.reproductor.detener()

    def _bucle(self) -> None:
        while not self._fin.is_set():
            with self._lock:
                nivel, silenciado = self._nivel, self._silenciado
            objetivo = None
            if self.activa and not silenciado:
                objetivo = {"ALTO": "alta", "MEDIO": "media"}.get(nivel)
            try:
                if objetivo == "alta":
                    if self._sonando != "alta":
                        self.reproductor.detener()
                        self.reproductor.reproducir("alta", bucle=True)
                        self._sonando = "alta"
                else:
                    if self._sonando == "alta":
                        self.reproductor.detener()
                        self._sonando = None
                    ahora = time.monotonic()
                    if objetivo == "media" and ahora - self._ultimo_medio >= self.intervalo_medio_s:
                        self.reproductor.reproducir("media", bucle=False)
                        self._ultimo_medio = ahora
            except Exception as exc:
                log.error("Error reproduciendo alarma: %s", exc)
            self._fin.wait(0.1)
