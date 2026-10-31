"""Pruebas del pipeline con un detector simulado (no requiere cámara ni modelo)."""
import numpy as np

from core.alarma import Alarma
from core.config import ConfigRiesgo, DATA_DIR
from core.landmarks import OJO_DERECHO, OJO_IZQUIERDO, ResultadoRostro
from core.pipeline import ProcesadorConductor, dibujar
from core.riesgo import EstadoRiesgo


class DetectorFalso:
    """Devuelve un rostro con el EAR indicado en `self.ear` (None = sin rostro)."""

    def __init__(self):
        self.ear = 0.30

    def procesar(self, frame, t_ms=None):
        if self.ear is None:
            return None
        puntos = np.random.default_rng(0).uniform(100, 300, (478, 3)).astype(np.float32)
        return ResultadoRostro(puntos=puntos, ear=self.ear, ear_izq=self.ear, ear_der=self.ear,
                               mar=0.05, pitch=0.0, yaw=0.0, bbox=(100, 100, 200, 200))

    def cerrar(self):
        pass


def test_carpeta_data_se_crea_automaticamente():
    assert DATA_DIR.exists() and (DATA_DIR / "capturas").exists()


def test_pipeline_genera_eventos_y_alarma():
    alarma = Alarma(backend="ninguno")
    det = DetectorFalso()
    proc = ProcesadorConductor(ConfigRiesgo(calibracion_s=1), expresiones=False, alarma=alarma,
                               detector=det, intervalo_muestra_s=1)
    frame = np.zeros((480, 640, 3), np.uint8)
    t = 0.0
    for _ in range(45):                  # 3 s ojos abiertos (incluye calibración)
        proc.procesar(frame, t)
        t += 1 / 15
    det.ear = 0.05
    for _ in range(40):                  # 2,7 s ojos cerrados -> ALTO
        out, e = proc.procesar(frame, t)
        t += 1 / 15
    assert out.shape == frame.shape
    assert e.nivel == "ALTO"
    eventos = proc.extraer_eventos()
    tipos = [ev["tipo"] for ev in eventos]
    assert "microsueno" in tipos and "muestra" in tipos
    assert proc.extraer_eventos() == []          # se vacía al extraer
    snap = proc.snapshot()
    assert snap["nivel"] == "ALTO" and snap["alarma_backend"] == "ninguno"
    import time
    time.sleep(0.3)
    assert "alta" in alarma.reproductor.historial
    proc.reconocer()
    assert proc.snapshot()["silenciado"] is False or proc.motor.nivel == "NORMAL"
    proc.cerrar()


def test_dibujar_sin_rostro_y_en_emergencia():
    frame = np.zeros((480, 640, 3), np.uint8)
    e = EstadoRiesgo(t=1.0, nivel="ALTO", motivos=["microsueno"], calibrando=False, emergencia=True)
    out = dibujar(frame, None, e, fps=15)
    assert out.shape == frame.shape and out.any()
    assert not frame.any()     # no modifica el original


def test_recorte_ojo():
    from core.landmarks import recorte_ojo
    frame = np.full((480, 640, 3), 128, np.uint8)
    puntos = np.zeros((478, 3), np.float32)
    puntos[OJO_DERECHO, :2] = [[200, 200], [210, 195], [220, 195], [240, 200], [220, 205], [210, 205]]
    puntos[OJO_IZQUIERDO, :2] = puntos[OJO_DERECHO, :2] + [100, 0]
    r = recorte_ojo(frame, puntos, OJO_DERECHO)
    assert r.shape == (64, 64)


def test_puntos_ear_snapshot_no_conserva_ojos_si_se_pierde_rostro():
    det = DetectorFalso()
    proc = ProcesadorConductor(expresiones=False, alarma=False, detector=det)
    frame = np.zeros((480, 640, 3), np.uint8)
    try:
        proc.procesar(frame, 0)
        snap = proc.snapshot()
        assert len(snap["ojos"]["derecho"]["puntos"]) == 6
        assert snap["ojos"]["izquierdo"]["ear"] == 0.30
        snap["ojos"]["derecho"]["puntos"][0][0] = -999
        assert proc.snapshot()["ojos"]["derecho"]["puntos"][0][0] != -999
        det.ear = None
        proc.procesar(frame, 1)
        assert proc.snapshot()["ojos"] == {}
    finally:
        proc.cerrar()
