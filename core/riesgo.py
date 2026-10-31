"""Motor de riesgo del conductor: reglas, calibración, histéresis y emergencia.

Es Python puro (no depende de la cámara ni de MediaPipe): recibe mediciones
con su marca de tiempo y devuelve el nivel de riesgo. Por eso se puede probar
con datos sintéticos en pytest y en el notebook 03.

Niveles: NORMAL < BAJO < MEDIO < ALTO

  ALTO   microsueño (ojos cerrados >= 1,5 s) o cabeza caída prolongada
  MEDIO  cansancio (PERCLOS > 30 %, bostezos o cabeceos repetidos) o
         distracción (cabeza girada / rostro ausente > 3 s)
  BAJO   expresión negativa aparente sostenida (no es una emoción real)

Protocolo "conductor que no responde": si el nivel es ALTO y el conductor
sigue con ojos cerrados, cabeza caída o sin rostro visible durante
`sin_respuesta_s` segundos, el estado marca `emergencia=True`. La interfaz
inicia entonces la cuenta regresiva antes de avisar a los contactos.
"""
from __future__ import annotations

from collections import deque
from dataclasses import asdict, dataclass, field
from statistics import median

from core.config import ConfigRiesgo
from core.landmarks import diferencia_angular

NIVELES = ["NORMAL", "BAJO", "MEDIO", "ALTO"]
ORDEN = {n: i for i, n in enumerate(NIVELES)}

DESCRIPCION_MOTIVOS = {
    "microsueno": "Microsueño: ojos cerrados de forma continua",
    "cabeza_caida": "Cabeza caída de forma prolongada",
    "cansancio_perclos": "Cansancio: ojos cerrados gran parte del último minuto (PERCLOS)",
    "bostezos": "Cansancio: bostezos frecuentes",
    "cabeceos": "Cansancio: cabeceos repetidos",
    "distraccion": "Distracción: cabeza girada",
    "rostro_ausente": "Rostro no visible para la cámara",
    "expresion_negativa": "Expresión negativa aparente sostenida",
}


@dataclass
class Medicion:
    """Una observación por fotograma. `t` en segundos (monótono)."""

    t: float
    rostro: bool
    ear: float | None = None
    mar: float | None = None
    pitch: float | None = None
    yaw: float | None = None
    expresion_negativa: bool | None = None   # None = sin nueva clasificación
    ojos_cerrados: bool | None = None        # opcional: decisión de la CNN


@dataclass
class EstadoRiesgo:
    """Resultado del motor para un instante."""

    t: float
    nivel: str = "NORMAL"
    motivos: list[str] = field(default_factory=list)
    calibrando: bool = True
    progreso_calibracion: float = 0.0
    rostro: bool = False
    ear: float | None = None
    ear_umbral: float = 0.21
    ear_base: float | None = None
    mar: float | None = None
    ojos_cerrados: bool = False
    tiempo_ojos_cerrados: float = 0.0
    perclos: float = 0.0
    bostezos: int = 0
    cabeceos: int = 0
    pitch_rel: float | None = None
    yaw_rel: float | None = None
    expresion_negativa_frac: float = 0.0
    tiempo_sin_respuesta: float = 0.0
    emergencia: bool = False
    silenciado: bool = False

    def a_dict(self) -> dict:
        datos = asdict(self)
        for k, v in datos.items():
            if isinstance(v, float):
                datos[k] = round(v, 4)
        return datos

    @property
    def motivo_principal(self) -> str | None:
        return self.motivos[0] if self.motivos else None


class MotorRiesgo:
    """Máquina de estados del riesgo del conductor."""

    def __init__(self, config: ConfigRiesgo | None = None):
        self.config = config or ConfigRiesgo()
        self.recalibrar()

    # ------------------------------------------------------------------ API
    def actualizar_config(self, config: ConfigRiesgo) -> None:
        """Aplica nuevos umbrales sin perder la calibración."""
        self.config = config
        self._ears = deque(self._ears, maxlen=max(1, int(config.suavizado_frames)))
        self._actualizar_umbral_ear()

    def _actualizar_umbral_ear(self) -> None:
        c = self.config
        self.ear_umbral = (c.ear_umbral_manual if c.ear_umbral_manual is not None else
                           c.factor_ear * self.ear_base if self.ear_base is not None else c.ear_umbral_defecto)

    def recalibrar(self) -> None:
        """Reinicia calibración, historial y alarmas."""
        c = self.config
        self.calibrado = c.calibracion_s <= 0
        self._t_inicio_calib: float | None = None
        self._calib_ear: list[float] = []
        self._calib_pitch: list[float] = []
        self._calib_yaw: list[float] = []
        self.ear_base: float | None = None
        self._actualizar_umbral_ear()
        self.pitch_base: float | None = None
        self.yaw_base: float | None = None
        self._ears: deque[float] = deque(maxlen=max(1, int(c.suavizado_frames)))
        self._limpiar_evidencia()
        self.nivel = "NORMAL"
        self._t_bajada: float | None = None
        self._t_sin_respuesta: float | None = None
        self.emergencia = False
        self._t_silencio_hasta = -1.0
        self.ultimo = EstadoRiesgo(t=0.0, calibrando=not self.calibrado)

    def reconocer(self, t: float) -> None:
        """El conductor (o un pasajero) pulsó "Estoy bien".

        Cancela la emergencia, silencia la alarma unos segundos y descarta la
        evidencia de cansancio acumulada; si el cansancio persiste, la
        evidencia se vuelve a acumular y la alerta reaparece.
        """
        self.emergencia = False
        self._t_sin_respuesta = None
        self._t_silencio_hasta = t + self.config.silencio_tras_ok_s
        self._limpiar_evidencia()
        self.nivel = "NORMAL"
        self._t_bajada = None

    def actualizar(self, m: Medicion) -> EstadoRiesgo:
        """Procesa una medición y devuelve el estado de riesgo."""
        c = self.config
        estado = EstadoRiesgo(t=m.t, rostro=m.rostro, ear_umbral=self.ear_umbral,
                              ear_base=self.ear_base, mar=m.mar)

        # ---------------- calibración
        if not self.calibrado:
            self._calibrar(m)
            estado.calibrando = not self.calibrado
            estado.progreso_calibracion = self._progreso_calibracion(m.t)
            estado.ear = m.ear
            estado.ear_umbral = self.ear_umbral
            if not self.calibrado:
                self.ultimo = estado
                return estado
        estado.calibrando = False
        estado.progreso_calibracion = 1.0
        estado.ear_base = self.ear_base

        if m.rostro and self.pitch_base is None and m.pitch is not None:
            self.pitch_base, self.yaw_base = m.pitch, m.yaw or 0.0

        # ---------------- ojos
        cerrados = False
        if m.rostro and m.ear is not None:
            self._ears.append(m.ear)
            estado.ear = sum(self._ears) / len(self._ears)
            cerrados = estado.ear < self.ear_umbral
        if m.rostro and m.ojos_cerrados is not None:
            cerrados = bool(m.ojos_cerrados)
        estado.ojos_cerrados = cerrados

        if cerrados:
            self._t_cierre = m.t if self._t_cierre is None else self._t_cierre
        else:
            self._t_cierre = None
        estado.tiempo_ojos_cerrados = 0.0 if self._t_cierre is None else m.t - self._t_cierre

        if m.rostro:
            self._perclos.append((m.t, cerrados))
        self._podar(self._perclos, m.t, c.perclos_ventana_s)
        perclos_valido = self._cobertura(self._perclos, m.t) >= 0.5 * c.perclos_ventana_s
        if self._perclos:
            estado.perclos = sum(1 for _, x in self._perclos if x) / len(self._perclos)

        # ---------------- boca (bostezos)
        if m.rostro and m.mar is not None and m.mar > c.mar_umbral:
            self._t_boca = m.t if self._t_boca is None else self._t_boca
            if not self._bostezo_contado and m.t - self._t_boca >= c.bostezo_min_s:
                self._bostezos.append(m.t)
                self._bostezo_contado = True
        else:
            self._t_boca, self._bostezo_contado = None, False
        self._podar_t(self._bostezos, m.t, c.bostezo_ventana_s)
        estado.bostezos = len(self._bostezos)

        # ---------------- cabeza (cabeceo y giro)
        caida = girada = False
        if m.rostro and m.pitch is not None and self.pitch_base is not None:
            estado.pitch_rel = diferencia_angular(m.pitch, self.pitch_base)
            caida = abs(estado.pitch_rel) > c.cabeceo_grados
        if m.rostro and m.yaw is not None and self.yaw_base is not None:
            estado.yaw_rel = diferencia_angular(m.yaw, self.yaw_base)
            girada = abs(estado.yaw_rel) > c.giro_grados

        if caida:
            self._t_caida = m.t if self._t_caida is None else self._t_caida
            if not self._cabeceo_contado and m.t - self._t_caida >= c.cabeceo_min_s:
                self._cabeceos.append(m.t)
                self._cabeceo_contado = True
        else:
            self._t_caida, self._cabeceo_contado = None, False
        self._podar_t(self._cabeceos, m.t, c.cabeceo_ventana_s)
        estado.cabeceos = len(self._cabeceos)
        t_caida = 0.0 if self._t_caida is None else m.t - self._t_caida

        # Distracción: cabeza girada o rostro ausente
        if (not m.rostro) or girada:
            self._t_distraccion = m.t if self._t_distraccion is None else self._t_distraccion
        else:
            self._t_distraccion = None
        t_distraccion = 0.0 if self._t_distraccion is None else m.t - self._t_distraccion

        # ---------------- expresiones
        if m.expresion_negativa is not None:
            self._expresiones.append((m.t, bool(m.expresion_negativa)))
        self._podar(self._expresiones, m.t, c.expresion_ventana_s)
        if self._expresiones:
            estado.expresion_negativa_frac = (
                sum(1 for _, x in self._expresiones if x) / len(self._expresiones))
        expresion_valida = (len(self._expresiones) >= 3 and
                            self._cobertura(self._expresiones, m.t) >= 0.5 * c.expresion_ventana_s)

        # ---------------- reglas
        motivos_alto, motivos_medio, motivos_bajo = [], [], []
        if estado.tiempo_ojos_cerrados >= c.microsueno_s:
            motivos_alto.append("microsueno")
        if t_caida >= c.cabeza_caida_s:
            motivos_alto.append("cabeza_caida")
        if perclos_valido and estado.perclos > c.perclos_umbral:
            motivos_medio.append("cansancio_perclos")
        if estado.bostezos >= c.bostezos_limite:
            motivos_medio.append("bostezos")
        if estado.cabeceos >= c.cabeceos_limite:
            motivos_medio.append("cabeceos")
        if t_distraccion >= c.distraccion_s:
            motivos_medio.append("rostro_ausente" if not m.rostro else "distraccion")
        if expresion_valida and estado.expresion_negativa_frac > c.expresion_umbral:
            motivos_bajo.append("expresion_negativa")

        if motivos_alto:
            nivel_crudo = "ALTO"
        elif motivos_medio:
            nivel_crudo = "MEDIO"
        elif motivos_bajo:
            nivel_crudo = "BAJO"
        else:
            nivel_crudo = "NORMAL"
        estado.motivos = motivos_alto + motivos_medio + motivos_bajo

        # ---------------- histéresis: subir de inmediato, bajar con retardo
        if ORDEN[nivel_crudo] >= ORDEN[self.nivel]:
            self.nivel = nivel_crudo
            self._t_bajada = None
        else:
            self._t_bajada = m.t if self._t_bajada is None else self._t_bajada
            if m.t - self._t_bajada >= c.histeresis_s:
                self.nivel = nivel_crudo
                self._t_bajada = None
        estado.nivel = self.nivel
        if not estado.motivos and self.nivel != "NORMAL":
            estado.motivos = list(self.ultimo.motivos)

        # ---------------- silencio tras "Estoy bien"
        estado.silenciado = m.t < self._t_silencio_hasta

        # ---------------- conductor que no responde
        sin_respuesta = (not m.rostro) or cerrados or caida
        if self.nivel == "ALTO" and sin_respuesta and not estado.silenciado:
            self._t_sin_respuesta = m.t if self._t_sin_respuesta is None else self._t_sin_respuesta
            estado.tiempo_sin_respuesta = m.t - self._t_sin_respuesta
            if estado.tiempo_sin_respuesta >= c.sin_respuesta_s:
                self.emergencia = True
        else:
            self._t_sin_respuesta = None
        estado.emergencia = self.emergencia

        self.ultimo = estado
        return estado

    # ------------------------------------------------------------ internos
    def _limpiar_evidencia(self) -> None:
        self._t_cierre: float | None = None
        self._perclos: deque[tuple[float, bool]] = deque()
        self._t_boca: float | None = None
        self._bostezo_contado = False
        self._bostezos: deque[float] = deque()
        self._t_caida: float | None = None
        self._cabeceo_contado = False
        self._cabeceos: deque[float] = deque()
        self._t_distraccion: float | None = None
        self._expresiones: deque[tuple[float, bool]] = deque()

    def _calibrar(self, m: Medicion) -> None:
        if not m.rostro or m.ear is None:
            return
        if self._t_inicio_calib is None:
            self._t_inicio_calib = m.t
        self._calib_ear.append(m.ear)
        if m.pitch is not None:
            self._calib_pitch.append(m.pitch)
        if m.yaw is not None:
            self._calib_yaw.append(m.yaw)
        if m.t - self._t_inicio_calib >= self.config.calibracion_s and len(self._calib_ear) >= 5:
            base = median(self._calib_ear)
            # Si el EAR base es muy bajo, el conductor parpadeó o cerró los
            # ojos durante la calibración: se usa el umbral por defecto.
            if base >= 0.15:
                self.ear_base = base
                self._actualizar_umbral_ear()
            self.pitch_base = median(self._calib_pitch) if self._calib_pitch else None
            self.yaw_base = median(self._calib_yaw) if self._calib_yaw else None
            self.calibrado = True

    def _progreso_calibracion(self, t: float) -> float:
        if self.calibrado:
            return 1.0
        if self._t_inicio_calib is None or self.config.calibracion_s <= 0:
            return 0.0
        return min(1.0, (t - self._t_inicio_calib) / self.config.calibracion_s)

    @staticmethod
    def _podar(cola: deque, t: float, ventana: float) -> None:
        while cola and t - cola[0][0] > ventana:
            cola.popleft()

    @staticmethod
    def _podar_t(cola: deque, t: float, ventana: float) -> None:
        while cola and t - cola[0] > ventana:
            cola.popleft()

    @staticmethod
    def _cobertura(cola: deque, t: float) -> float:
        return (t - cola[0][0]) if cola else 0.0
