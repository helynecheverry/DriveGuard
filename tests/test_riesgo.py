"""Pruebas del motor de riesgo con escenarios sintéticos."""
import pytest

from core.config import ConfigRiesgo
from core.riesgo import Medicion, MotorRiesgo
from core.simulacion import ejecutar, generar_mediciones


def resumen(escenario, **kw):
    df = ejecutar(generar_mediciones(escenario), **kw)
    return df, df[~df.calibrando]


def test_conduccion_normal_no_genera_alertas():
    _, d = resumen("normal")
    assert set(d.nivel) == {"NORMAL"}
    assert not d.emergencia.any()


def test_calibracion_ajusta_umbral_al_conductor():
    df, _ = resumen("normal")
    assert df.calibrando.iloc[0]
    assert df.ear_base.iloc[-1] == pytest.approx(0.30, abs=0.01)
    assert df.ear_umbral.iloc[-1] == pytest.approx(0.75 * df.ear_base.iloc[-1], abs=1e-3)


def test_calibracion_con_ojos_cerrados_usa_umbral_defecto():
    motor = MotorRiesgo(ConfigRiesgo(calibracion_s=2))
    for i in range(60):
        motor.actualizar(Medicion(t=i / 15, rostro=True, ear=0.08, mar=0.0, pitch=0, yaw=0))
    assert motor.calibrado
    assert motor.ear_umbral == pytest.approx(ConfigRiesgo().ear_umbral_defecto)


def test_umbral_manual_se_conserva_al_calibrar_y_cambia_la_deteccion():
    cfg = ConfigRiesgo(calibracion_s=1, suavizado_frames=1, ear_umbral_manual=0.18)
    motor = MotorRiesgo(cfg)
    for i in range(20):
        motor.actualizar(Medicion(t=i / 10, rostro=True, ear=0.30))
    assert motor.ear_base == pytest.approx(0.30)
    e = motor.actualizar(Medicion(t=2, rostro=True, ear=0.20))
    assert not e.ojos_cerrados
    assert e.ear_umbral == pytest.approx(0.18)
    motor.actualizar_config(ConfigRiesgo(calibracion_s=1, suavizado_frames=1, factor_ear=0.75))
    e = motor.actualizar(Medicion(t=2.1, rostro=True, ear=0.20))
    assert e.ojos_cerrados
    assert e.ear_umbral == pytest.approx(0.225)


def test_umbral_se_actualiza_sin_base_y_se_conserva_al_recalibrar():
    motor = MotorRiesgo(ConfigRiesgo(calibracion_s=0, ear_umbral_manual=0.25))
    assert motor.actualizar(Medicion(t=0, rostro=True, ear=0.23)).ojos_cerrados
    motor.recalibrar()
    assert motor.ear_umbral == pytest.approx(0.25)
    motor.actualizar_config(ConfigRiesgo(calibracion_s=0, ear_umbral_defecto=0.19))
    assert motor.ear_umbral == pytest.approx(0.19)


def test_microsueno_es_riesgo_alto():
    _, d = resumen("microsueno")
    alto = d[d.nivel == "ALTO"]
    assert not alto.empty
    assert "microsueno" in alto.motivos.iloc[0]
    # se activa después de 1,5 s de ojos cerrados (cierre en t=20 s)
    assert alto.t.iloc[0] == pytest.approx(21.5, abs=0.2)
    assert not d.emergencia.any()


def test_parpadeo_normal_no_es_microsueno():
    _, d = resumen("normal")
    assert d.tiempo_ojos_cerrados.max() < 0.5


def test_cansancio_por_perclos_y_bostezos():
    _, d = resumen("cansancio")
    assert d.perclos.max() > 0.30
    assert d.bostezos.max() == 3
    motivos = ",".join(d.motivos)
    assert "cansancio_perclos" in motivos and "bostezos" in motivos
    assert "MEDIO" in set(d.nivel)


def test_cabeceos_repetidos():
    _, d = resumen("cabeceos")
    assert d.cabeceos.max() == 3
    assert "cabeceos" in ",".join(d.motivos)


def test_distraccion_y_rostro_ausente_son_medio():
    for esc, motivo in [("distraccion", "distraccion"), ("rostro_ausente", "rostro_ausente")]:
        _, d = resumen(esc)
        assert d.nivel.map({"NORMAL": 0, "BAJO": 1, "MEDIO": 2, "ALTO": 3}).max() == 2
        assert motivo in ",".join(d.motivos)


def test_conductor_que_no_responde_activa_emergencia():
    _, d = resumen("sin_respuesta")
    emerg = d[d.emergencia]
    assert not emerg.empty
    # ALTO en t≈21,5 s + 10 s sin respuesta
    assert emerg.t.iloc[0] == pytest.approx(31.5, abs=0.3)


def test_estoy_bien_cancela_emergencia_y_silencia():
    df = ejecutar(generar_mediciones("sin_respuesta"), reconocer_en=25.0)
    despues = df[(df.t >= 25.0) & (df.t < 30.0)]
    assert despues.silenciado.all()
    assert not df[df.t < 30.0].emergencia.any()


def test_histeresis_retrasa_la_bajada_de_nivel():
    motor = MotorRiesgo(ConfigRiesgo(calibracion_s=0, histeresis_s=2.0))
    t = 0.0
    for _ in range(40):                         # 2,6 s con ojos cerrados -> ALTO
        motor.actualizar(Medicion(t=t, rostro=True, ear=0.05, mar=0, pitch=0, yaw=0))
        t += 1 / 15
    assert motor.nivel == "ALTO"
    e = motor.actualizar(Medicion(t=t, rostro=True, ear=0.30, mar=0, pitch=0, yaw=0))
    assert e.nivel == "ALTO"                     # aún no baja
    for _ in range(40):
        t += 1 / 15
        e = motor.actualizar(Medicion(t=t, rostro=True, ear=0.30, mar=0, pitch=0, yaw=0))
    assert e.nivel == "NORMAL"


def test_expresion_negativa_sostenida_es_bajo():
    motor = MotorRiesgo(ConfigRiesgo(calibracion_s=0, expresion_ventana_s=10))
    e = None
    for i in range(15 * 12):
        t = i / 15
        neg = True if i % 30 == 0 else None     # una clasificación cada 2 s
        e = motor.actualizar(Medicion(t=t, rostro=True, ear=0.3, mar=0, pitch=0, yaw=0,
                                      expresion_negativa=neg))
    assert e.nivel == "BAJO"
    assert e.motivos == ["expresion_negativa"]


def test_cnn_puede_decidir_ojos_cerrados():
    motor = MotorRiesgo(ConfigRiesgo(calibracion_s=0))
    for i in range(40):
        e = motor.actualizar(Medicion(t=i / 15, rostro=True, ear=0.30, ojos_cerrados=True))
    assert e.nivel == "ALTO"


def test_config_desde_dict_ignora_claves_desconocidas():
    c = ConfigRiesgo.desde_dict({"microsueno_s": 2.0, "no_existe": 1})
    assert c.microsueno_s == 2.0
