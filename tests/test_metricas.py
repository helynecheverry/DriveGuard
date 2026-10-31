"""Pruebas de las métricas geométricas (EAR, MAR, ángulos)."""
import math

import numpy as np
import pytest

from core.landmarks import (OJO_DERECHO, angulos_desde_matriz, calcular_ear, calcular_mar,
                            diferencia_angular, metricas_desde_puntos)


def ojo(apertura: float, ancho: float = 30.0) -> np.ndarray:
    """Ojo sintético: p1 y p4 en los extremos, párpados a +/- apertura/2."""
    a = apertura / 2
    return np.array([[0, 0], [10, -a], [20, -a], [ancho, 0], [20, a], [10, a]], float)


def test_ear_ojo_abierto_y_cerrado():
    assert calcular_ear(ojo(9)) == pytest.approx(9 / 30)
    assert calcular_ear(ojo(1)) == pytest.approx(1 / 30)
    assert calcular_ear(ojo(9)) > calcular_ear(ojo(1))


def test_ear_es_invariante_a_la_escala():
    assert calcular_ear(ojo(9) * 3.5) == pytest.approx(calcular_ear(ojo(9)))


def test_ear_requiere_seis_puntos():
    with pytest.raises(ValueError):
        calcular_ear(np.zeros((5, 2)))


def test_ear_ancho_cero_no_divide_por_cero():
    assert calcular_ear(np.zeros((6, 2))) == 0.0


def test_mar_boca_abierta():
    comisuras = np.array([[0, 0], [40, 0]], float)
    verticales = [(np.array([10, -12.0]), np.array([10, 12.0]))] * 3
    assert calcular_mar(comisuras, verticales) == pytest.approx(24 / 40)


def test_metricas_desde_malla_completa():
    puntos = np.zeros((478, 3), float)
    puntos[OJO_DERECHO, :2] = ojo(9) + [100, 100]
    puntos[[362, 385, 387, 263, 373, 380], :2] = ojo(6) + [200, 100]
    puntos[78, :2], puntos[308, :2] = [140, 200], [180, 200]
    for a, b in [(81, 178), (13, 14), (311, 402)]:
        puntos[a, :2], puntos[b, :2] = [160, 195], [160, 205]
    m = metricas_desde_puntos(puntos)
    assert m["ear_der"] == pytest.approx(0.30)
    assert m["ear_izq"] == pytest.approx(0.20)
    assert m["ear"] == pytest.approx(0.25)
    assert m["mar"] == pytest.approx(10 / 40)


def test_angulos_matriz_identidad_y_rotacion():
    assert angulos_desde_matriz(np.eye(4)) == pytest.approx((0, 0, 0))
    th = math.radians(20)
    rx = np.eye(4)
    rx[1:3, 1:3] = [[math.cos(th), -math.sin(th)], [math.sin(th), math.cos(th)]]
    pitch, yaw, roll = angulos_desde_matriz(rx)
    assert pitch == pytest.approx(20, abs=1e-6)
    assert yaw == pytest.approx(0, abs=1e-6)


def test_diferencia_angular_normalizada():
    assert diferencia_angular(179, -179) == pytest.approx(-2)
    assert diferencia_angular(10, 350) == pytest.approx(20)
