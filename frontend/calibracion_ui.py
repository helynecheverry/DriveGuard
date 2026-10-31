"""Calibrador EAR: geometría editable y lectura de puntos reales en vivo."""
from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st

from core.landmarks import calcular_ear

PUNTOS_EJEMPLO = [[20, 60], [60, 36], [100, 36], [140, 60], [100, 84], [60, 84]]


def ojo_svg(puntos, titulo: str = "Ojo · 6 puntos") -> str:
    """Normaliza usando una única escala para conservar las proporciones del EAR."""
    pts = np.asarray(puntos, dtype=float)[:, :2]
    escala = 230 / max(float(np.ptp(pts, axis=0).max()), 1)
    pts = (pts - pts.mean(axis=0)) * escala + [170, 105]
    contorno = " ".join(f"{x:.2f},{y:.2f}" for x, y in pts)
    lineas = "".join(f'<line x1="{pts[a,0]}" y1="{pts[a,1]}" x2="{pts[b,0]}" y2="{pts[b,1]}" '
                     'stroke="#38bdf8" stroke-width="2" stroke-dasharray="5 4"/>'
                     for a, b in [(0, 3), (1, 5), (2, 4)])
    nodos = "".join(f'<circle cx="{x}" cy="{y}" r="5" fill="#5eead4"/>'
                    f'<text x="{x}" y="{y + (-14 if i < 4 else 24)}" text-anchor="middle" '
                    f'fill="#e2e8f0" font-size="13">P{i+1}</text>'
                    for i, (x, y) in enumerate(pts))
    return (f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 340 210" role="img" '
            f'aria-label="{titulo}" style="width:100%;border-radius:16px;background:#101e32">'
            f'<title>{titulo}</title><polygon points="{contorno}" fill="#173a49" '
            f'stroke="#5eead4" stroke-width="2"/>{lineas}{nodos}</svg>')


def panel_ojos(snap: dict) -> None:
    st.markdown("#### Calibrador de ojos")
    ojos = snap.get("ojos") or {}
    if not ojos:
        st.info("Buscando ojos. Centre el rostro, mejore la iluminación y mire a la cámara.")
        return
    for col, (nombre, ojo) in zip(st.columns(2), ojos.items()):
        with col:
            st.markdown(ojo_svg(ojo["puntos"]), unsafe_allow_html=True)
            st.caption(f"Ojo {nombre} · EAR **{ojo['ear']:.3f}**")
    base = snap.get("ear_base")
    a, b = st.columns(2)
    a.metric("EAR base personal", f"{base:.3f}" if base is not None else "Pendiente")
    b.metric("Umbral de cierre", f"{snap['ear_umbral']:.3f}")
    if not snap.get("calibrando"):
        if st.session_state.config.get("ear_umbral_manual") is not None:
            st.caption("Umbral manual activo. Puedes ajustarlo en Ojos y calibración.")
        elif base is not None:
            st.caption("Calibración completada. El umbral se calcula a partir de tu EAR base.")
        else:
            st.caption("Calibración automática desactivada. Se usa el umbral por defecto.")


def puntos_apertura(ear: float) -> np.ndarray:
    """Ojo de ancho 120 cuya relación de apertura coincide con el EAR solicitado."""
    apertura = ear * 120 / 2
    return np.array([[20, 60], [60, 60-apertura], [100, 60-apertura],
                     [140, 60], [100, 60+apertura], [60, 60+apertura]])


def render() -> None:
    cfg = st.session_state.config
    st.subheader("Ojos y calibración")
    st.caption("Ajusta la detección a tu mirada y comprueba cómo responde a distintas aperturas.")
    if st.session_state.pop("aviso_ojos_aplicados", False):
        st.success("Cambios aplicados. El umbral se actualiza en la cámara activa; para repetir la medición base, pulsa Recalibrar.")
    izquierda, derecha = st.columns([1.1, 1], gap="large")
    with izquierda, st.container(border=True, key="controles_ojos"):
        st.markdown("##### 01 / SENSIBILIDAD DE DETECCIÓN")
        manual_actual = cfg.get("ear_umbral_manual")
        modo = st.radio("¿Cómo definir el cierre del ojo?", ["Automático", "Manual"],
                        index=1 if manual_actual is not None else 0, horizontal=True,
                        help="Automático: EAR base × factor. Manual: valor fijo que tú defines.")
        manual = modo == "Manual"
        if manual:
            umbral = st.number_input("Umbral de ojo cerrado (EAR)", 0.05, 0.45,
                                     float(manual_actual if manual_actual is not None else cfg["ear_umbral_defecto"]),
                                     0.005, format="%.3f")
            st.caption("EAR menor que este valor = ojo cerrado. Un umbral mayor detecta cierres con menos reducción de apertura.")
            factor = cfg["factor_ear"]
        else:
            factor = st.slider("Sensibilidad respecto a tu apertura habitual", 0.50, 0.95,
                               float(cfg["factor_ear"]), 0.01, format="%.2f")
            st.caption(f"El cierre se detectará por debajo del {factor * 100:.0f} % de tu EAR base. La cámara medirá esa base al calibrar.")
            umbral = cfg["ear_umbral_defecto"]
        segundos = st.number_input("Duración de calibración (s)", 0, 30,
                                    max(0, min(30, int(cfg["calibracion_s"]))),
                                    help="0 desactiva la calibración inicial. En modo automático se usa el umbral de respaldo.")
        with st.expander("Ajustes de respaldo y alerta"):
            respaldo = st.number_input("Umbral EAR de respaldo", 0.05, 0.45,
                                       float(cfg["ear_umbral_defecto"]), 0.005, format="%.3f")
            microsueno = st.slider("Tiempo de ojos cerrados para alertar (s)", 0.5, 5.0,
                                   float(cfg["microsueno_s"]), 0.1)
            st.caption("El respaldo se usa si no se obtiene una calibración válida. Un parpadeo breve no genera una alerta de microsueño.")
        if st.button("Aplicar a la detección", type="primary", width="stretch"):
            st.session_state.config = {**cfg, "ear_umbral_manual": float(umbral) if manual else None,
                                       "factor_ear": float(factor), "calibracion_s": float(segundos),
                                       "ear_umbral_defecto": float(respaldo), "microsueno_s": float(microsueno)}
            st.session_state.aviso_ojos_aplicados = True
            st.rerun()
        st.caption("Estos ajustes se conservan durante la sesión de interfaz; no se guardan como perfil permanente del conductor.")

    with derecha, st.container(border=True, key="prueba_apertura"):
        st.markdown("##### 02 / PRUEBA DE APERTURA")
        st.caption("Desliza para abrir o cerrar el ojo de prueba. No modifica tu medición real.")
        ear_prueba = st.slider("Apertura del ojo de prueba (EAR)", 0.0, 0.45, 0.30, 0.005, format="%.3f")
        st.markdown(ojo_svg(puntos_apertura(ear_prueba)), unsafe_allow_html=True)
        if manual:
            referencia = umbral
            st.caption("Comparación con el umbral manual que estás editando. Pulsa Aplicar para usarlo en la detección.")
        else:
            base = st.number_input("EAR base de ejemplo", 0.15, 0.50, 0.30, 0.01, format="%.3f")
            referencia = base * factor
            st.caption("Comparación ilustrativa: la cámara obtendrá tu EAR base real al calibrar.")
        a, b = st.columns(2)
        a.metric("EAR de prueba", f"{ear_prueba:.3f}")
        b.metric("Umbral de prueba", f"{referencia:.3f}")
        if ear_prueba < referencia:
            st.warning("OJO CERRADO · La apertura está por debajo del umbral.")
        else:
            st.success("OJO ABIERTO · La apertura alcanza o supera el umbral.")

    with st.expander("Editar las seis coordenadas P1–P6"):
        st.caption("Ajuste geométrico educativo. Para cambiar la detección real usa los controles de sensibilidad de arriba.")
        datos = pd.DataFrame(PUNTOS_EJEMPLO, columns=["X", "Y"], index=[f"P{i}" for i in range(1, 7)])
        tabla = st.data_editor(datos, key="puntos_ear_editor", num_rows="fixed", width="stretch",
                               column_config={c: st.column_config.NumberColumn(c, min_value=0,
                                              max_value=200, required=True) for c in ("X", "Y")})
        puntos = tabla.to_numpy(dtype=float)
        if not np.isfinite(puntos).all() or np.linalg.norm(puntos[0] - puntos[3]) < 1:
            st.error("Completa los seis puntos y separa P1 y P4 para medir el ancho del ojo.")
        else:
            a, b = st.columns(2)
            a.markdown(ojo_svg(puntos), unsafe_allow_html=True)
            b.metric("EAR de las coordenadas", f"{calcular_ear(puntos):.3f}")
            b.latex(r"EAR = \frac{\|P_2-P_6\| + \|P_3-P_5\|}{2\|P_1-P_4\|}")
            b.caption("P1 y P4: extremos. P2 y P3: párpado superior. P5 y P6: párpado inferior.")

    with st.container(border=True, key="guia_calibracion"):
        st.markdown("##### 03 / CALIBRACIÓN CON CÁMARA")
        a, b, c = st.columns(3)
        a.markdown("**Selecciona tu conductor**\n\nElige un perfil en la barra lateral para guardar la sesión.")
        b.markdown("**Inicia la cámara**\n\nEn Monitoreo en vivo pulsa START. Mira al frente con los ojos abiertos.")
        c.markdown("**Comprueba los puntos**\n\nRevisa ambos ojos en el panel EAR. Recalibra si cambias de posición o iluminación.")


