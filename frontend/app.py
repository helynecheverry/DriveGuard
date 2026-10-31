"""DriveGuard Lab — interfaz Streamlit.

Ejecutar desde la raíz del proyecto (con la API ya iniciada):
    python -m streamlit run frontend/app.py
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import streamlit as st  # noqa: E402

from frontend import api_client as api  # noqa: E402
from frontend import estado as est  # noqa: E402

st.set_page_config(page_title="DriveGuard Lab", page_icon="🚗", layout="wide")
est.inicializar()
from frontend import diseno  # noqa: E402
diseno.aplicar()


def barra_lateral() -> None:
    with st.sidebar:
        st.markdown('<div class="dg-brand"><div class="dg-brand-icon">◉</div>'
                    '<div><strong>DriveGuard</strong><small>DRIVER SAFETY LAB</small></div></div>',
                    unsafe_allow_html=True)

        try:
            salud = api.health()
            st.markdown('<div class="dg-online">● &nbsp; API conectada</div>', unsafe_allow_html=True)
            if salud.get("mongodb"):
                st.markdown('<div class="dg-online">● &nbsp; MongoDB conectado</div>', unsafe_allow_html=True)
            else:
                st.warning("MongoDB no disponible: no se guardarán sesiones", icon="⚠️")
        except api.ApiError as exc:
            st.error("API no disponible", icon="⛔")
            st.caption(str(exc))

        st.markdown("### Conductor")
        if not st.session_state.conductores:
            est.cargar_conductores()
        conductores = st.session_state.conductores
        ids = [None] + [c["id"] for c in conductores]
        actual = st.session_state.conductor["id"] if st.session_state.conductor else None
        elegido = st.selectbox(
            "Conductor activo", ids, index=ids.index(actual) if actual in ids else 0,
            format_func=lambda i: "— Sin seleccionar —" if i is None else
            next(c["nombre"] for c in conductores if c["id"] == i),
            disabled=st.session_state.sesion_id is not None,
            help="No se puede cambiar mientras la cámara está activa")
        st.session_state.conductor = next((c for c in conductores if c["id"] == elegido), None)
        if st.button("↻ Actualizar lista", width="stretch"):
            est.cargar_conductores()
            st.rerun()
        if st.session_state.conductor:
            n = len(st.session_state.conductor.get("contactos") or [])
            st.caption(f"{n} contacto(s) de emergencia")

        st.markdown("### Ubicación de la sesión")
        u = st.session_state.ubicacion
        u["descripcion"] = st.text_input("Descripción", u.get("descripcion", ""),
                                         placeholder="Vía Pasto - Ipiales, km 12")
        c1, c2 = st.columns(2)
        u["lat"] = c1.number_input("Latitud", value=u.get("lat"), format="%.6f", min_value=-90.0, max_value=90.0)
        u["lon"] = c2.number_input("Longitud", value=u.get("lon"), format="%.6f", min_value=-180.0, max_value=180.0)
        st.caption("Se incluye en el mensaje de emergencia. En el prototipo se ingresa a mano.")


barra_lateral()

st.markdown('''<div class="dg-hero"><small>CENTRO DE MONITOREO · DRIVEGUARD LAB</small>
<h1>Cada mirada cuenta.</h1>
<p>Un espacio para ajustar la detección, seguir tu atención y entender las señales de cada sesión.</p></div>''',
            unsafe_allow_html=True)

diseno.resumen()

tab_c, tab_cal, tab_m, tab_v, tab_h, tab_cfg = st.tabs(
    ["Conductor", "Ojos y calibración", "Monitoreo en vivo", "Video de prueba", "Historial", "Configuración"])

from frontend import calibracion_ui, conductor_ui, config_ui, historial_ui, monitoreo_ui, video_ui  # noqa: E402

with tab_c:
    conductor_ui.render()
with tab_cal:
    calibracion_ui.render()
with tab_m:
    monitoreo_ui.render()
with tab_v:
    video_ui.render()
with tab_h:
    historial_ui.render()
with tab_cfg:
    config_ui.render()

st.divider()
st.warning("**Prototipo académico.** No sustituye sistemas de seguridad vehicular certificados. "
           "No lo use mientras conduce en vía pública: las pruebas se hacen con el vehículo detenido o "
           "en simulación. Las expresiones faciales son apariencias, no emociones reales.", icon="⚠️")
