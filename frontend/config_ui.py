"""Pestaña de configuración: umbrales, sonido, expresiones y pruebas de notificación."""
from __future__ import annotations

import pandas as pd
import streamlit as st

from core.config import ConfigRiesgo
from frontend import api_client as api


@st.cache_resource
def _reproductor():
    from core.alarma import crear_reproductor
    return crear_reproductor()


def render() -> None:
    st.subheader("Configuración")
    cfg = ConfigRiesgo.desde_dict(st.session_state.config)
    op = st.session_state.opciones
    st.info("La apertura de ojos, el umbral EAR y la calibración se ajustan en la pestaña Ojos y calibración.")

    with st.form("umbrales"):
        st.markdown("#### Umbrales del motor de riesgo")
        a, b, c = st.columns(3)
        with a:
            st.markdown("**Cansancio acumulado**")
            perclos = st.slider("PERCLOS máximo (%)", 10, 80, int(cfg.perclos_umbral * 100), 5)
            st.caption("Porcentaje de muestras con ojos cerrados durante el último minuto para detectar cansancio sostenido.")
        with b:
            st.markdown("**Boca y cabeza**")
            mar = st.slider("MAR de bostezo", 0.30, 1.20, float(cfg.mar_umbral), 0.05)
            bostezo_s = st.slider("Duración mínima de bostezo (s)", 0.5, 5.0, float(cfg.bostezo_min_s), 0.5)
            bostezos = st.slider("Bostezos en 5 min para alertar", 1, 10, int(cfg.bostezos_limite))
            cabeceo = st.slider("Cabeceo: inclinación (°)", 5.0, 40.0, float(cfg.cabeceo_grados), 1.0)
            giro = st.slider("Distracción: giro (°)", 10.0, 60.0, float(cfg.giro_grados), 1.0)
        with c:
            st.markdown("**Tiempos y emergencia**")
            distraccion_s = st.slider("Distracción / rostro ausente (s)", 1.0, 10.0, float(cfg.distraccion_s), 0.5)
            histeresis_s = st.slider("Histéresis al bajar de nivel (s)", 0.0, 10.0, float(cfg.histeresis_s), 0.5)
            sin_respuesta_s = st.slider("Sin respuesta tras ALTO → emergencia (s)", 3.0, 60.0,
                                        float(cfg.sin_respuesta_s), 1.0)
            expresion = st.slider("Expresión negativa sostenida (%)", 30, 100, int(cfg.expresion_umbral * 100), 5)
        if st.form_submit_button("Aplicar umbrales", type="primary"):
            st.session_state.config = {**st.session_state.config,
                                       "perclos_umbral": perclos / 100, "mar_umbral": mar,
                                       "bostezo_min_s": bostezo_s, "bostezos_limite": bostezos,
                                       "cabeceo_grados": cabeceo, "giro_grados": giro,
                                       "distraccion_s": distraccion_s, "histeresis_s": histeresis_s,
                                       "sin_respuesta_s": sin_respuesta_s, "expresion_umbral": expresion / 100}
            st.success("Umbrales aplicados. Si la cámara está activa se actualizan de inmediato.")
    if st.button("Restaurar valores por defecto"):
        st.session_state.config = ConfigRiesgo().a_dict()
        st.rerun()

    st.divider()
    a, b = st.columns(2, gap="large")
    with a:
        st.markdown("#### Sonido y video")
        op["sonido"] = st.toggle("Alarma sonora", value=op["sonido"])
        op["volumen"] = st.slider("Volumen (pygame)", 0.0, 1.0, float(op["volumen"]), 0.05)
        op["espejo"] = st.toggle("Vista en espejo", value=op["espejo"])
        rep = _reproductor()
        st.caption(f"Reproductor detectado: **{rep.nombre}**")
        b1, b2 = st.columns(2)
        if b1.button("🔔 Probar aviso MEDIO", width="stretch"):
            rep.volumen(op["volumen"])
            rep.reproducir("media", bucle=False)
        if b2.button("🚨 Probar alarma ALTO", width="stretch"):
            rep.volumen(op["volumen"])
            rep.reproducir("alta", bucle=False)

        st.markdown("#### Expresiones faciales (ViT)")
        op["expresiones"] = st.toggle("Analizar expresiones", value=op["expresiones"],
                                      help="Se aplica al iniciar de nuevo la cámara")
        op["intervalo_expresiones"] = st.slider(
            "Intervalo entre clasificaciones (s)", 1.0, 10.0, float(op["intervalo_expresiones"]), 0.5,
            help="El ViT es lento en CPU. Subir este valor no afecta la detección de somnolencia.")

    with b:
        st.markdown("#### Notificaciones")
        try:
            canales = api.canales()
            st.dataframe(pd.DataFrame([{"canal": k, "configurado": "sí" if v else "no"} for k, v in canales.items()]),
                         hide_index=True, width="stretch")
        except api.ApiError as exc:
            st.error(str(exc))
        conductor = st.session_state.conductor
        simulado = st.toggle("Prueba simulada (no envía mensajes reales)", value=False)
        if st.button("✉️ Enviar mensaje de prueba", disabled=conductor is None, width="stretch"):
            try:
                r = api.prueba_alertas(conductor["id"], simulado=simulado)
                (st.success if r["estado"] == "enviada" else st.error)(f"Resultado: {r['estado']}")
                st.dataframe(pd.DataFrame(r["resultados"]), hide_index=True, width="stretch")
            except api.ApiError as exc:
                st.error(str(exc))
        if conductor is None:
            st.caption("Seleccione un conductor con contactos para probar el envío.")
        with st.expander("¿Cómo obtengo el chat_id de Telegram?"):
            st.markdown("1. Cree un bot con **@BotFather** (`/newbot`) y copie el token en `TELEGRAM_BOT_TOKEN`.\n"
                        "2. Cada contacto abre el bot y le escribe cualquier mensaje.\n"
                        "3. Pulse el botón: aparecerán los chat_id disponibles.")
            if st.button("Buscar chat_id"):
                try:
                    chats = api.telegram_chats()
                    if chats:
                        st.dataframe(pd.DataFrame(chats), hide_index=True)
                    else:
                        st.info("No hay mensajes recientes. Escriba al bot y vuelva a intentar.")
                except api.ApiError as exc:
                    st.error(str(exc))
