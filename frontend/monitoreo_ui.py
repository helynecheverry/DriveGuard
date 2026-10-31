"""Pestaña de monitoreo en vivo con la cámara (streamlit-webrtc).

Arquitectura de hilos:
- Hilo de video (streamlit-webrtc): recibe cada fotograma y ejecuta
  ProcesadorConductor (MediaPipe + motor de riesgo + alarma local).
- Hilo de Streamlit: dos fragmentos que se refrescan cada 0,5 s leen el
  estado del procesador, muestran el semáforo, envían los eventos a la API
  y manejan la cuenta regresiva del protocolo de emergencia.
"""
from __future__ import annotations

import time

import cv2
import pandas as pd
import streamlit as st

from core.config import EJEMPLOS_DIR, ConfigRiesgo
from core.riesgo import DESCRIPCION_MOTIVOS
from frontend import api_client as api
from frontend import estado as est

RTC_CONFIG = {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}


def _crear_procesador_webrtc():
    """Define la clase del procesador de video (importa streamlit-webrtc solo si existe)."""
    import av
    from streamlit_webrtc import VideoProcessorBase

    class ProcesadorVideo(VideoProcessorBase):
        def __init__(self, config: ConfigRiesgo, opciones: dict):
            self.proc = None
            self.error = None
            self.espejo = opciones.get("espejo", True)
            try:
                from core.alarma import Alarma
                from core.pipeline import ProcesadorConductor
                alarma = Alarma(volumen=opciones.get("volumen", 0.8), activa=opciones.get("sonido", True))
                self.proc = ProcesadorConductor(config=config, expresiones=opciones.get("expresiones", True),
                                                intervalo_expresiones_s=opciones.get("intervalo_expresiones", 2.0),
                                                alarma=alarma)
            except Exception as exc:
                self.error = f"{type(exc).__name__}: {exc}"

        def recv(self, frame):
            img = frame.to_ndarray(format="bgr24")
            if self.espejo:
                img = cv2.flip(img, 1)
            if self.proc is None:
                cv2.putText(img, "Error al iniciar el detector (ver panel)", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                return av.VideoFrame.from_ndarray(img, format="bgr24")
            try:
                img, _ = self.proc.procesar(img)
            except Exception as exc:
                self.error = f"{type(exc).__name__}: {exc}"
            return av.VideoFrame.from_ndarray(img, format="bgr24")

        def on_ended(self):
            if self.proc is not None:
                self.proc.cerrar()

    return ProcesadorVideo


def _tarjeta_nivel(snap: dict) -> None:
    nivel = snap["nivel"]
    color = est.COLORES_NIVEL.get(nivel, "#666")
    motivos = "<br>".join(DESCRIPCION_MOTIVOS.get(m, m) for m in snap.get("motivos", [])) or "Sin señales de riesgo"
    if snap.get("calibrando"):
        nivel, color = "CALIBRANDO", "#3b6fd6"
        motivos = "Mire al frente con los ojos abiertos y expresión neutra."
    silencio = " · alarma silenciada" if snap.get("silenciado") else ""
    texto = "#0f2033" if nivel in ("NORMAL", "BAJO", "MEDIO") else "#ffffff"
    st.markdown(
        f"""<div style="background:{color};color:{texto};border-radius:14px;padding:16px 20px;">
        <div style="font-size:0.85rem;">NIVEL DE RIESGO{silencio}</div>
        <div style="font-size:2.1rem;font-weight:700;line-height:1.2;">{nivel}</div>
        <div style="font-size:0.95rem;margin-top:4px;">{motivos}</div></div>""",
        unsafe_allow_html=True)


def _enviar_eventos(proc) -> None:
    """Envía a la API los eventos acumulados (cada ~5 s)."""
    nuevos = proc.extraer_eventos()
    sesion_id = st.session_state.sesion_id
    if not sesion_id:
        return
    pendientes = (st.session_state.eventos_pendientes + nuevos)[-3000:]
    if pendientes and time.time() - st.session_state.t_envio_eventos >= 5:
        st.session_state.t_envio_eventos = time.time()
        try:
            api.enviar_eventos(sesion_id, pendientes)
            pendientes = []
        except api.ApiError:
            pass  # se reintenta en el próximo ciclo
    st.session_state.eventos_pendientes = pendientes


@st.fragment(run_every=0.5)
def panel_en_vivo(ctx) -> None:
    vp = ctx.video_processor if ctx is not None and ctx.state.playing else None
    if vp is None:
        st.info("Pulse **START** para iniciar la cámara. Durante los primeros segundos el sistema "
                "se calibra: mire al frente con los ojos abiertos.")
        return
    if vp.proc is None:
        st.error(f"No se pudo iniciar el detector: {vp.error}")
        st.caption("Si falta el modelo, ejecute `python scripts/descargar_modelos.py`.")
        return

    snap = vp.proc.snapshot()
    _tarjeta_nivel(snap)
    if snap.get("calibrando"):
        st.progress(snap.get("progreso_calibracion", 0.0), text="Calibrando EAR base del conductor…")
    from frontend.calibracion_ui import panel_ojos
    with st.expander("🎯 Ver los seis puntos EAR", expanded=True):
        panel_ojos(snap)

    c1, c2, c3 = st.columns(3)
    c1.metric("EAR", f"{snap['ear']:.3f}" if snap.get("ear") is not None else "—",
              help=f"Umbral de ojo cerrado: {snap['ear_umbral']:.3f}")
    c2.metric("PERCLOS (60 s)", f"{snap['perclos'] * 100:.0f} %")
    c3.metric("FPS", f"{snap['fps']:.0f}")
    c4, c5, c6 = st.columns(3)
    c4.metric("Ojos cerrados", f"{snap['tiempo_ojos_cerrados']:.1f} s")
    c5.metric("Bostezos (5 min)", snap["bostezos"])
    c6.metric("Cabeceos (60 s)", snap["cabeceos"])
    expr = snap.get("expresion") or {}
    mar = f"{snap['mar']:.2f}" if snap.get("mar") is not None else "—"
    st.caption(f"Expresión aparente: **{expr.get('etiqueta', '—')}** · MAR: {mar} · "
               f"Sonido: {snap['alarma_backend']}")
    if snap.get("error_expresiones"):
        st.warning(f"Expresiones desactivadas por error: {snap['error_expresiones']}")
    if snap.get("aviso_cnn"):
        st.caption(snap["aviso_cnn"])
    if vp.error:
        st.warning(f"Último error del procesador: {vp.error}")

    if st.button("✅ Estoy bien", key="ok_vivo", width="stretch",
                 disabled=snap["nivel"] == "NORMAL" and not snap.get("emergencia")):
        vp.proc.reconocer()
        est.cancelar_emergencia()

    _enviar_eventos(vp.proc)

    # Conductor que no responde -> protocolo de emergencia
    if snap.get("emergencia") and st.session_state.emergencia is None:
        est.iniciar_emergencia("sin_respuesta",
                               detalle=f"Ojos cerrados {snap['tiempo_ojos_cerrados']:.0f} s; "
                                       f"sin respuesta durante {snap['tiempo_sin_respuesta']:.0f} s")


@st.fragment(run_every=0.5)
def panel_emergencia(ctx) -> None:
    em = st.session_state.emergencia
    if em is None:
        return
    motivo_txt = {"sin_respuesta": "El conductor no responde a la alarma",
                  "sos": "Botón SOS activado",
                  "impacto_simulado": "Posible impacto detectado (simulación)"}.get(em["motivo"], em["motivo"])

    if not em["enviada"]:
        restante = em["duracion"] - (time.time() - em["inicio"])
        if restante > 0:
            st.error(f"🚨 **{motivo_txt}.** Se avisará a los contactos de emergencia en "
                     f"**{restante:.0f} s**. Pulse *Estoy bien* para cancelar.")
            st.progress(max(0.0, 1 - restante / max(em["duracion"], 1)))
            if st.button("✅ Estoy bien — cancelar aviso", key="ok_emergencia", type="primary",
                         width="stretch"):
                vp = ctx.video_processor if ctx is not None and ctx.state.playing else None
                if vp is not None and vp.proc is not None:
                    vp.proc.reconocer()
                est.cancelar_emergencia()
                st.rerun(scope="fragment")
            return
        with st.spinner("Enviando alerta a los contactos de emergencia…"):
            est.enviar_emergencia()

    res = em["resultado"] or {}
    if res.get("estado") == "enviada":
        st.success(f"🚨 Alerta enviada: {motivo_txt}.")
    elif res.get("estado") == "omitida_cooldown":
        st.info(f"Ya se envió una alerta en esta sesión; se omitió para evitar mensajes repetidos "
                f"(espera {res.get('espera_s', 0):.0f} s).")
    else:
        st.error(f"No se pudo enviar la alerta. {res.get('error', '')}")
    if res.get("aviso"):
        st.caption(res["aviso"])
    if res.get("resultados"):
        st.dataframe(pd.DataFrame(res["resultados"]), hide_index=True, width="stretch")
    if st.button("Cerrar aviso", key="cerrar_emergencia"):
        vp = ctx.video_processor if ctx is not None and ctx.state.playing else None
        if vp is not None and vp.proc is not None:
            vp.proc.reconocer()
        est.cancelar_emergencia()
        st.rerun(scope="fragment")


def _gestionar_sesion(ctx) -> None:
    """Crea la sesión en MongoDB al iniciar la cámara y la cierra al detenerla."""
    conductor = st.session_state.conductor
    playing = ctx is not None and ctx.state.playing
    if playing and conductor and not st.session_state.sesion_id:
        try:
            st.session_state.sesion_id = api.crear_sesion(conductor["id"], est.ubicacion_actual(),
                                                          st.session_state.config)
            st.session_state.eventos_pendientes = []
        except api.ApiError as exc:
            st.warning(f"La sesión no se guardará en MongoDB: {exc}")
    if not playing and st.session_state.sesion_id:
        sesion_id = st.session_state.sesion_id
        try:
            if st.session_state.eventos_pendientes:
                api.enviar_eventos(sesion_id, st.session_state.eventos_pendientes)
            api.cerrar_sesion(sesion_id)
        except api.ApiError:
            pass
        st.session_state.ultima_sesion_id = sesion_id
        st.session_state.sesion_id = None
        st.session_state.eventos_pendientes = []


def _simulacion_impacto() -> None:
    from core.impacto import cargar_acelerometro, detectar_impacto
    import plotly.graph_objects as go

    st.caption("Una cámara que mira al conductor no puede detectar un choque; en un vehículo real lo "
               "detecta un acelerómetro. Aquí se simula con un archivo CSV (columnas t, ax, ay, az).")
    c1, c2 = st.columns(2)
    if c1.button("💥 Simular choque (7,5 g)", width="stretch"):
        est.iniciar_emergencia("impacto_simulado", detalle="Pico de 7.5 g (botón de simulación)")
        st.rerun()
    ejemplo = EJEMPLOS_DIR / "acelerometro_choque.csv"
    if ejemplo.exists():
        c2.download_button("Descargar CSV de ejemplo", ejemplo.read_bytes(), "acelerometro_choque.csv",
                           "text/csv", width="stretch")
    umbral = st.slider("Umbral de impacto (g)", 2.0, 10.0, 4.0, 0.5)
    archivo = st.file_uploader("CSV de acelerómetro", type=["csv"], key="csv_impacto")
    if archivo is None:
        return
    try:
        df = cargar_acelerometro(archivo)
    except Exception as exc:
        st.error(f"CSV inválido: {exc}")
        return
    r = detectar_impacto(df, umbral)
    fig = go.Figure(go.Scatter(x=df["t"], y=df["magnitud_g"], mode="lines", name="|a| (g)"))
    fig.add_hline(y=umbral, line_dash="dash", line_color="#d62f2f", annotation_text="umbral")
    fig.update_layout(height=260, margin=dict(l=10, r=10, t=30, b=10),
                      xaxis_title="tiempo (s)", yaxis_title="aceleración (g)")
    st.plotly_chart(fig, width="stretch")
    if r["impacto"]:
        st.error(f"Impacto detectado: pico de {r['pico_g']} g en t = {r['t_pico']} s.")
        clave = f"{archivo.name}-{archivo.size}-{umbral}"
        if st.session_state.get("impacto_procesado") != clave:
            st.session_state.impacto_procesado = clave
            est.iniciar_emergencia("impacto_simulado", detalle=f"Pico de {r['pico_g']} g en t={r['t_pico']} s (CSV)")
            st.rerun()
    else:
        st.success(f"Sin impacto: pico máximo {r['pico_g']} g.")


def render() -> None:
    st.subheader("Monitoreo en vivo")
    st.caption("Inicia la cámara, completa la calibración y observa tus indicadores de atención.")
    conductor = st.session_state.conductor
    if conductor is None:
        st.warning("No hay conductor seleccionado: el monitoreo funciona, pero la sesión no se guarda "
                   "y no hay contactos para el protocolo de emergencia.")

    try:
        from streamlit_webrtc import WebRtcMode, webrtc_streamer
    except ImportError:
        st.error("Instale streamlit-webrtc y av para habilitar la cámara en vivo.")
        return

    cfg = ConfigRiesgo.desde_dict(st.session_state.config)
    opciones = dict(st.session_state.opciones)
    clase = _crear_procesador_webrtc()

    col_video, col_panel = st.columns([3, 2], gap="large")
    with col_video:
        ctx = webrtc_streamer(
            key="driveguard-vivo",
            mode=WebRtcMode.SENDRECV,
            video_processor_factory=lambda: clase(cfg, opciones),
            media_stream_constraints={"video": {"width": {"ideal": 640}, "height": {"ideal": 480},
                                                "frameRate": {"ideal": 30}}, "audio": False},
            rtc_configuration=RTC_CONFIG,
            async_processing=True,
        )
        b1, b2 = st.columns(2)
        if b1.button("🎯 Recalibrar", width="stretch", disabled=not ctx.state.playing):
            if ctx.video_processor and ctx.video_processor.proc:
                ctx.video_processor.proc.recalibrar()
        if b2.button("🆘 SOS", type="primary", width="stretch"):
            est.iniciar_emergencia("sos", detalle="Activado manualmente desde la interfaz", inmediata=True)
        st.caption("La alarma suena en los parlantes de este equipo. No se guardan imágenes: solo métricas.")

    # Aplicar cambios de configuración al procesador activo
    if ctx.video_processor is not None and ctx.video_processor.proc is not None:
        ctx.video_processor.proc.actualizar_config(cfg, opciones.get("intervalo_expresiones"),
                                                   opciones.get("volumen"), opciones.get("sonido"))
    _gestionar_sesion(ctx)

    with col_panel:
        panel_emergencia(ctx)
        panel_en_vivo(ctx)
        if st.session_state.sesion_id:
            st.caption(f"Sesión en MongoDB: `{st.session_state.sesion_id}`")

    with st.expander("💥 Simulación de impacto (acelerómetro)"):
        _simulacion_impacto()
