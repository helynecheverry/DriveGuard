"""Pestaña para analizar un video grabado (trabajo en clase sin conducir)."""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path

import streamlit as st

from core.config import VIDEOS_DIR, ConfigRiesgo
from frontend import api_client as api
from frontend import estado as est
from frontend.graficos import grafico_ear, grafico_niveles, grafico_perclos


def _eventos_desde_df(df, intervalo_muestra: float = 5.0) -> list[dict]:
    """Convierte el análisis por fotograma en eventos (cambios de nivel + muestras)."""
    inicio = datetime.now(timezone.utc)
    eventos, nivel_prev, t_muestra = [], "NORMAL", -1e9
    for fila in df[~df["calibrando"]].itertuples():
        base = {"t_s": round(fila.t, 3), "fecha": (inicio + timedelta(seconds=fila.t)).isoformat(),
                "nivel": fila.nivel, "motivos": [m for m in fila.motivos.split(",") if m],
                "ear": None if fila.ear is None else round(float(fila.ear), 4),
                "ear_umbral": round(float(fila.ear_umbral), 4), "mar": fila.mar,
                "perclos": round(float(fila.perclos), 4), "bostezos": int(fila.bostezos),
                "cabeceos": int(fila.cabeceos), "rostro": bool(fila.rostro)}
        if fila.nivel != nivel_prev:
            tipo = base["motivos"][0] if fila.nivel != "NORMAL" and base["motivos"] else "recuperacion"
            eventos.append({**base, "tipo": tipo, "nivel_anterior": nivel_prev})
            nivel_prev = fila.nivel
        if fila.t - t_muestra >= intervalo_muestra:
            eventos.append({**base, "tipo": "muestra"})
            t_muestra = fila.t
    return eventos


def render() -> None:
    from core.pipeline import analizar_video

    st.subheader("Analizar un video de prueba")
    st.caption("Grabe 1-2 minutos frente a la cámara simulando parpadeos normales, ojos cerrados "
               "más de 2 s, bostezos y cabeceos. El análisis usa el mismo motor que la cámara en vivo.")
    archivo = st.file_uploader("Video (mp4, avi, mov, mkv)", type=["mp4", "avi", "mov", "mkv"], key="video_prueba")
    c1, c2, c3 = st.columns(3)
    max_s = c1.number_input("Analizar como máximo (s)", 10, 900, 180, 10)
    expresiones = c2.toggle("Incluir expresiones (más lento)", value=False)
    guardar = c3.toggle("Guardar en MongoDB", value=False, disabled=st.session_state.conductor is None,
                        help="Requiere un conductor seleccionado")
    if archivo is None or not st.button("Analizar video", type="primary"):
        if "df_video" in st.session_state:
            _mostrar(st.session_state.df_video, st.session_state.get("video_anotado"))
        return

    ruta = VIDEOS_DIR / Path(archivo.name).name
    ruta.write_bytes(archivo.getvalue())
    salida = VIDEOS_DIR / f"{ruta.stem}_anotado.mp4"
    barra = st.progress(0.0, text="Procesando fotogramas…")
    try:
        df = analizar_video(ruta, ConfigRiesgo.desde_dict(st.session_state.config), expresiones=expresiones,
                            salida=salida, max_segundos=max_s,
                            progreso=lambda p: barra.progress(p, text=f"Procesando… {p * 100:.0f} %"))
    except Exception as exc:
        st.error(f"No se pudo analizar el video: {exc}")
        return
    barra.empty()
    st.session_state.df_video, st.session_state.video_anotado = df, salida

    if guardar and st.session_state.conductor:
        try:
            sid = api.crear_sesion(st.session_state.conductor["id"], est.ubicacion_actual(),
                                   st.session_state.config, origen="video")
            api.enviar_eventos(sid, _eventos_desde_df(df))
            api.cerrar_sesion(sid)
            st.success(f"Resultados guardados en la sesión `{sid}` (ver Historial).")
        except api.ApiError as exc:
            st.warning(f"No se guardó en MongoDB: {exc}")
    _mostrar(df, salida)


def _mostrar(df, salida) -> None:
    if df is None or df.empty:
        st.warning("El video no contiene fotogramas analizables.")
        return
    validos = df[~df["calibrando"]]
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Duración analizada", f"{df['t'].max():.0f} s")
    c2.metric("Rostro detectado", f"{df['rostro'].mean() * 100:.0f} %")
    orden = ["NORMAL", "BAJO", "MEDIO", "ALTO"]
    maximo = max((orden.index(n) for n in validos["nivel"]), default=0)
    c3.metric("Nivel máximo", orden[maximo])
    c4.metric("Emergencia", "Sí" if df["emergencia"].any() else "No")
    st.plotly_chart(grafico_ear(validos), width="stretch")
    a, b = st.columns(2)
    a.plotly_chart(grafico_niveles(validos), width="stretch")
    b.plotly_chart(grafico_perclos(validos, "t"), width="stretch")
    col1, col2 = st.columns(2)
    col1.download_button("Descargar métricas (CSV)", df.to_csv(index=False).encode(), "metricas_video.csv", "text/csv")
    if salida and Path(salida).exists():
        col2.download_button("Descargar video anotado", Path(salida).read_bytes(), Path(salida).name, "video/mp4")
