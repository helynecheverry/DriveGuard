"""Pestaña de historial: sesiones, eventos, gráficos y alertas enviadas."""
from __future__ import annotations

import pandas as pd
import streamlit as st

from core.riesgo import DESCRIPCION_MOTIVOS
from frontend import api_client as api
from frontend.graficos import grafico_ear, grafico_perclos, grafico_tipos


def render() -> None:
    st.subheader("Historial de sesiones")
    conductor = st.session_state.conductor
    todos = st.toggle("Ver sesiones de todos los conductores", value=conductor is None)
    try:
        sesiones = api.listar_sesiones(None if todos or conductor is None else conductor["id"])
    except api.ApiError as exc:
        st.error(str(exc))
        return
    if not sesiones:
        st.info("Aún no hay sesiones. Inicie la cámara en *Monitoreo en vivo* o analice un video de prueba.")
        return

    nombres = {c["id"]: c["nombre"] for c in st.session_state.conductores}

    def etiqueta(s):
        inicio = pd.to_datetime(s["inicio"]).tz_convert(None).strftime("%Y-%m-%d %H:%M")
        resumen = s.get("resumen") or {}
        return (f"{inicio} · {nombres.get(s['conductor_id'], 'conductor')} · {s.get('origen', 'camara')} · "
                f"{resumen.get('eventos_riesgo', '?')} eventos")

    tabla = pd.DataFrame([{"inicio": s["inicio"], "conductor": nombres.get(s["conductor_id"], s["conductor_id"]),
                           "origen": s.get("origen"), "eventos_riesgo": (s.get("resumen") or {}).get("eventos_riesgo"),
                           "alertas": (s.get("resumen") or {}).get("alertas"), "cerrada": bool(s.get("fin"))}
                          for s in sesiones])
    with st.expander(f"Resumen de {len(sesiones)} sesiones", expanded=False):
        st.dataframe(tabla, hide_index=True, width="stretch")

    sel = st.selectbox("Sesión", sesiones, format_func=etiqueta)
    try:
        stats = api.estadisticas(sel["id"])
        eventos = pd.DataFrame(api.eventos(sel["id"]))
        alertas = api.alertas(sel["id"])
    except api.ApiError as exc:
        st.error(str(exc))
        return

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Duración", f"{stats['duracion_s'] / 60:.1f} min")
    c2.metric("Eventos de riesgo", stats["eventos_riesgo"])
    c3.metric("EAR promedio", stats["ear_promedio"] if stats["ear_promedio"] is not None else "—")
    c4.metric("PERCLOS máx.", f"{stats['perclos_max'] * 100:.0f} %" if stats["perclos_max"] is not None else "—")
    c5.metric("Alertas enviadas", stats["alertas"])

    if eventos.empty:
        st.info("La sesión no tiene eventos registrados.")
        return
    eventos["fecha"] = pd.to_datetime(eventos["fecha"]).dt.tz_convert(None)
    muestras = eventos[eventos["tipo"] == "muestra"].dropna(subset=["ear"])
    riesgos = eventos[~eventos["tipo"].isin(["muestra", "recuperacion"])]

    if not muestras.empty:
        a, b = st.columns(2)
        a.plotly_chart(grafico_ear(muestras, x="fecha", titulo="EAR (muestras cada 5 s)"), width="stretch")
        b.plotly_chart(grafico_perclos(muestras, "fecha"), width="stretch")
    if stats["por_tipo"]:
        nombres_tipo = {k: DESCRIPCION_MOTIVOS.get(k, k).split(":")[0] for k in stats["por_tipo"]}
        st.plotly_chart(grafico_tipos({nombres_tipo[k]: v for k, v in stats["por_tipo"].items()}),
                        width="stretch")

    st.markdown("#### Eventos de riesgo")
    if riesgos.empty:
        st.success("Sin eventos de riesgo en esta sesión.")
    else:
        cols = [c for c in ["fecha", "tipo", "nivel", "nivel_anterior", "ear", "perclos", "bostezos",
                            "cabeceos", "expresion"] if c in riesgos]
        st.dataframe(riesgos[cols], hide_index=True, width="stretch")

    if alertas:
        st.markdown("#### Alertas de emergencia")
        filas = [{"fecha": a["fecha"], "motivo": a.get("motivo"), "estado": a.get("estado"),
                  "canales": ", ".join(f"{r['canal']}:{'ok' if r['ok'] else 'error'}" for r in a.get("resultados", []))}
                 for a in alertas]
        st.dataframe(pd.DataFrame(filas), hide_index=True, width="stretch")

    st.download_button("Descargar eventos (CSV)", eventos.to_csv(index=False).encode(),
                       f"eventos_{sel['id']}.csv", "text/csv")
