"""Gráficos Plotly reutilizados por las pestañas de video e historial."""
from __future__ import annotations

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from frontend.estado import COLORES_NIVEL

_LAYOUT = dict(height=300, margin=dict(l=10, r=10, t=40, b=10), legend=dict(orientation="h", y=-0.25))


def grafico_ear(df: pd.DataFrame, x: str = "t", titulo: str = "EAR frente al umbral") -> go.Figure:
    """Línea de EAR con el umbral y las zonas de ojos cerrados sombreadas."""
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df[x], y=df["ear"], mode="lines", name="EAR", line=dict(color="#3b6fd6", width=1.6)))
    if "ear_umbral" in df:
        fig.add_trace(go.Scatter(x=df[x], y=df["ear_umbral"], mode="lines", name="Umbral",
                                 line=dict(color="#d62f2f", dash="dash", width=1.2)))
    if "ojos_cerrados" in df and df["ojos_cerrados"].any():
        cerrado = df["ojos_cerrados"].fillna(False).astype(bool)
        bloques = (cerrado != cerrado.shift()).cumsum()
        for _, g in df[cerrado].groupby(bloques[cerrado]):
            fig.add_vrect(x0=g[x].iloc[0], x1=g[x].iloc[-1], fillcolor="#d62f2f", opacity=0.12, line_width=0)
    fig.update_layout(title=titulo, yaxis_title="EAR", **_LAYOUT)
    return fig


def grafico_niveles(df: pd.DataFrame, x: str = "t") -> go.Figure:
    orden = ["NORMAL", "BAJO", "MEDIO", "ALTO"]
    y = df["nivel"].map({n: i for i, n in enumerate(orden)})
    fig = go.Figure(go.Scatter(x=df[x], y=y, mode="lines", line=dict(shape="hv", color="#444"), name="Nivel"))
    fig.update_yaxes(tickvals=list(range(4)), ticktext=orden, range=[-0.3, 3.3])
    fig.update_layout(title="Nivel de riesgo en el tiempo", **_LAYOUT)
    return fig


def grafico_tipos(conteo: dict) -> go.Figure:
    datos = pd.DataFrame({"tipo": list(conteo), "eventos": list(conteo.values())})
    fig = px.bar(datos.sort_values("eventos"), x="eventos", y="tipo", orientation="h",
                 color_discrete_sequence=["#f07f13"])
    fig.update_layout(title="Eventos de riesgo por tipo", **_LAYOUT)
    return fig


def grafico_perclos(df: pd.DataFrame, x: str, umbral: float = 0.30) -> go.Figure:
    fig = go.Figure(go.Scatter(x=df[x], y=df["perclos"] * 100, mode="lines", name="PERCLOS",
                               line=dict(color="#8e44ad")))
    fig.add_hline(y=umbral * 100, line_dash="dash", line_color="#d62f2f", annotation_text="umbral")
    fig.update_layout(title="PERCLOS (%) — ventana de 60 s", yaxis_title="%", **_LAYOUT)
    return fig


def color_nivel(nivel: str) -> str:
    return COLORES_NIVEL.get(nivel, "#666")
