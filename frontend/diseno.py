"""Estilo compartido del centro de monitoreo."""
import streamlit as st


def aplicar() -> None:
    st.markdown("""<style>
    :root {--dg-ink:#172e48;--dg-teal:#087f8c;--dg-muted:#475569}
    [data-testid="stAppViewContainer"] {background:#f3f6fa;color:var(--dg-ink)}
    .block-container {padding-top:2rem;max-width:1536px;padding-bottom:2rem}
    [data-testid="stHeader"] {background:#f3f6faee}
    [data-testid="stSidebar"] {background:#fff;border-right:1px solid #e2e8f0}
    [data-testid="stSidebar"] .block-container {padding-top:1rem}
    h1,h2,h3,h4,h5 {color:var(--dg-ink);letter-spacing:-.025em}
    [data-testid="stMarkdownContainer"], [data-testid="stWidgetLabel"] {color:var(--dg-ink)}
    [data-testid="stCaptionContainer"], [data-testid="stCaptionContainer"] p,
    [data-testid="stCaptionContainer"] [data-testid="stMarkdownContainer"] {color:var(--dg-muted)!important;opacity:1!important}
    [data-testid="stWidgetLabel"] p {color:var(--dg-ink)!important;opacity:1!important}
    input,textarea,select,[role="combobox"] {color:#172e48!important;background-color:#fff!important;-webkit-text-fill-color:#172e48}
    input::placeholder,textarea::placeholder {color:#52647a!important;opacity:1}
    input:disabled,textarea:disabled {color:#52647a!important;-webkit-text-fill-color:#52647a;opacity:1}
    [data-testid="stVerticalBlockBorderWrapper"] > div {border-radius:18px}
    [data-testid="stVerticalBlockBorderWrapper"] {background:#fff;border-radius:18px}
    [data-testid="stMetric"] {background:#fff;border:1px solid #e3eaf2;border-radius:14px;padding:16px 20px}
    [data-testid="stMetricLabel"] {color:var(--dg-muted);font-size:.8rem}
    [data-testid="stMetricLabel"] p {color:var(--dg-muted)!important}
    [data-testid="stMetricValue"] {font-size:1.65rem;color:var(--dg-ink);font-weight:650}
    [role="tablist"] {gap:6px!important;background:#e8edf4;padding:6px!important;border-radius:14px;flex-wrap:wrap;border:0!important}
    [role="tab"] {border-radius:10px;padding:12px 16px!important;height:auto!important;color:#52647a;border:0!important}
    [role="tab"][aria-selected="true"] {background:#fff!important;color:#087f8c!important;box-shadow:0 2px 6px #172e4810}
    [role="tab"] [data-testid="stMarkdownContainer"], [role="tab"] p {color:inherit!important}
    [role="tab"]:after,[role="tab"]:before {display:none!important}
    [role="tabpanel"] {padding-top:26px}
    .st-key-controles_ojos,.st-key-prueba_apertura,.st-key-guia_calibracion {background:#fff;border-radius:18px}
    button[kind="primary"],button[kind="primaryFormSubmit"] {background:#087f8c!important;border-color:#087f8c!important;border-radius:10px;color:#fff!important}
    button[kind="primary"] *,button[kind="primaryFormSubmit"] * {color:#fff!important}
    button[kind="secondary"],button[kind="secondaryFormSubmit"] {background:#fff;color:#172e48;border-radius:10px;border-color:#94a3b8}
    button[kind="secondary"] p,button[kind="secondaryFormSubmit"] p {color:#172e48!important}
    button:disabled {opacity:.65}
    [data-testid="stAlert"] [data-testid="stMarkdownContainer"] {color:#172e48!important}
    [data-baseweb="input"],[data-baseweb="select"] > div {border-radius:10px}
    [data-testid="stExpander"] {background:#fff;border-radius:12px}
    .dg-brand {display:flex;gap:12px;align-items:center;margin:8px 0 22px}
    .dg-brand-icon {display:grid;place-items:center;width:42px;height:42px;background:#087f8c;color:white;border-radius:13px;font-size:22px}
    .dg-brand strong {font-size:20px;letter-spacing:-.5px;color:#172e48}
    .dg-brand small {display:block;font-size:11px;letter-spacing:1.6px;color:#475569}
    .dg-hero {position:relative;overflow:hidden;padding:30px 34px;border-radius:22px;background:linear-gradient(110deg,#112942,#154a58);color:#f1f5f9;margin-bottom:6px}
    .dg-hero:after {content:"";position:absolute;width:210px;height:210px;right:-25px;top:-65px;border:35px solid #5eead40c;border-radius:50%;pointer-events:none}
    .dg-hero small {color:#7dded3;letter-spacing:2px;font-size:11px;font-weight:600}
    .dg-hero h1 {color:#f8fafc;font-size:2.15rem;padding:8px 0;letter-spacing:-1px}
    .dg-hero p {color:#bfd3de;margin:0;max-width:690px;font-size:.95rem}
    .dg-online {padding:9px 12px;border-radius:9px;background:#ecf9f4;color:#19745a;font-size:13px;margin:6px 0}
    @media(max-width:700px) {.block-container{padding:1.3rem 1rem}.dg-hero{padding:22px}.dg-hero h1{font-size:1.7rem}[role="tab"]{padding:9px!important}}
    </style>""", unsafe_allow_html=True)


def resumen() -> None:
    conductor = st.session_state.conductor
    cfg = st.session_state.config
    a, b, c = st.columns(3)
    a.metric("Conductor activo", conductor["nombre"] if conductor else "Sin seleccionar")
    manual = cfg.get("ear_umbral_manual")
    b.metric("Detección de ojos", "Manual" if manual is not None else "Automática")
    c.metric("Umbral EAR" if manual is not None else "Factor de calibración",
             f"{manual:.3f}" if manual is not None else f"{cfg['factor_ear'] * 100:.0f} %")
    st.write("")
