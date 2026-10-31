"""Pestaña de registro de conductores y contactos de emergencia."""
from __future__ import annotations

import pandas as pd
import streamlit as st

from frontend import api_client as api
from frontend import estado as est

COLUMNAS = ["nombre", "correo", "telegram_chat_id", "telefono"]


def _editor_contactos(contactos: list[dict], key: str) -> list[dict]:
    df = pd.DataFrame(contactos or [], columns=COLUMNAS).astype("string")
    editado = st.data_editor(
        df, key=key, num_rows="dynamic", width="stretch", hide_index=True,
        column_config={
            "nombre": st.column_config.TextColumn("Nombre", required=True),
            "correo": st.column_config.TextColumn("Correo electrónico"),
            "telegram_chat_id": st.column_config.TextColumn("Telegram chat_id",
                                                            help="Ver Configuración → Buscar chat_id"),
            "telefono": st.column_config.TextColumn("Teléfono (SMS, opcional)", help="+573001234567"),
        })
    salida = []
    for fila in editado.fillna("").to_dict("records"):
        fila = {k: (str(v).strip() or None) for k, v in fila.items()}
        if fila.get("nombre"):
            salida.append(fila)
    return salida


def render() -> None:
    st.subheader("Conductores y contactos de emergencia")
    st.caption("Los contactos reciben el aviso por correo y/o Telegram si se activa el protocolo de emergencia.")

    izquierda, derecha = st.columns(2, gap="large")
    with izquierda:
        st.markdown("#### Registrar conductor")
        with st.form("nuevo_conductor", clear_on_submit=True):
            nombre = st.text_input("Nombre completo")
            documento = st.text_input("Documento (opcional)")
            st.markdown("Contactos de emergencia")
            contactos = _editor_contactos([], "contactos_nuevo")
            if st.form_submit_button("Guardar conductor", type="primary"):
                if len(nombre.strip()) < 2:
                    st.error("Escriba el nombre del conductor.")
                else:
                    try:
                        nuevo_id = api.crear_conductor({"nombre": nombre.strip(), "documento": documento or None,
                                                        "contactos": contactos})
                        est.cargar_conductores()
                        st.session_state.conductor = next(
                            (c for c in st.session_state.conductores if c["id"] == nuevo_id), None)
                        st.success("Conductor registrado y seleccionado.")
                    except api.ApiError as exc:
                        st.error(str(exc))

    with derecha:
        st.markdown("#### Conductor seleccionado")
        c = st.session_state.conductor
        if c is None:
            st.info("Seleccione un conductor en la barra lateral o registre uno nuevo.")
            return
        with st.form(f"editar_{c['id']}"):
            nombre = st.text_input("Nombre completo", c.get("nombre", ""))
            documento = st.text_input("Documento", c.get("documento") or "")
            contactos = _editor_contactos(c.get("contactos", []), f"contactos_{c['id']}")
            guardar = st.form_submit_button("Guardar cambios", type="primary")
        if guardar:
            try:
                api.actualizar_conductor(c["id"], {"nombre": nombre.strip(), "documento": documento or None,
                                                   "contactos": contactos})
                est.cargar_conductores()
                st.session_state.conductor = next(
                    (x for x in st.session_state.conductores if x["id"] == c["id"]), None)
                st.success("Cambios guardados.")
            except api.ApiError as exc:
                st.error(str(exc))
        with st.popover("🗑️ Eliminar conductor"):
            st.write(f"¿Eliminar a **{c['nombre']}**? Sus sesiones se conservan en MongoDB.")
            if st.button("Sí, eliminar", type="primary"):
                try:
                    api.eliminar_conductor(c["id"])
                    st.session_state.conductor = None
                    est.cargar_conductores()
                    st.rerun()
                except api.ApiError as exc:
                    st.error(str(exc))
