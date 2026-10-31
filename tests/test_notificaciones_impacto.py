"""Pruebas de notificaciones (con simulaciones, sin enviar nada) e impacto."""
from unittest import mock

import pytest

from core import notificaciones as n
from core.impacto import cargar_acelerometro, detectar_impacto, generar_csv_ejemplo

CONDUCTOR = {"nombre": "Ana Pérez", "contactos": [
    {"nombre": "Luis", "correo": "luis@example.com", "telegram_chat_id": "12345"},
    {"nombre": "Marta", "correo": "marta@example.com"}]}


def test_mensaje_incluye_datos_clave():
    asunto, cuerpo = n.construir_mensaje(CONDUCTOR, "sin_respuesta", "ALTO",
                                         {"descripcion": "Vía Pasto", "lat": 1.2, "lon": -77.3})
    assert "Ana Pérez" in asunto
    assert "no responde" in cuerpo and "ALTO" in cuerpo
    assert "maps.google.com/?q=1.2,-77.3" in cuerpo
    assert "prototipo académico" in cuerpo


def test_modo_simulado_no_envia(monkeypatch):
    with mock.patch.object(n, "enviar_correo") as correo, mock.patch.object(n, "enviar_telegram") as tg:
        res = n.notificar_contactos(CONDUCTOR, "sos", simulado=True)
    correo.assert_not_called()
    tg.assert_not_called()
    assert len(res) == 3 and all(r["ok"] and r["simulado"] for r in res)


def test_correo_sin_credenciales_informa_error(monkeypatch):
    monkeypatch.delenv("SMTP_USER", raising=False)
    monkeypatch.delenv("SMTP_PASSWORD", raising=False)
    r = n.enviar_correo("a@b.com", "x", "y")
    assert not r["ok"] and "no configurados" in r["error"]


def test_correo_usa_starttls_y_login(monkeypatch):
    monkeypatch.setenv("SMTP_USER", "bot@gmail.com")
    monkeypatch.setenv("SMTP_PASSWORD", "clave")
    with mock.patch("smtplib.SMTP") as smtp:
        r = n.enviar_correo("a@b.com", "asunto", "cuerpo")
    servidor = smtp.return_value.__enter__.return_value
    servidor.starttls.assert_called_once()
    servidor.login.assert_called_once_with("bot@gmail.com", "clave")
    servidor.send_message.assert_called_once()
    assert r["ok"]


def test_telegram_ok_y_error(monkeypatch):
    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "TOKEN")
    respuesta = mock.Mock(ok=True, headers={"content-type": "application/json"})
    respuesta.json.return_value = {"ok": True}
    with mock.patch("requests.post", return_value=respuesta) as post:
        assert n.enviar_telegram("123", "hola")["ok"]
    assert "botTOKEN/sendMessage" in post.call_args.args[0]
    respuesta.ok = False
    respuesta.status_code = 400
    respuesta.json.return_value = {"ok": False, "description": "chat not found"}
    with mock.patch("requests.post", return_value=respuesta):
        r = n.enviar_telegram("123", "hola")
    assert not r["ok"] and r["error"] == "chat not found"


def test_sin_contactos():
    res = n.notificar_contactos({"nombre": "X", "contactos": []}, "sos", simulado=True)
    assert not res[0]["ok"]


def test_impacto_detectado_y_no_detectado(tmp_path):
    con = cargar_acelerometro(generar_csv_ejemplo(tmp_path / "c.csv", choque=True))
    sin = cargar_acelerometro(generar_csv_ejemplo(tmp_path / "s.csv", choque=False))
    r = detectar_impacto(con, 4.0)
    assert r["impacto"] and r["t_pico"] == pytest.approx(12.0, abs=0.05)
    assert not detectar_impacto(sin, 4.0)["impacto"]


def test_acelerometro_en_ms2_se_convierte_a_g(tmp_path):
    ruta = generar_csv_ejemplo(tmp_path / "c.csv")
    import pandas as pd
    df = pd.read_csv(ruta)
    df[["ax", "ay", "az"]] *= 9.80665
    df.to_csv(ruta, index=False)
    assert cargar_acelerometro(ruta)["magnitud_g"].median() == pytest.approx(1.0, abs=0.1)


def test_csv_sin_columnas(tmp_path):
    ruta = tmp_path / "malo.csv"
    ruta.write_text("a,b\n1,2\n")
    with pytest.raises(ValueError):
        cargar_acelerometro(ruta)
