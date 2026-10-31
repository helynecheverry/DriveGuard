"""Carga datos de demostración en MongoDB (sin cámara).

Crea un conductor de ejemplo y una sesión por escenario sintético
(normal, cansancio, microsueño, sin respuesta, distracción) para que la
pestaña Historial tenga datos desde la primera clase.

Uso: python scripts/datos_demo.py           (requiere MongoDB en ejecución)
     python scripts/datos_demo.py --borrar  (elimina antes los datos demo)
"""
import argparse
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core import storage  # noqa: E402
from core.riesgo import MotorRiesgo  # noqa: E402
from core.simulacion import generar_mediciones  # noqa: E402

ESCENARIOS = ["normal", "cansancio", "microsueno", "sin_respuesta", "distraccion"]


def eventos_de_escenario(escenario: str, inicio: datetime) -> list[dict]:
    motor, eventos, nivel_prev, t_muestra = MotorRiesgo(), [], "NORMAL", -1e9
    for m in generar_mediciones(escenario):
        e = motor.actualizar(m)
        if e.calibrando:
            continue
        base = {"t_s": round(e.t, 2), "fecha": (inicio + timedelta(seconds=e.t)).isoformat(),
                "nivel": e.nivel, "motivos": e.motivos, "ear": e.ear, "ear_umbral": e.ear_umbral,
                "mar": e.mar, "perclos": e.perclos, "bostezos": e.bostezos, "cabeceos": e.cabeceos,
                "rostro": e.rostro, "demo": True}
        if e.nivel != nivel_prev:
            tipo = e.motivo_principal if e.nivel != "NORMAL" else "recuperacion"
            eventos.append({**base, "tipo": tipo or "cambio_nivel", "nivel_anterior": nivel_prev})
            nivel_prev = e.nivel
        if e.t - t_muestra >= 5:
            eventos.append({**base, "tipo": "muestra"})
            t_muestra = e.t
    return eventos


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--borrar", action="store_true")
    args = parser.parse_args()
    try:
        db = storage.get_db()
    except Exception as exc:
        sys.exit(f"[ERROR] MongoDB no disponible: {exc}")

    if args.borrar:
        ids = [str(c["_id"]) for c in db.conductores.find({"demo": True})]
        sesiones = [str(s["_id"]) for s in db.sesiones.find({"conductor_id": {"$in": ids}})]
        db.eventos.delete_many({"sesion_id": {"$in": sesiones}})
        db.alertas.delete_many({"sesion_id": {"$in": sesiones}})
        db.sesiones.delete_many({"conductor_id": {"$in": ids}})
        db.conductores.delete_many({"demo": True})
        print(f"[OK] Eliminados {len(ids)} conductores demo y {len(sesiones)} sesiones")

    cid = storage.crear_conductor({"nombre": "Conductor Demo", "documento": "DEMO-001", "demo": True,
                                   "contactos": [{"nombre": "Contacto Demo", "correo": "contacto@example.com"}]})
    inicio = datetime.now(timezone.utc) - timedelta(hours=len(ESCENARIOS))
    for i, esc in enumerate(ESCENARIOS):
        sid = storage.crear_sesion(cid, {"descripcion": f"Simulación: {esc}"}, {}, origen="demo")
        eventos = eventos_de_escenario(esc, inicio + timedelta(hours=i))
        storage.insertar_eventos(sid, eventos)
        storage.cerrar_sesion(sid)
        print(f"[OK] Sesión {esc:14s} {sid}  ({len(eventos)} eventos)")
    print(f"[OK] Conductor demo: {cid}")


if __name__ == "__main__":
    main()
