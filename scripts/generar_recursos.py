"""Crea carpetas, sonidos de alarma y archivos CSV de ejemplo.

Uso: python scripts/generar_recursos.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.alarma import generar_sonidos  # noqa: E402
from core.config import DATA_DIR, EJEMPLOS_DIR, asegurar_carpetas  # noqa: E402
from core.impacto import generar_csv_ejemplo  # noqa: E402

if __name__ == "__main__":
    asegurar_carpetas()
    print(f"[OK] Carpetas de trabajo en {DATA_DIR}")
    for clave, ruta in generar_sonidos(forzar=True).items():
        print(f"[OK] Sonido {clave}: {ruta}")
    print("[OK]", generar_csv_ejemplo(EJEMPLOS_DIR / "acelerometro_choque.csv", choque=True))
    print("[OK]", generar_csv_ejemplo(EJEMPLOS_DIR / "acelerometro_sin_choque.csv", choque=False))
