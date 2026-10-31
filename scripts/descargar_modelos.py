"""Descarga los modelos preentrenados antes de la clase.

Uso (desde la raíz del proyecto):
    python scripts/descargar_modelos.py          # FaceLandmarker (obligatorio, ~3,6 MB)
    python scripts/descargar_modelos.py --vit    # además el ViT de expresiones (~350 MB)
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.config import FACE_MODEL_ID, LANDMARKER_PATH, LANDMARKER_URL  # noqa: E402
from core.landmarks import descargar_modelo  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--vit", action="store_true", help="Descargar también el modelo ViT de expresiones")
    args = parser.parse_args()

    try:
        ruta = descargar_modelo()
        print(f"[OK] FaceLandmarker: {ruta} ({ruta.stat().st_size / 1e6:.1f} MB)")
    except Exception as exc:
        print(f"[ERROR] No se pudo descargar FaceLandmarker: {exc}")
        print(f"        Descárguelo a mano desde:\n        {LANDMARKER_URL}\n        y guárdelo como {LANDMARKER_PATH}")
        sys.exit(1)

    if args.vit:
        print(f"Descargando {FACE_MODEL_ID} (puede tardar varios minutos)...")
        from core.faces import classifier
        classifier()
        print(f"[OK] Modelo de expresiones listo: {FACE_MODEL_ID}")


if __name__ == "__main__":
    main()
