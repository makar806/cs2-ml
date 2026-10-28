import hashlib
import json
import os
from pathlib import Path

os.environ["YOLO_AUTOINSTALL"] = "false"

ROOT = Path(__file__).resolve().parents[1]


def main():
    import numpy as np
    from mvp_native_trt import NativeTensorRT
    from ultralytics import YOLO

    weights = ROOT / "models/best.pt"
    engine = Path(
        YOLO(str(weights), task="pose").export(
            format="engine",
            imgsz=(736, 1280),
            quantize=16,
            dynamic=True,
            batch=1,
            workspace=1,
            device=0,
            simplify=True,
        )
    )
    manifest = ROOT / "models/runtime.json"
    if manifest.exists():
        manifest.unlink()
    runtime = NativeTensorRT(engine, size=960)
    try:
        for _ in range(3):
            runtime.predict(np.zeros((720, 1280, 3), np.uint8))
    finally:
        runtime.close()
    metadata = {
        "engine_approved": True,
        "engine_sha256": hashlib.sha256(engine.read_bytes()).hexdigest(),
        "weights_sha256": hashlib.sha256(weights.read_bytes()).hexdigest(),
        "validation": "Local engine loading, output decoding and warmup only; not accuracy evaluation.",
    }
    manifest.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"Engine ready: {engine}")


if __name__ == "__main__":
    main()
