import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import mvp_runtime as runtime


def test_explicit_pytorch_backend_loads_portable_weights(monkeypatch, tmp_path):
    monkeypatch.setattr(runtime, "ROOT", tmp_path)
    calls = []

    class Model:
        def __init__(self, path, task):
            calls.append((path, task))

        def predict(self, image, **kwargs):
            calls.append(kwargs)

    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(set_num_threads=lambda n: None))
    monkeypatch.setitem(sys.modules, "ultralytics", SimpleNamespace(YOLO=Model))
    model, backend = runtime.load_model("pytorch", 960)
    assert isinstance(model, Model)
    assert backend == "PyTorch FP32"
    assert calls[0] == (str(tmp_path / "runs/pose-v7/weights/best.pt"), "pose")
    assert len(calls) == 4


@pytest.mark.parametrize("backend", ["auto", "native"])
def test_native_without_verified_engine_fails_closed(monkeypatch, tmp_path, backend):
    monkeypatch.setattr(runtime, "ROOT", tmp_path)
    with pytest.raises(RuntimeError, match="requires a verified engine and weights"):
        runtime.load_model(backend, 960)
