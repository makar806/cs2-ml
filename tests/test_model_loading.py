import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import mvp_runtime as runtime


def test_auto_without_machine_specific_engine_loads_portable_weights(monkeypatch, tmp_path):
    monkeypatch.setattr(runtime, "ROOT", tmp_path)
    calls = []

    class Model:
        def __init__(self, path, task):
            calls.append((path, task))

        def predict(self, image, **kwargs):
            calls.append(kwargs)

    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(set_num_threads=lambda n: None))
    monkeypatch.setitem(sys.modules, "ultralytics", SimpleNamespace(YOLO=Model))
    model, backend = runtime.load_model("auto", 960)
    assert isinstance(model, Model)
    assert backend == "PyTorch FP32"
    assert calls[0] == (str(tmp_path / "models/best.pt"), "pose")
    assert len(calls) == 4


def test_native_without_verified_engine_explains_export_step(monkeypatch, tmp_path):
    monkeypatch.setattr(runtime, "ROOT", tmp_path)
    with pytest.raises(RuntimeError, match="export_engine.py"):
        runtime.load_model("native", 960)
