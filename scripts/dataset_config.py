from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]


def resolve_dataset(source):
    source = Path(source).resolve()
    data = yaml.safe_load(source.read_text(encoding="utf-8"))
    base = Path(data.get("path", "."))
    if not base.is_absolute():
        base = (source.parent / base).resolve()
    if not base.is_dir():
        raise FileNotFoundError(f"Dataset directory does not exist: {base}")
    data["path"] = str(base)
    output = ROOT / "runs/dataset-resolved.yaml"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
    return output
