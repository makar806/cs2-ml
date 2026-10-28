"""Calibrate from an existing screenshot; never captures or controls the PC."""

import argparse
import json
import time
from pathlib import Path

import cv2
from mvp_killfeed import VisualKillFeed, calibrate, highlighted_rows

ROOT = Path(__file__).resolve().parents[1]

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("image")
    p.add_argument("--row", type=int, default=0)
    a = p.parse_args()
    image = cv2.imread(a.image)
    if image is None:
        raise FileNotFoundError(a.image)
    template = calibrate(image, a.row)
    output = ROOT / "work/killfeed"
    output.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output / "self-name.png"), template)
    sensor = VisualKillFeed(template)
    times = []
    for _ in range(100):
        start = time.perf_counter()
        rows = sensor.own_rows(image)
        times.append((time.perf_counter() - start) * 1000)
    report = dict(
        source=str(Path(a.image).resolve()),
        template_shape=template.shape,
        highlighted_rows=highlighted_rows(image),
        matched_rows=[dict(box=r["box"], name_score=r["name_score"]) for r in rows],
        median_ms=sorted(times)[50],
        p95_ms=sorted(times)[95],
        note="Saved-image microbenchmark, not live CS2 end-to-end latency or accuracy.",
    )
    (output / "calibration.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
