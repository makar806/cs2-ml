"""Fit movement timing from finalized, first-shot-success stationary demonstrations."""

import argparse
import hashlib
import json
import math
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def samples_from_session(directory):
    summary = json.loads((directory / "summary.json").read_text(encoding="utf-8"))
    if (
        summary["settings"].get("test_session")
        or summary["settings"]["mode"] != "stationary"
        or summary.get("recording_error")
        or summary["accepted_events"] != summary["written_events"]
    ):
        return []
    rows = [json.loads(line) for line in (directory / "events.jsonl").read_text().splitlines()]
    samples = []
    for trial in summary["trials"]:
        if trial["outcome"] != "hit" or not trial["shots"] or not trial["shots"][0]["hit"]:
            continue
        initial = trial["camera_start"]
        yaw = trial["target_start"][0] - initial[0]
        vector = [math.atan2(math.sin(yaw), math.cos(yaw)), trial["target_start"][1] - initial[1]]
        den = sum(v * v for v in vector)
        angle = math.degrees(math.sqrt(den))
        if angle < 1 or trial["path_degrees"] / angle > 1.5:
            continue
        progress = []
        for event in rows:
            if event["event"] != "mouse" or event.get("trial") != trial["id"]:
                continue
            if event["t"] > trial["shots"][0]["t"]:
                continue
            difference = event["camera_angles"][0] - initial[0]
            delta = [
                math.atan2(math.sin(difference), math.cos(difference)),
                event["camera_angles"][1] - initial[1],
            ]
            progress.append((event["t"], sum(delta[i] * vector[i] for i in range(2)) / den))
        t10 = next((t for t, p in progress if p >= 0.1), None)
        t90 = next((t for t, p in progress if p >= 0.9), None)
        if t10 is None or t90 is None or t90 <= t10:
            continue
        duration = (t90 - t10) / 0.506728
        if 0.12 <= duration <= 0.9:
            samples.append(
                dict(session=directory.name, trial=trial["id"], duration=duration, angle=angle)
            )
    return samples


def fit(recordings):
    samples = []
    sources = []
    for directory in sorted(Path(recordings).iterdir()):
        if not directory.is_dir() or not (directory / "summary.json").exists():
            continue
        part = samples_from_session(directory)
        if part:
            samples.extend(part)
            sources.append(
                dict(
                    session=directory.name,
                    samples=len(part),
                    events_sha256=hashlib.sha256(
                        (directory / "events.jsonl").read_bytes()
                    ).hexdigest(),
                )
            )
    if len(samples) < 5:
        raise ValueError("Need at least 5 usable first-shot stationary trajectories")
    return dict(
        schema_version=1,
        duration_seconds=statistics.median(x["duration"] for x in samples),
        reference_angle_degrees=statistics.median(x["angle"] for x in samples),
        sample_count=len(samples),
        sources=sources,
        method="Median minimum-jerk duration estimated from 10–90% primary movement; timing calibration, not a neural policy",
        excluded="Missed-first-shot, interrupted, moving-target and test sessions; clicks are not imitated",
        samples=samples,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--recordings", default=str(ROOT / "runs/aim-recorder"))
    parser.add_argument("--output", default=str(ROOT / "configs/aim-profile.json"))
    args = parser.parse_args()
    result = fit(args.recordings)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(
        json.dumps({k: v for k, v in result.items() if k not in ("sources", "samples")}, indent=2)
    )
