"""Compare V18 confidence with a causal motion score on frozen new matches."""

import argparse
from dataclasses import asdict, dataclass
from itertools import product
import json
from pathlib import Path
from time import perf_counter_ns

import cv2
import numpy as np
import torch

from motion import box_features, changes, prepare
from protocol import DATA, ROOT, associated, boxes_for, fingerprint, load, materialize, network, raw_prediction


@dataclass(frozen=True)
class Mode:
    floor: float
    weak_floor: float = 1.
    motion: float = 0.


BASE = Mode(.13)


def read_frames(path, frames, first_frame=0):
    """Yield pairs of adjacent source frames in increasing order."""
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened() or capture.get(cv2.CAP_PROP_FPS) != 32:
        capture.release()
        raise OSError(path)
    last_index = -2
    last = None
    try:
        for index in sorted(set(frames)):
            if index != last_index + 1:
                if index <= first_frame:
                    raise ValueError(f'Missing the preceding frame of {index} in {path}')
                capture.set(cv2.CAP_PROP_POS_FRAMES, index - 1 - first_frame)
                ok, last = capture.read()
                if not ok:
                    raise IOError((path, index-1))
            ok, image = capture.read()
            if not ok or image.shape != (720, 1280, 3):
                raise IOError((path, index))
            yield index, last, image
            last_index, last = index, image
    finally:
        capture.release()


def observe(model, previous, current, use_motion):
    raw = raw_prediction(model, current)
    output, shape, scale, pad = raw
    candidates = output[:, :, output[0, 6] >= .0002]
    boxes = boxes_for(candidates, shape, scale, pad)
    boxes /= 1.5
    confidence = candidates[0, 6]
    start = perf_counter_ns()
    change = changes(prepare(previous[:, 160:1120]), prepare(current[:, 160:1120])) if use_motion else None
    strength = np.zeros(len(boxes), dtype=np.float32)
    if change is not None:
        local, fraction, ring, _, coverage = box_features(change, boxes).T
        width, height = (boxes[:, 2:] - boxes[:, :2]).T
        compact = (width >= 3) & (height >= 6) & (width <= 70) & (height <= 130)
        supported = compact & (coverage >= .95) & (fraction >= .13) & (local > ring + 1)
        strength[supported] = (local[supported] - .5*ring[supported])*fraction[supported]
    auxiliary_ms = (perf_counter_ns() - start)/1e6
    boxes[:, [0, 2]] += 160
    return boxes, confidence, strength, auxiliary_ms, change is not None


def select(observation, mode):
    boxes, confidence, strength = observation[:3]
    allowed = (confidence >= mode.floor) | ((confidence >= mode.weak_floor) & (strength >= mode.motion))
    if not allowed.any():
        return []
    selected = boxes[allowed]
    xywh = selected.copy()
    xywh[:, 2:] -= xywh[:, :2]
    scores = confidence[allowed]
    kept = np.asarray(cv2.dnn.NMSBoxes(xywh.tolist(), scores.tolist(), 0., .7)).reshape(-1)[:300]
    return [(*selected[i].tolist(), float(scores[i]), 0) for i in kept]


def overlap(box, truth):
    a, b, c, d = box[:4]
    x, y, r, s = truth
    intersection = max(0., min(c, r)-max(a, x))*max(0., min(d, s)-max(b, y))
    return intersection / ((c-a)*(d-b) + (r-x)*(s-y) - intersection)


def fit(calibration):
    budget = sum(bool(select(observation, BASE)) for observation, truth in calibration if truth is None)
    modes = [Mode(floor) for floor in (.13, .18, .22, .28, .33)]
    modes += [Mode(*settings) for settings in product(
        (.13, .18, .22, .28, .33), (.0002, .002, .015, .03, .06), (2., 3.5, 5., 7.5))]
    results = []
    for mode in modes:
        matches = negatives = 0
        for observation, truth in calibration:
            boxes = select(observation, mode)
            if truth is None:
                negatives += bool(boxes)
            else:
                matches += any(overlap(box, truth) >= .30 for box in boxes)
        results.append(dict(mode=asdict(mode), correct=matches, false=negatives))
    permitted = [row for row in results if row['false'] <= budget]
    def best(filter_motion):
        valid = [row for row in permitted if bool(row['mode']['motion']) == filter_motion]
        picked = max(valid, key=lambda row: (row['correct'], -row['false'], row['mode']['weak_floor']))
        return Mode(**picked['mode'])
    return budget, best(False), best(True), results


def center_in(box, roi):
    cx, cy = (box[0]+box[2])/2, (box[1]+box[3])/2
    return roi[0] <= cx <= roi[2] and roi[1] <= cy <= roi[3]


def evaluate(episodes, empty_windows, modes, *, strict_first_in_pair):
    results = {}
    for name, mode in modes.items():
        examples = []
        for episode, frames in episodes:
            first, last = episode['first'], episode['last_partial']
            decoded = {frame: select(data, mode) for frame, data in frames.items()}
            accurate = any(overlap(box, episode['box']) >= .30 for box in decoded[first])
            eligible = {i: [box for box in decoded[i] if center_in(box, episode['roi'])]
                        for i in range(first, last+1)}
            if strict_first_in_pair:
                eligible[first] = [box for box in eligible[first] if overlap(box, episode['box']) >= .30]
            earliest = next((i for i, items in eligible.items() if items), None)
            confirmed = next((i for i in range(first+1, last+1)
                              if any(associated(a, b) for a in eligible[i-1] for b in eligible[i])), None)
            examples.append(dict(case=episode['case'], first=first, strict_first=accurate,
                                 earliest_frame=earliest, confirmed_frame=confirmed,
                                 first_boxes=decoded[first]))
        negatives = [dict(window, false_frames=sum(bool(select(data, mode)) for data in frames.values()),
                          empty_frames=len(frames)) for window, frames in empty_windows]
        for row in examples:
            same_case = [window for window in negatives if window['case'] == row['case']]
            row['false_frames'] = sum(window['false_frames'] for window in same_case)
            row['empty_frames'] = sum(window['empty_frames'] for window in same_case)
        results[name] = dict(mode=asdict(mode), cases=examples, strict_first=sum(row['strict_first'] for row in examples),
            any_while_partial=sum(row['earliest_frame'] is not None for row in examples),
            confirmed_while_partial=sum(row['confirmed_frame'] is not None for row in examples),
            false_frames=sum(row['false_frames'] for row in negatives),
            empty_frames=sum(row['empty_frames'] for row in negatives))
    return results


def calibration_observations(model, sources, additions, cache):
    by_case = {}
    for row in additions:
        box = row['box_source']
        by_case.setdefault(row['case'], {})[row['frame']] = box['box'] if box else None
    calibration = []
    for case, labels in by_case.items():
        path, start = materialize(sources[case], cache)
        for frame, last, current in read_frames(path, labels, start):
            calibration.append((observe(model, last, current, True), labels[frame]))
    return calibration


def benchmark_inputs(benchmark):
    if benchmark == 'previous':
        earlier = load('previous.json')
        return earlier['appearances'], earlier['empty_windows']
    fresh = load('fresh.json')
    return fresh, [dict(case=row['case'], start=row['empty'][0], stop=row['empty'][1]) for row in fresh]


def build_comparison(model, cache, confidence, use_motion, benchmark):
    sources = {row['case']: row for row in load('sources.json')}
    additions = load('calibration.json')
    used = {row['match_id'] for row in additions}
    appearances, negatives = benchmark_inputs(benchmark)
    modes = {'single_frame': Mode(confidence)}
    calibration_report = None
    if use_motion:
        calibration = calibration_observations(model, sources, additions, cache)
        budget, static, temporal, tuning = fit(calibration)
        print('Frozen modes:', static, temporal, 'budget', budget, flush=True)
        modes.update(static_calibrated=static, temporal=temporal)
        calibration_report = dict(budget=budget, positives=sum(b is not None for _, b in calibration),
                                  negatives=sum(b is None for _, b in calibration), sweep=tuning)
    by_case = {}
    for group, first, last in ((appearances, 'first', 'last_partial'), (negatives, 'start', 'stop')):
        for row in group:
            by_case.setdefault(row['case'], set()).update(range(row[first], row[last]+1))
    observations = {}
    for case, frames in by_case.items():
        match = sources[case]['match_id']
        if match in used:
            raise ValueError(f'Repeated calibration or holdout source {match}')
        used.add(match)
        path, media_start = materialize(sources[case], cache)
        observations[case] = {i: observe(model, last, current, use_motion) for i, last, current in
                              read_frames(path, frames, media_start)}
        print(case, len(frames), flush=True)
    def subrange(entries, first, last):
        return [(row, {i: observations[row['case']][i] for i in range(row[first], row[last]+1)}) for row in entries]
    episodes = subrange(appearances, 'first', 'last_partial')
    windows = subrange(negatives, 'start', 'stop')
    state = [row for frames in observations.values() for row in frames.values()]
    timing = dict(median_aux_ms=float(np.median([row[3] for row in state])),
                  p95_aux_ms=float(np.percentile([row[3] for row in state], 95)),
                  registered=sum(row[4] for row in state), frames=len(state))
    result = evaluate(episodes, windows, modes, strict_first_in_pair=benchmark=='fresh')
    return dict(results=result, timing=timing, calibration=calibration_report)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', default='v18')
    parser.add_argument('--confidence', type=float)
    parser.add_argument('--benchmark', choices=('fresh', 'previous'), default='fresh')
    parser.add_argument('--motion', action='store_true', help='Run the earlier causal motion experiment')
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--cache', type=Path, default=ROOT/'.cache')
    parser.add_argument('--output', type=Path, default=ROOT/'replay-result.json')
    args = parser.parse_args()
    local = ROOT / 'checkpoints' / f'{args.checkpoint}.pt'
    weights = local if args.checkpoint in ('v18', 'v19') else Path(args.checkpoint).expanduser().resolve()
    confidence = args.confidence if args.confidence is not None else (.19 if args.checkpoint == 'v19' else .13)
    if not .0002 <= confidence < 1:
        parser.error('Confidence must be at least .0002 and below 1')
    torch.set_num_threads(2)
    cv2.setNumThreads(1)
    with network(weights, torch.device(args.device)) as model:
        report = build_comparison(model, args.cache, confidence, args.motion, args.benchmark)
    report['weights_sha256'] = fingerprint(weights)
    report['benchmark_sha256'] = fingerprint(DATA/f'{args.benchmark}.json')
    args.output.write_text(json.dumps(report, indent=2)+'\n', encoding='utf-8')
    summary = {name: {k: v for k, v in row.items() if k != 'cases'} for name, row in report['results'].items()}
    print(json.dumps(dict(summary=summary, timing=report['timing']), indent=2))


if __name__ == '__main__':
    main()
