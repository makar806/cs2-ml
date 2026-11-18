"""Verify the branch package and, optionally, a new V18 replay."""

import argparse
from hashlib import sha256
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / 'data'
EVIDENCE = ROOT / 'evidence'


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def fingerprint(path):
    return sha256(path.read_bytes()).hexdigest()


def verify_replay(expected, path):
    replay = read(path)
    if replay['weights_sha256'] != expected['weights_sha256']:
        raise ValueError('Replay used different weights')
    if replay['benchmark_sha256'] != expected['benchmark_sha256']:
        raise ValueError('Replay used different evaluation data')
    keys = ('mode', 'strict_first', 'confirmed_while_partial', 'any_while_partial', 'false_frames', 'empty_frames')
    case_keys = ('case', 'earliest_frame', 'confirmed_frame', 'strict_first', 'false_frames')
    for name, old in expected['results'].items():
        new = replay['results']['single_frame' if name == 'v18' else name]
        if any(old[key] != new[key] for key in keys):
            raise ValueError(f'The summary changed for {name}')
        if [tuple(item[k] for k in case_keys) for item in old['cases']] != [
            tuple(item[k] for k in case_keys) for item in new['cases']
        ]:
            raise ValueError(f'Per-scene results changed for {name}')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--replay', type=Path, help='Compare a V18 CUDA motion replay with published evidence')
    args = parser.parse_args()
    current = read(EVIDENCE/'motion-comparison.json')
    if fingerprint(DATA/'fresh.json') != current['benchmark_sha256']:
        raise ValueError('The fresh ground truth does not match the original experiment')
    vs19 = read(EVIDENCE/'v19-comparison.json')
    for name, report in (('v18', current), ('v19', vs19)):
        if fingerprint(ROOT/f'checkpoints/{name}.pt') != report['weights_sha256']:
            raise ValueError(f'Weight hash mismatch for {name}')
    sources = read(DATA/'sources.json')
    lookup = {row['case']: row['match_id'] for row in sources}
    if len(lookup) != len(sources) or len(set(lookup.values())) != len(sources):
        raise ValueError('Video cases or source matches overlap')
    fresh, prior, calibration = (read(DATA/name) for name in ('fresh.json', 'previous.json', 'calibration.json'))
    groups = [{r['case'] for r in rows} for rows in (fresh, prior['appearances']+prior['empty_windows'], calibration)]
    if set.union(*groups) != set(lookup) or sum(map(len, groups)) != len(lookup):
        raise ValueError('The manifest and the independent splits disagree')
    if [len(fresh), len(prior['appearances']), len(calibration)] != [9, 17, 83]:
        raise ValueError('An annotation is missing')
    for row in sources:
        short = row['excerpt']
        path = DATA/short['file']
        if path.stat().st_size != short['bytes'] or fingerprint(path) != short['sha256']:
            raise ValueError(f'Corrupt excerpt {path}')
    print(f"Verified two checkpoints and {len(sources)} excerpts with separate source matches.")
    if args.replay:
        verify_replay(current, args.replay)
        print('V18 motion experiment and its per-scene results reproduced.')


if __name__ == '__main__':
    main()
