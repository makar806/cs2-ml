"""Rebuild the compact published videos from the named OpenCS2 originals."""

import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import re
import subprocess

import cv2
import numpy as np

from protocol import DATA, fingerprint, load, materialize


def benchmark_ranges():
    cases = {}
    for row in load('fresh.json'):
        start, stop = row['empty']
        cases[row['case']] = set(range(start-1, stop+1)) | set(range(row['first']-1, row['last_partial']+1))
    earlier = load('previous.json')
    for row in earlier['appearances']:
        cases[row['case']] = set(range(row['first']-5, row['first']+21))
    for row in earlier['empty_windows']:
        cases.setdefault(row['case'], set()).update(range(row['start']-1, row['stop']+1))
    for row in load('calibration.json'):
        cases.setdefault(row['case'], set()).update((row['frame']-1, row['frame']))
    return cases


def keyframes(ffmpeg, path):
    log = subprocess.run([ffmpeg, '-hide_banner', '-skip_frame', 'nokey', '-i', str(path),
                          '-map', '0:v:0', '-an', '-vf', 'showinfo', '-f', 'null', '-'],
                         check=True, capture_output=True, text=True, encoding='utf-8', errors='replace')
    before = [round(float(x)*32) for x in re.findall(r'pts_time:([\d.]+).*?iskey:1', log.stderr)]
    if not before:
        raise IOError(f'Could not identify keyframes in {path}')
    return before


def verify_pixels(original, clip, required, first):
    source, reduced = cv2.VideoCapture(str(original)), cv2.VideoCapture(str(clip))
    try:
        if reduced.get(cv2.CAP_PROP_FPS) != 32:
            raise ValueError(f'Incorrect frame rate for {clip}')
        last = -2
        for frame in sorted(required):
            if frame != last+1:
                source.set(cv2.CAP_PROP_POS_FRAMES, frame)
                reduced.set(cv2.CAP_PROP_POS_FRAMES, frame-first)
            ok_a, expected = source.read()
            ok_b, actual = reduced.read()
            if not (ok_a and ok_b and np.array_equal(expected, actual)):
                raise ValueError(f'Excerpt differs from {original} at source frame {frame}')
            last = frame
    finally:
        source.release()
        reduced.release()


def excerpt(row, frames, ffmpeg, cache):
    source, start = materialize(dict(row, excerpt=None), cache)
    assert start == 0
    keys = keyframes(ffmpeg, source)
    first = max(key for key in keys if key <= min(frames))
    end = min((key for key in keys if key > max(frames)), default=max(frames)+9)
    target = DATA / 'media' / f"{row['case']}.mp4"
    subprocess.run([ffmpeg, '-hide_banner', '-loglevel', 'error', '-nostdin', '-y',
                    '-ss', f'{first/32:.8f}', '-i', str(source), '-map', '0:v:0',
                    '-t', f'{(end-first)/32:.8f}', '-an', '-c:v', 'copy',
                    '-avoid_negative_ts', 'make_zero', str(target)], check=True)
    verify_pixels(source, target, frames, first)
    return dict(row, excerpt=dict(file=target.relative_to(DATA).as_posix(), first_frame=first,
                                  bytes=target.stat().st_size, sha256=fingerprint(target),
                                  verified_frames=len(frames)))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--ffmpeg', default='ffmpeg')
    parser.add_argument('--original-cache', type=Path, required=True)
    args = parser.parse_args()
    sources = load('sources.json')
    ranges = benchmark_ranges()
    (DATA/'media').mkdir(exist_ok=True)
    cv2.setNumThreads(1)
    with ThreadPoolExecutor(max_workers=3) as pool:
        results = list(pool.map(lambda row: excerpt(row, ranges[row['case']],
                                                    args.ffmpeg, args.original_cache), sources))
    (DATA/'sources.json').write_text(json.dumps(results, indent=2)+'\n', encoding='utf-8')
    print(f"{len(results)} excerpts, {sum(r['excerpt']['bytes'] for r in results)} bytes, "
          f"{sum(r['excerpt']['verified_frames'] for r in results)} verified frames.")


if __name__ == '__main__':
    main()
