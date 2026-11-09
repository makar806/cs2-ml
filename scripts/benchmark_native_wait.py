"""Saved-image A/B of CPU wait policy. Never captures screen or controls input."""
from dataclasses import asdict
import argparse
import json
from pathlib import Path
import statistics
import time
import cv2
from mvp_native_trt import NativeTensorRT

ROOT=Path(__file__).resolve().parents[1]

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--wait-game',action='store_true')
    parser.add_argument('--label',choices=['fps120','unlimited'],default=None);args=parser.parse_args()
    game=None
    if args.wait_game:
        from profile_stutter import wait_for_game
        from mvp_windows import find_window,foreground
        game=wait_for_game(find_window,foreground)
    paths=sorted((ROOT/'data/pose-v7/images/val').glob('*'))[:12]
    images=[cv2.resize(cv2.imread(str(path)),(1920,1080)) for path in paths]
    if not images:raise RuntimeError('No saved images')
    result=[];reference=None
    for blocking in (False,True,True,False):
        model=NativeTensorRT(ROOT/'runs/pose-v7/weights/best.engine',960,blocking_wait=blocking)
        try:
            for _ in range(3):model.predict(images[0])
            predictions=[[asdict(d) for d in model.predict(image)] for image in images]
            if reference is None:reference=predictions
            if predictions!=reference:raise RuntimeError('Wait mode changed detections')
            durations=[];cpu=time.thread_time();start=time.perf_counter()
            for i in range(60):
                if game and not foreground(game):raise RuntimeError('Game lost focus; comparison invalid')
                before=time.perf_counter();model.predict(images[i%len(images)])
                durations.append((time.perf_counter()-before)*1000)
            result.append(dict(blocking_wait=blocking,samples=len(durations),
                median_ms=statistics.median(durations),p95_ms=sorted(durations)[57],
                wall_ms=(time.perf_counter()-start)*1000,cpu_ms=(time.thread_time()-cpu)*1000,
                detections_identical=True))
        finally:model.close()
    output=ROOT/'runs/mvp-v3.5';output.mkdir(exist_ok=True)
    filename=f'native-wait-ab-{args.label}.json' if args.label else ('native-wait-ab-game.json' if game else 'native-wait-ab.json')
    (output/filename).write_text(json.dumps(result,indent=2))
    print(json.dumps(result,indent=2))
