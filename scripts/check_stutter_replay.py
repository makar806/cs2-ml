"""Smoke-test only the saved-image worker; no ETW, capture or game input."""
import json
from pathlib import Path
import statistics
import time
from profile_stutter import ROOT,own_worker

if __name__=='__main__':
    output=ROOT/'runs/stutter-profile'/('replay-smoke-'+time.strftime('%Y%m%d-%H%M%S'))
    output.mkdir(parents=True)
    process,stop,packets,messages=own_worker('inference_replay',output)
    try:time.sleep(2)
    finally:
        stop.set();process.join(5)
        if process.is_alive():process.terminate();process.join(2)
        for queue in (packets,messages):queue.close();queue.cancel_join_thread()
    rows=[json.loads(line) for line in (output/'api.jsonl').read_text().splitlines()]
    if process.exitcode or not rows:raise RuntimeError('Replay worker smoke failed: '+str(output))
    print(json.dumps(dict(output=str(output),exitcode=process.exitcode,samples=len(rows),
        median_ms=statistics.median(row['ms'] for row in rows),stages=rows[-1]['stages'])))
