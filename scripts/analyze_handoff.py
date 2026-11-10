"""Recompute public evidence without extracting private data or requiring CS2."""
import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import io
import json
import math
from pathlib import Path
import statistics
import zipfile


def distribution(values):
    values=sorted(v for v in values if isinstance(v,(int,float)) and math.isfinite(v))
    if not values:
        return None
    def q(p):
        i=(len(values)-1)*p
        a,b=math.floor(i),math.ceil(i)
        return values[a]+(values[b]-values[a])*(i-a)
    return dict(n=len(values),p50=q(.5),p95=q(.95),p99=q(.99),max=max(values),
                above_100ms=sum(v>100 for v in values))


def analyze(archive):
    report=dict(archive_sha256=hashlib.sha256(Path(archive).read_bytes()).hexdigest(),
                stages=[],live={},alignments=[],missing_settings=True,
                note='Historical measurements; no new game performance measurement')
    with zipfile.ZipFile(archive) as z:
        names=set(z.namelist())
        def read(path):
            return json.loads(z.read(path))
        def lines(path):
            if path not in names:
                return []
            with z.open(path) as source:
                return [json.loads(line) for line in source if line.strip()]
        presentations={}
        for name in sorted(names):
            if not (name.startswith('runs/stutter-profile/') and name.count('/')==3 and name.endswith('/summary.json')):
                continue
            run=name.split('/')[2]
            for saved in read(name):
                path=name.rsplit('/',1)[0]+'/'+saved['stage']+'/present.csv'
                rows=list(csv.DictReader(io.StringIO(z.read(path).decode('utf-8-sig'))))
                groups=defaultdict(list)
                for row in rows:
                    groups[row.get('SwapChainAddress','main')].append(row)
                rows=max(groups.values(),key=len)
                values=[float(r[saved['field']]) for r in rows if r.get(saved['field'])]
                values=[v for v in values if math.isfinite(v) and v>0]
                stats=distribution(values)
                expected={'p50':'median_ms','p95':'p95_ms','p99':'p99_ms','max':'max_ms','above_100ms':'above_100ms','n':'n'}
                agrees=all(abs(stats[k]-saved[v])<1e-7 for k,v in expected.items())
                report['stages'].append(dict(run=run,stage=saved['stage'],frametime_ms=stats,
                    game_fps=len(values)*1000/sum(values),matches_saved_summary=agrees,
                    ram_peak_percent=saved.get('ram_peak_percent'),
                    foreground_fraction=saved.get('foreground_fraction'),versions=saved.get('versions'),
                    capture_target_hz=saved.get('capture_target_hz')))
                if saved['stage']=='live':
                    presentations[run]=(rows,saved['field'])
        selected=('20261102-013850','20261108-231801','20261108-232528',
                  '20261108-232957','20261109-182732','20261109-182950',
                  '20261109-184844','20261109-201421')
        caches={}
        for run in selected:
            base=f'runs/live-mvp/{run}/'
            frames=lines(base+'frames.jsonl')
            controls=lines(base+'control.jsonl')
            inputs=lines(base+'input-timing.jsonl')
            actions=lines(base+'actions.jsonl')
            summary=read(base+'summary.json')
            processing_intervals=[(b['time']-a['time'])*1000 for a,b in zip(frames,frames[1:]) if b['time']>a['time']]
            # IDs count captured frames, while these rows count processed detections.
            segments=[(b['id']-a['id'],b['captured_at']-a['captured_at']) for a,b in zip(frames,frames[1:])
                      if b['id']>a['id'] and b['captured_at']>a['captured_at'] and a.get('capture_creations')==b.get('capture_creations')]
            offsets=[r['time']-r['api_ended'] for r in inputs if r.get('api_ended') is not None]
            s=dict(version=summary.get('version'),frames=len(frames),
                   detector_interval_ms=distribution(processing_intervals),
                   detector_session_throughput_hz=1000/statistics.mean(processing_intervals) if processing_intervals else None,
                   capture_session_rate_estimate_hz=sum(a for a,b in segments)/sum(b for a,b in segments) if segments else None,
                   rate_caveat='Whole logged session including pauses; active-only rates cannot be recovered from these logs',
                   inference_ms=distribution([f.get('predict_ms') for f in frames]),
                   capture_call_ms=distribution([f.get('capture_ms') for f in frames]),
                   capture_to_inference_end_ms=distribution([f.get('latency_ms') for f in frames]),
                   capture_to_input_ms=distribution([a.get('capture_to_input_ms') for a in actions]),
                   input_api_ms=distribution([i.get('api_ms') for i in inputs]),
                   input_lock_ms=distribution([i.get('lock_ms') for i in inputs]),
                   control_reasons=dict(Counter(c.get('reason') for c in controls)),
                   visual_sequence_max=max((f.get('visual_kill_sequence',0) for f in frames),default=0),
                   clock_offset_range_ms=(max(offsets)-min(offsets))*1000 if offsets else None,
                   exact_queue_timing_available=all('packet_ready_at' in f for f in frames) if frames else False)
            report['live'][run]=s
            caches[run]=(frames,controls,inputs,offsets)
        pairs={'20261108-231747':'20261108-231801','20261108-232454':'20261108-232528',
               '20261108-233206':'20261108-232957','20261109-182730':'20261109-182732'}
        for trace,run in pairs.items():
            rows,field=presentations[trace]
            frames,controls,inputs,offsets=caches[run]
            if not rows[0].get('QPCTime'):
                report['alignments'].append(dict(trace=trace,live=run,quality='relative PresentMon clock; exact alignment unavailable'))
                continue
            # This recorder exported QPC in seconds. Verify units from the
            # independent relative column before comparing perf_counter values.
            dt=float(rows[-1]['TimeInSeconds'])-float(rows[0]['TimeInSeconds'])
            dq=float(rows[-1]['QPCTime'])-float(rows[0]['QPCTime'])
            if dt<=0 or abs(dt-dq)>.001:
                raise ValueError('Unexpected QPC units')
            events=[]
            for r in rows:
                duration=float(r[field])
                if duration<=100:
                    continue
                end=float(r['QPCTime']);start=end-duration/1000
                overlap=[i for i in inputs if i.get('api_started',math.inf)<end and i.get('api_ended',-math.inf)>start]
                active=[f for f in frames if start<=f['captured_at']<=end]
                events.append(dict(start_qpc=start,end_qpc=end,frametime_ms=duration,
                                   input_calls_overlapping=len(overlap),
                                   slow_input_calls_overlapping=sum(i.get('api_ms',0)>100 for i in overlap),
                                   detector_frames_captured_inside=len(active)))
            report['alignments'].append(dict(trace=trace,live=run,quality='shared QPC seconds for Present/input/capture; old control and queue timing incomplete',
                long_frames=events,correlation_is_not_causation=True))
        report['missing_settings']=not any(n.endswith('/settings.json') for n in names)
    return report


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--archive',type=Path,default=Path('evidence/offline-measurements.zip'))
    p.add_argument('--output',type=Path,default=Path('work/v332-measurements.json'))
    args=p.parse_args()
    report=analyze(args.archive)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(f"Recomputed {len(report['stages'])} stages; mismatches: {sum(not s['matches_saved_summary'] for s in report['stages'])}")
