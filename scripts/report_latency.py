"""Read-only session latency distributions. Never mixes clock domains."""
import argparse,json,math
from pathlib import Path

def distribution(values):
    x=sorted(v for v in values if isinstance(v,(int,float)) and math.isfinite(v) and v>=0)
    if not x:return dict(n=0)
    def percentile(p):
        i=(len(x)-1)*p;lo=int(i);hi=min(lo+1,len(x)-1)
        return x[lo]+(x[hi]-x[lo])*(i-lo)
    return dict(n=len(x),p50=percentile(.5),p95=percentile(.95),p99=percentile(.99),max=x[-1])

def rows(path):
    if not path.exists():return []
    result=[]
    with path.open(encoding='utf-8') as f:
        for line in f:
            try:result.append(json.loads(line))
            except json.JSONDecodeError:pass # unfinished buffered final line
    return result

def report(directory):
    directory=Path(directory);frames=rows(directory/'frames.jsonl');actions=rows(directory/'actions.jsonl')
    metrics={k:distribution([r.get(k) for r in frames]) for k in ('capture_ms','predict_ms','latency_ms','visual_feed_ms','visual_tracking_ms')}
    metrics['capture_to_input_ms']=distribution([r.get('capture_to_input_ms') for r in actions])
    metrics['input_api_ms']=distribution([r.get('api_ms') for r in rows(directory/'input-timing.jsonl')])
    metrics['model_source_interval_ms']=distribution([(b['captured_at']-a['captured_at'])*1000 for a,b in zip(frames,frames[1:]) if 'captured_at' in a and 'captured_at' in b])
    metrics['source_gaps_over_500ms']=dict(n=sum(b['captured_at']-a['captured_at']>.5 for a,b in zip(frames,frames[1:]) if 'captured_at' in a and 'captured_at' in b))
    return dict(session=str(directory.resolve()),units='milliseconds',metrics=metrics,note='Source intervals use monotonic capture times. Frame logs describe processed/published frames, not every captured frame. Large source gaps may include user pauses/focus changes; they are not game freezes. Feed costs can repeat cached samples. No game frametimes or head ground truth are inferred.')

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('session');parser.add_argument('--out');a=parser.parse_args()
    result=json.dumps(report(a.session),ensure_ascii=False,indent=2)
    if a.out:Path(a.out).write_text(result,encoding='utf-8')
    else:print(result)
