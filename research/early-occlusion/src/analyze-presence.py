"""Describe misses, paired outcomes, latency in frames and annotation sensitivity."""

import argparse
import json
from math import sqrt
from pathlib import Path

import numpy as np
import torch

from protocol import ROOT, fingerprint
from replay import Mode, overlap, select


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cache', type=Path, required=True)
    parser.add_argument('--report', type=Path, required=True)
    parser.add_argument('--annotations', type=Path, default=ROOT/'data/next/benchmark.json')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    cache = torch.load(args.cache, weights_only=False)
    report = json.loads(args.report.read_text())
    annotations = json.loads(args.annotations.read_text())
    if cache['signature'] != report['signature'] or fingerprint(args.annotations) != report['signature']['annotations']:
        raise ValueError('Mismatched predictions, results or annotations')
    observations = cache['observations']
    result = dict(signature=report['signature'], first_frame_signal=[], settings={}, negatives_by_match={})
    for row in annotations['appearances']:
        first = observations[row['case']][row['first']]
        item = dict(case=row['case'], frame=row['first'], visible_box=row['box'], signal={})
        for name, (boxes,scores,_) in first.items():
            quality = np.asarray([overlap(b,row['box']) for b in boxes])
            allowed = quality >= .3
            best = int(np.argmax(np.where(allowed,scores,-1))) if allowed.any() else None
            item['signal'][name] = dict(best_presence_with_iou03=float(scores[best]) if best is not None else None,
                box=boxes[best].tolist() if best is not None else None,
                iou=float(quality[best]) if best is not None else None,
                maximum_iou_above_cache_floor=float(quality.max(initial=0)))
        result['first_frame_signal'].append(item)
    for setting,entries in report['results'].items():
        paired = np.asarray([int(b['strict_first'])-int(a['strict_first']) for a,b in
                             zip(entries['v18']['cases'],entries['candidate']['cases'])])
        result['settings'][setting] = dict(first_frame_gains=int((paired==1).sum()),
            first_frame_losses=int((paired==-1).sum()), paired_difference=float(paired.mean()),
            models={})
        for name,entry in entries.items():
            delays = {}
            for stage in ['earliest','confirmed']:
                values = [r[f'{stage}_delay_ms'] for r in entry['cases'] if r[f'{stage}_delay_ms'] is not None]
                leads = [(e['full_frame']-r[f'{stage}_frame'])*1000/32 for r,e in zip(entry['cases'],annotations['appearances'])
                         if not e['censored'] and r[f'{stage}_frame'] is not None]
                delays[stage] = dict(detected=len(values), missed=len(entry['cases'])-len(values),
                    median_delay_ms=float(np.median(values)) if values else None,
                    max_delay_ms=max(values) if values else None,
                    median_lead_to_full_ms=float(np.median(leads)) if leads else None)
            result['settings'][setting]['models'][name] = delays
            n, successes, z = len(entry['cases']), entry['strict_first'], 1.959963984540054
            p = successes/n
            center = (p+z*z/(2*n))/(1+z*z/n)
            radius = z*sqrt(p*(1-p)/n+z*z/(4*n*n))/(1+z*z/n)
            delays['first_frame_wilson_95'] = [center-radius,center+radius]
            threshold = entry['mode']['floor']
            result['negatives_by_match'].setdefault(setting,{})[name] = [dict(case=r['case'],
                false_boxes=sum(len(select(observations[r['case']][i][name],Mode(threshold)))
                                for i in range(r['start'],r['stop']+1))) for r in annotations['empty_windows']]
    result['limitations'] = ['Six episode/match pairs cannot establish equivalence; marginal Wilson intervals are descriptive for this manually selected pilot.',
        'Raw signal audit omits anchors below 0.0002; null is not evidence that geometry is absent.',
        'Delay summaries condition on successful detections; misses are shown separately.']
    args.output.write_text(json.dumps(result,indent=2)+'\n')


if __name__ == '__main__':
    main()
