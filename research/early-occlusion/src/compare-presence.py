"""Compare fixed-geometry presence heads at frozen thresholds and equal error budgets."""

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.nn import functional as F

from protocol import ROOT, associated, boxes_for, fingerprint, load, materialize, network, raw_prediction
from replay import Mode, benchmark_inputs, evaluate, overlap, read_frames, select


def observe_pair(model, candidate, image, saved):
    output, shape, scale, pad = raw_prediction(model, image)
    head = candidate.model[-1]
    confidence = torch.cat([F.conv2d(saved[i], branch[-1].weight[2:3], branch[-1].bias[2:3]).sigmoid().flatten()
                            for i, branch in enumerate(head.cv3)]).detach().numpy()
    boxes = boxes_for(output, shape, scale, pad) / 1.5
    boxes[:, [0, 2]] += 160
    result = {}
    for name, scores in [('v18', output[0, 6]), ('candidate', confidence)]:
        kept = scores >= .0002
        result[name] = (boxes[kept], scores[kept], np.zeros(int(kept.sum()), np.float32))
    return result


def threshold_at_budget(observations, budget):
    scores = sorted([float(o[1].max(initial=0)) for o in observations], reverse=True)
    # score >= threshold; nextafter excludes the first disallowed frame, including ties.
    return max(.0002, float(np.nextafter(np.float32(scores[budget]), np.float32(np.inf)))) if budget < len(scores) else .0002


def threshold_at_box_budget(observations, budget):
    scores = sorted([b[4] for o in observations for b in select(o, Mode(.0002))], reverse=True)
    return max(.0002, float(np.nextafter(np.float32(scores[budget]), np.float32(np.inf)))) if budget < len(scores) else .0002


def visible_episode(episode, frames, threshold, iou=.30, padding=0):
    """Score only this and earlier frames; both confirmation boxes must localize GT."""
    accurate = {}
    for index, observation in frames.items():
        truths = [[x-padding, y-padding, r+padding, s+padding]
                  for x,y,r,s in episode['boxes'][str(index)]]
        accurate[index] = [box for box in select(observation, Mode(threshold))
                           if any(overlap(box, truth) >= iou for truth in truths)]
    first, last = episode['first'], episode['last_partial']
    return dict(strict_first=bool(accurate[first]),
                earliest_frame=next((i for i in sorted(accurate) if accurate[i]), None),
                confirmed_frame=next((i for i in range(first+1,last+1)
                                      if any(associated(a,b) for a in accurate[i-1] for b in accurate[i])), None))


def validate_new(annotations, sources, excluded_matches):
    lookup = {r['case']:r for r in sources}
    if len(lookup) != len(sources) or len({r['match_id'] for r in sources}) != len(sources):
        raise ValueError('Duplicate source case or match')
    if {r['match_id'] for r in sources} & set(excluded_matches):
        raise ValueError('A new match overlaps a known training/development source')
    if annotations['fps'] != 32 or annotations['image_size'] != [1280,720]:
        raise ValueError('Unexpected image clock or geometry')
    for row in annotations['appearances']:
        if row['case'] not in lookup or row['last_partial'] < row['first']:
            raise ValueError('Invalid appearance')
        if set(map(int,row['boxes'])) != set(range(row['first'],row['last_partial']+1)):
            raise ValueError('Each partial frame needs a label')
        if row['box'] not in row['boxes'][str(row['first'])]:
            raise ValueError('First-frame labels disagree')
        if not row['censored'] and row['full_frame'] != row['last_partial']+1:
            raise ValueError('Full-exit boundary disagrees')
        for boxes in row['boxes'].values():
            for x,y,r,s in boxes:
                if not 160 <= x < r <= 1120 or not 0 <= y < s <= 720:
                    raise ValueError('Invalid visible box')
    seen = set()
    for row in annotations['empty_windows']:
        if row['case'] not in lookup or row['start'] > row['stop']:
            raise ValueError('Invalid negative window')
        for i in range(row['start'],row['stop']+1):
            key = row['case'],i
            if key in seen:
                raise ValueError('Double-counted negative frame')
            seen.add(key)
    for row in annotations['appearances']:
        if any((row['case'],i) in seen for i in range(row['first'],row['last_partial']+1)):
            raise ValueError('A frame is labeled both positive and empty')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--candidate', type=Path, required=True)
    parser.add_argument('--benchmark', choices=['previous', 'fresh', 'new'], default='previous')
    parser.add_argument('--annotations', type=Path)
    parser.add_argument('--sources', type=Path)
    parser.add_argument('--media', type=Path)
    parser.add_argument('--cache', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--calibration-report', type=Path)
    parser.add_argument('--iou', type=float, default=.30)
    parser.add_argument('--gt-padding', type=int, default=0, help='Sensitivity only: expand visible GT in source pixels')
    args = parser.parse_args()
    torch.set_num_threads(2)
    cv2.setNumThreads(1)
    if args.benchmark == 'new':
        if any(value is None for value in [args.annotations,args.sources,args.media,args.calibration_report]):
            parser.error('New evaluation requires annotations, sources, media and a frozen calibration report')
        annotations = json.loads(args.annotations.read_text())
        appearances, negatives = annotations['appearances'], annotations['empty_windows']
        rows = json.loads(args.sources.read_text())
        excluded = {r['match_id'] for r in load('sources.json')}
        excluded.update(json.loads(args.calibration_report.read_text())['source_match_ids'])
        checkpoint = torch.load(args.candidate,map_location='cpu',weights_only=False)
        excluded.update(checkpoint['early_fragment_training']['training_match_ids'])
        del checkpoint
        validate_new(annotations,rows,excluded)
        sources = {r['case']:r for r in rows}
    else:
        appearances, negatives = benchmark_inputs(args.benchmark)
        sources = {r['case']:r for r in load('sources.json')}
    by_case = {}
    for group, first, last in [(appearances, 'first', 'last_partial'), (negatives, 'start', 'stop')]:
        for row in group:
            by_case.setdefault(row['case'], set()).update(range(row[first], row[last]+1))
    signature = dict(v18=fingerprint(ROOT/'checkpoints/v18.pt'), candidate=fingerprint(args.candidate),
                     annotations=fingerprint(args.annotations if args.benchmark=='new' else ROOT/f'data/{args.benchmark}.json'),
                     sources=fingerprint(args.sources if args.benchmark=='new' else ROOT/'data/sources.json'))
    if args.cache.is_file():
        cache = torch.load(args.cache, weights_only=False)
        if cache['signature'] != signature:
            raise ValueError('Stale prediction cache')
        observations = cache['observations']
    else:
        observations = {}
        with network(ROOT/'checkpoints/v18.pt', torch.device('cpu')) as model, network(args.candidate, torch.device('cpu')) as candidate:
            # This shortcut is valid only if all geometry/features/side/head parameters are identical.
            for key, value in model.state_dict().items():
                new = candidate.state_dict()[key]
                changed = not torch.equal(value, new)
                if changed and not (key.startswith('model.26.cv3.') and key.endswith(('.2.weight', '.2.bias')) and torch.equal(value[:2], new[:2])):
                    raise ValueError('Candidate is not a frozen-feature presence head')
            saved = {}
            hooks = [branch[-1].register_forward_pre_hook(lambda module, inputs, i=i: saved.update({i:inputs[0]}))
                     for i, branch in enumerate(model.model[-1].cv3)]
            for case, frames in by_case.items():
                if args.benchmark == 'new':
                    path, offset = args.media/sources[case]['file'], 0
                    if fingerprint(path) != sources[case]['sha256']:
                        raise ValueError('Media hash mismatch')
                else:
                    path, offset = materialize(sources[case], ROOT/'.cache')
                observations[case] = {i: observe_pair(model, candidate, image, saved)
                                      for i, _, image in read_frames(path, frames, offset)}
                print(case, len(frames), flush=True)
            for hook in hooks:
                hook.remove()
        args.cache.parent.mkdir(parents=True, exist_ok=True)
        torch.save(dict(signature=signature, observations=observations), args.cache)
    empty = {name: [observations[r['case']][i][name] for r in negatives for i in range(r['start'],r['stop']+1)]
             for name in ('v18','candidate')}
    total = len(empty['v18'])
    if args.calibration_report:
        thresholds = json.loads(args.calibration_report.read_text())['frozen_thresholds']
    else:
        budget = sum(bool(select(o, Mode(.13))) for o in empty['v18'])
        thresholds = dict(v18=.13, candidate=threshold_at_budget(empty['candidate'],budget))
    report = dict(signature=signature, benchmark=args.benchmark, frozen_thresholds=thresholds,
                  iou=args.iou, gt_padding=args.gt_padding,
                  empty_frames=total, source_match_ids=sorted({sources[k]['match_id'] for k in by_case}), results={})
    budgets = [('frozen', None), ('zero',0), ('one_percent',int(total*.01)), ('five_percent',int(total*.05)),
               ('twelve_percent',int(total*.12)), ('equal_boxes_1pct',int(total*.01)),
               ('equal_boxes_5pct',int(total*.05)), ('equal_boxes_12pct',int(total*.12))]
    for setting, budget in budgets:
        report['results'][setting] = {}
        for name in ('v18','candidate'):
            threshold = thresholds[name] if budget is None else (
                threshold_at_box_budget(empty[name],budget) if setting.startswith('equal_boxes') else threshold_at_budget(empty[name],budget))
            episodes = [(r,{i:observations[r['case']][i][name] for i in range(r['first'],r['last_partial']+1)}) for r in appearances]
            windows = [(r,{i:observations[r['case']][i][name] for i in range(r['start'],r['stop']+1)}) for r in negatives]
            result = evaluate(episodes, windows, {name:Mode(threshold)}, strict_first_in_pair=True)[name]
            # For the new set every partial frame has a visible-box label. No ROI proximity shortcut.
            if args.benchmark == 'new':
                for row, (episode, frames) in zip(result['cases'], episodes):
                    row.update(visible_episode(episode, frames, threshold, args.iou, args.gt_padding))
                    row['partial_frames'] = episode['last_partial']-episode['first']+1
                    row['censored'] = episode['censored']
                result['strict_first'] = sum(r['strict_first'] for r in result['cases'])
                result['any_while_partial'] = sum(r['earliest_frame'] is not None for r in result['cases'])
                result['confirmed_while_partial'] = sum(r['confirmed_frame'] is not None for r in result['cases'])
                uncensored = [item for item, (episode, _) in zip(result['cases'], episodes) if not episode['censored']]
                result['uncensored'] = dict(episodes=len(uncensored),
                    strict_first=sum(row['strict_first'] for row in uncensored),
                    confirmed_before_full_exit=sum(row['confirmed_frame'] is not None for row in uncensored),
                    two_frame_opportunity=sum(row['partial_frames'] >= 2 for row in uncensored))
            for case in result['cases']:
                for key in ['earliest','confirmed']:
                    value=case[f'{key}_frame']
                    case[f'{key}_delay_ms'] = None if value is None else (value-case['first'])*1000/32
            result['budget'] = budget
            result['budget_unit'] = 'false_boxes' if setting.startswith('equal_boxes') else 'false_frames'
            result['false_boxes'] = sum(len(select(o,Mode(threshold))) for o in empty[name])
            result['negative_confirmed_frames'] = sum(
                any(associated(a,b) for a in select(frames[i-1],Mode(threshold)) for b in select(frames[i],Mode(threshold)))
                for _,frames in windows for i in frames if i-1 in frames)
            report['results'][setting][name] = result
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({setting:{name:{k:v for k,v in data.items() if k!='cases'} for name,data in entries.items()}
                      for setting,entries in report['results'].items()},indent=2))


if __name__ == '__main__':
    main()
