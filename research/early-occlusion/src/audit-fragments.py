"""Measure V18 assignment on annotated fragments; cache frozen classification features."""

import argparse
import json
import runpy
from pathlib import Path

import cv2
import numpy as np
import torch
from ultralytics.utils.tal import make_anchors

from protocol import ROOT, fingerprint, load, materialize, network, raw_prediction
from replay import read_frames


def ious(boxes, truth):
    intersection = (np.minimum(boxes[:, 2:], truth[2:]) - np.maximum(boxes[:, :2], truth[:2])).clip(0).prod(1)
    return intersection / (np.diff(boxes[:, [0, 2]], axis=1).ravel() *
                           np.diff(boxes[:, [1, 3]], axis=1).ravel() +
                           (truth[2]-truth[0])*(truth[3]-truth[1]) - intersection + 1e-9)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--features', type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(2)
    cv2.setNumThreads(1)
    PresenceAssigner = runpy.run_path(str(ROOT/'src/train-candidate.py'))['PresenceAssigner']
    weights = ROOT/'checkpoints/v18.pt'
    sources = {row['case']: row for row in load('sources.json')}
    samples = load('calibration.json')
    if any(row['expected_class'] not in (0,1) for row in samples):
        raise ValueError('Expected class mapping 0=T, 1=CT')
    grouped = {}
    for row in samples:
        grouped.setdefault(row['case'], {})[row['frame']] = row
    audit, cached = [], []
    rng = np.random.default_rng(42)
    with network(weights, torch.device('cpu')) as model:
        if model.names != {0:'T',1:'CT'}:
            raise ValueError('Checkpoint side mapping changed')
        head = model.model[-1]
        assigner = PresenceAssigner(topk=10, num_classes=3, alpha=.5, beta=6., stride=head.stride.tolist())
        saved, features = {}, {}
        hooks = [head.register_forward_hook(lambda module, inputs, output: saved.update(pred=output[1]))]
        for level, branch in enumerate(head.cv3):
            hooks.append(branch[-1].register_forward_pre_hook(
                lambda module, inputs, level=level: features.update({level: inputs[0].detach()})))
        for case, frames in grouped.items():
            path, offset = materialize(sources[case], ROOT/'.cache')
            for frame, _, image in read_frames(path, frames, offset):
                row = frames[frame]
                raw, _, _, _ = raw_prediction(model, image)
                pred = saved['pred']
                anchors, strides = make_anchors(pred['feats'], head.stride, .5)
                xywh = raw[0, :4].T
                boxes = np.concatenate((xywh[:, :2]-xywh[:, 2:]/2, xywh[:, :2]+xywh[:, 2:]/2), axis=1)
                score = raw[0, 6]
                gt = row['box_source']
                quality = np.zeros(len(boxes), dtype=np.float32)
                target = np.zeros_like(quality)
                fg = np.zeros(len(boxes), dtype=bool)
                if gt:
                    truth = np.asarray(gt['box'], dtype=np.float32)
                    truth[[0, 2]] -= 160
                    truth *= 4/3
                    quality = ious(boxes, truth)
                    labels = torch.tensor([[[row['expected_class']]]], dtype=torch.float32)
                    assigned = assigner(pred['scores'].permute(0, 2, 1).sigmoid(),
                                        torch.from_numpy(boxes)[None], anchors*strides,
                                        labels, torch.from_numpy(truth)[None, None], torch.ones(1, 1, 1))
                    target = assigned[2][0].amax(-1).numpy()
                    fg = assigned[3][0].numpy()
                    levels = strides[:, 0].numpy()
                    audit.append(dict(case=case, frame=frame, width_1280=float(truth[2]-truth[0]),
                                      height_1280=float(truth[3]-truth[1]),
                                      best_presence=float(score[quality >= .3].max(initial=0)),
                                      best_iou=float(quality.max()), positive_anchors=int(fg.sum()),
                                      positive_p2=int((fg & (levels == 4)).sum()),
                                      max_target=float(target.max()), target_mass=float(target.sum()),
                                      p2_target_mass=float(target[levels == 4].sum()),
                                      max_geometry_positive_target=float(target[quality >= .3].max(initial=0))))
                # Keep all plausible localization candidates, the hardest background and uniform background.
                # Sampling is fixed before learning. Neither selection nor labels use a test set.
                keep = set(np.flatnonzero((quality >= .1) | fg).tolist())
                keep.update(np.argsort(score)[-512:].tolist())
                keep.update(rng.choice(len(score), size=512, replace=False).tolist())
                keep = np.asarray(sorted(keep))
                all_features = np.concatenate([features[i][0].reshape(64, -1).T.numpy()
                                               for i in range(len(head.cv3))])
                level_id = np.concatenate([np.full(features[i].shape[-2]*features[i].shape[-1], i)
                                           for i in range(len(head.cv3))])
                cached.append(dict(case=case, frame=frame, match_id=row['match_id'],
                                   features=all_features[keep], level=level_id[keep],
                                   iou=quality[keep], tal_target=target[keep], fg=fg[keep],
                                   baseline=score[keep], positive=gt is not None))
            print('audited', case, flush=True)
        for hook in hooks:
            hook.remove()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    report = dict(weights_sha256=fingerprint(weights), calibration_sha256=fingerprint(ROOT/'data/calibration.json'),
                  backend='cpu-fp32', assignment=dict(alpha=.5, beta=6, topk=10, stride=[8,16,32,4]), samples=audit)
    for name, rows in [('all', audit), ('width_le_8', [r for r in audit if r['width_1280'] <= 8.00001])]:
        report[name] = dict(count=len(rows), zero_fg=sum(r['positive_anchors']==0 for r in rows),
                            max_target_median=float(np.median([r['max_target'] for r in rows])),
                            target_mass_median=float(np.median([r['target_mass'] for r in rows])),
                            target_below_013=sum(r['max_target'] < .13 for r in rows),
                            no_p2=sum(r['positive_p2']==0 for r in rows))
    args.output.write_text(json.dumps(report, indent=2)+'\n')
    args.features.parent.mkdir(parents=True, exist_ok=True)
    torch.save(dict(rows=cached, weights_sha256=fingerprint(weights)), args.features)
    print(json.dumps({k:v for k,v in report.items() if k != 'samples'}, indent=2))


if __name__ == '__main__':
    main()
