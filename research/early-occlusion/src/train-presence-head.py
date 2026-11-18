"""Fit only 260 presence parameters; keep V18 geometry, teams and keypoints exactly fixed."""

import argparse
from copy import deepcopy
import json
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F
from ultralytics import YOLO

from protocol import ROOT, fingerprint


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--features', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--steps', type=int, default=400)
    parser.add_argument('--regularization', type=float, default=.01)
    parser.add_argument('--target', choices=['visible', 'tal'], default='visible')
    args = parser.parse_args()
    torch.set_num_threads(2)
    torch.manual_seed(42)
    weights = ROOT/'checkpoints/v18.pt'
    data = torch.load(args.features, map_location='cpu', weights_only=False)
    if data['weights_sha256'] != fingerprint(weights):
        raise ValueError('Feature cache came from different weights')
    model = YOLO(str(weights), task='pose').model.float().eval()
    head = model.model[-1]
    original = torch.stack([torch.cat((branch[-1].weight[2, :, 0, 0], branch[-1].bias[2:3]))
                            for branch in head.cv3]).detach()
    delta = torch.nn.Parameter(torch.zeros_like(original))
    optimizer = torch.optim.Adam([delta], lr=.002)
    features, levels, labels, importance = [], [], [], []
    for row in data['rows']:
        quality = row['iou']
        # Hard presence target, but only when the frozen box already localizes visible pixels.
        positive = quality >= .30 if args.target == 'visible' else row['fg'] & (row['tal_target'] > 0)
        negative = quality < .10
        valid = positive | negative
        w = np.zeros(len(quality), np.float32)
        if positive.any():
            w[positive] = 1 / positive.sum()
        # Each reviewed frame has equal negative mass, including paired empty views.
        w[negative] = 1 / max(1, negative.sum())
        features.append(row['features'][valid])
        levels.append(row['level'][valid])
        labels.append(positive[valid].astype(np.float32) if args.target == 'visible' else row['tal_target'][valid])
        importance.append(w[valid])
    x = torch.from_numpy(np.concatenate(features)).float()
    x = torch.cat((x, torch.ones(len(x), 1)), 1)
    level = torch.from_numpy(np.concatenate(levels)).long()
    target = torch.from_numpy(np.concatenate(labels)).float()
    weight = torch.from_numpy(np.concatenate(importance)).float()
    weight /= weight.sum()
    log = []
    for step in range(args.steps):
        optimizer.zero_grad()
        logits = (x * (original+delta)[level]).sum(1)
        supervised = (F.binary_cross_entropy_with_logits(logits, target, reduction='none') * weight).sum()
        # Penalize change in log odds on sampled backgrounds, preserving the V18 starting point.
        drift = (x * delta[level]).sum(1)
        penalty = (drift.square() * weight).sum()
        loss = supervised + args.regularization*penalty
        loss.backward()
        optimizer.step()
        if step % 50 == 0 or step == args.steps-1:
            record = dict(step=step+1, bce=float(supervised.detach()), logit_drift=float(penalty.detach()),
                          objective=float(loss.detach()))
            log.append(record)
            print(record, flush=True)
    before = deepcopy(model.state_dict())
    with torch.no_grad():
        fitted = original + delta
        for i, branch in enumerate(head.cv3):
            branch[-1].weight[2, :, 0, 0].copy_(fitted[i, :64])
            branch[-1].bias[2].copy_(fitted[i, 64])
    changed = []
    for name, value in model.state_dict().items():
        if not torch.equal(value, before[name]):
            if not (name.startswith('model.26.cv3.') and name.endswith(('.2.weight', '.2.bias'))):
                raise AssertionError(f'Unexpected changed parameter {name}')
            if not torch.equal(value[:2], before[name][:2]):
                raise AssertionError('Team outputs changed')
            changed.append(name)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    checkpoint = torch.load(weights, map_location='cpu', weights_only=False)
    checkpoint['model'] = model
    checkpoint['ema'] = None
    checkpoint['optimizer'] = None
    checkpoint['early_fragment_training'] = dict(method='frozen-visible-box-presence', seed=42,
        source_sha256=fingerprint(weights), features_sha256=fingerprint(args.features),
        steps=args.steps, regularization=args.regularization, target=args.target, learned_parameters=260,
        training_match_ids=sorted({r['match_id'] for r in data['rows']}),
        positive_iou=.30 if args.target == 'visible' else None,
        positive_assignment='IoU >= .30' if args.target == 'visible' else 'TAL foreground with nonzero target',
        negative_iou_below=.10, changed_tensors=changed)
    torch.save(checkpoint, args.output)
    report = dict(checkpoint['early_fragment_training'], checkpoint_sha256=fingerprint(args.output),
                  frames=len(data['rows']), anchors=len(x), positive_anchors=int((target > 0).sum()),
                  target_mass=float(target.sum()), training_log=log)
    args.output.with_suffix('.json').write_text(json.dumps(report, indent=2)+'\n')
    print('Saved', args.output, flush=True)


if __name__ == '__main__':
    main()
