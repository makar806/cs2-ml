"""Paired full-network CPU timing and shared-feature equivalence checks on raw pixels."""

import argparse
from contextlib import ExitStack
import json
from pathlib import Path
import platform
import runpy
from time import perf_counter_ns

import cv2
import numpy as np
import torch
import ultralytics

from protocol import ROOT, boxes_for, fingerprint, network, raw_prediction
from replay import Mode, read_frames, select


def infer(model, image, threshold):
    output, shape, scale, pad = raw_prediction(model, image)
    mask = output[0, 6] >= .0002
    boxes = boxes_for(output[:, :, mask], shape, scale, pad)/1.5
    boxes[:, [0, 2]] += 160
    observation = boxes, output[0, 6, mask], np.zeros(int(mask.sum()), np.float32)
    return output, observation, select(observation, Mode(threshold))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--candidate', type=Path, required=True)
    parser.add_argument('--media', type=Path, required=True)
    parser.add_argument('--annotations', type=Path, default=ROOT/'data/next/benchmark.json')
    parser.add_argument('--sources', type=Path, default=ROOT/'data/next/sources.json')
    parser.add_argument('--calibration-report', type=Path, default=ROOT/'evidence/next/v21-development.json')
    parser.add_argument('--repeats', type=int, default=4)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(2)
    cv2.setNumThreads(1)
    annotations = json.loads(args.annotations.read_text())
    sources = {r['case']: r for r in json.loads(args.sources.read_text())}
    thresholds = json.loads(args.calibration_report.read_text())['frozen_thresholds']
    sample = [(r['case'], r['first']) for r in annotations['appearances']]
    sample += [(r['case'], r['start']) for r in annotations['empty_windows'][:4]]
    images = []
    for case, index in sample:
        row = sources[case]
        path = args.media/row['file']
        if fingerprint(path) != row['sha256']:
            raise ValueError('Media hash mismatch')
        images.append(next(read_frames(path, [index]))[2])
    timing = {'v18': [], 'candidate': []}
    checks = []
    with ExitStack() as stack:
        models = {name: stack.enter_context(network(path, torch.device('cpu')))
                  for name,path in [('v18', ROOT/'checkpoints/v18.pt'), ('candidate', args.candidate)]}
        for model in models.values():
            for image in images[:3]:
                infer(model, image, .13)
        for repeat in range(args.repeats):
            for index, image in enumerate(images):
                order = ('v18','candidate') if (repeat+index)%2 == 0 else ('candidate','v18')
                for name in order:
                    start = perf_counter_ns()
                    infer(models[name], image, thresholds[name])
                    timing[name].append((perf_counter_ns()-start)/1e6)
            print('timing round', repeat+1, flush=True)
        # Equivalence checks run outside the timing region. Full V21 is the reference.
        compare = runpy.run_path(str(ROOT/'src/compare-presence.py'))
        saved = {}
        hooks = [branch[-1].register_forward_pre_hook(lambda module, inputs, i=i: saved.update({i:inputs[0]}))
                 for i,branch in enumerate(models['v18'].model[-1].cv3)]
        for image in images[:4]:
            pair = compare['observe_pair'](models['v18'], models['candidate'], image, saved)
            baseline = raw_prediction(models['v18'], image)[0]
            actual, observation, decoded = infer(models['candidate'], image, thresholds['candidate'])
            unchanged = np.array_equal(baseline[:, [0,1,2,3,4,5,7,8,9]], actual[:, [0,1,2,3,4,5,7,8,9]])
            np.testing.assert_allclose(pair['candidate'][0], observation[0], rtol=0, atol=0)
            np.testing.assert_allclose(pair['candidate'][1], observation[1], rtol=2e-5, atol=2e-7)
            np.testing.assert_allclose(select(pair['candidate'], Mode(thresholds['candidate'])), decoded,
                                       rtol=2e-5, atol=2e-7)
            assert unchanged, 'Geometry/team/keypoint channels changed'
            raw_prediction(models['candidate'], images[-1])
            repeated = raw_prediction(models['candidate'], image)[0]
            assert np.array_equal(actual, repeated), 'Output depends on unrelated preceding frame'
            checks.append(dict(unchanged_channels=unchanged, current_frame_only=True,
                               max_presence_error=float(np.abs(pair['candidate'][1]-observation[1]).max(initial=0))))
        for hook in hooks:
            hook.remove()
    report = dict(device='cpu', dtype='float32', torch_threads=2, opencv_threads=1,
                  platform=platform.platform(), machine=platform.machine(), python=platform.python_version(),
                  torch=torch.__version__, ultralytics=ultralytics.__version__, opencv=cv2.__version__, numpy=np.__version__,
                  sample=sample, repeats=args.repeats, warmup_per_model=3,
                  included='two-stage resize, tensor preparation, full forward, box decode, score filter, NMS',
                  excluded='video decode, capture, display, load/warmup, shared-feature shortcut, confirmation waiting',
                  thresholds=thresholds, candidate_sha256=fingerprint(args.candidate),
                  v18_sha256=fingerprint(ROOT/'checkpoints/v18.pt'), equivalence_checks=checks, timing={})
    for name, values in timing.items():
        report['timing'][name] = dict(n=len(values), median_ms=float(np.median(values)),
            p95_ms=float(np.percentile(values,95)), max_ms=max(values), samples_ms=values)
    paired = np.asarray(timing['candidate'])-np.asarray(timing['v18'])
    report['paired_median_difference_ms'] = float(np.median(paired))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps({name:{k:v for k,v in row.items() if k!='samples_ms'} for name,row in report['timing'].items()}))


if __name__ == '__main__':
    main()
