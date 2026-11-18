"""Audit all raw anchors on test first frames, including scores below the cache floor."""

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import torch

from protocol import ROOT, boxes_for, fingerprint, network, raw_prediction
from replay import overlap, read_frames


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--candidate', type=Path, required=True)
    parser.add_argument('--media', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(2)
    cv2.setNumThreads(1)
    annotations = json.loads((ROOT/'data/next/benchmark.json').read_text())
    sources = {r['case']:r for r in json.loads((ROOT/'data/next/sources.json').read_text())}
    report = dict(benchmark_sha256=fingerprint(ROOT/'data/next/benchmark.json'), models={})
    for name,weights in [('v18',ROOT/'checkpoints/v18.pt'),('v21',args.candidate)]:
        rows = []
        with network(weights,torch.device('cpu')) as model:
            for episode in annotations['appearances']:
                source = sources[episode['case']]
                path = args.media/source['file']
                if fingerprint(path) != source['sha256']:
                    raise ValueError('Media hash mismatch')
                image = next(read_frames(path,[episode['first']]))[2]
                output,shape,scale,pad = raw_prediction(model,image)
                boxes = boxes_for(output,shape,scale,pad)/1.5
                boxes[:,[0,2]] += 160
                iou = np.asarray([overlap(box,episode['box']) for box in boxes])
                valid = np.flatnonzero(iou>=.3)
                index = valid[np.argmax(output[0,6,valid])] if len(valid) else None
                rows.append(dict(case=episode['case'],frame=episode['first'],max_iou=float(iou.max()),
                    anchors_iou03=len(valid), best_presence=float(output[0,6,index]) if index is not None else None,
                    best_score_box=boxes[index].tolist() if index is not None else None))
        report['models'][name] = dict(sha256=fingerprint(weights),cases=rows)
    args.output.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report['models'],indent=2))


if __name__ == '__main__':
    main()
