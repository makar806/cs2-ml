"""Compare V9 and V10 on the preserved held-out images."""
import collections
import json
from pathlib import Path
import sys

import numpy as np
from ultralytics import YOLO

root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(root / 'scripts'))
from evaluate_pose import iou

data = root / 'data/pose-v10'
report = {'confidence': .35, 'iou': .5, 'imgsz': 960, 'models': {}}
for version in ('v9', 'v10'):
    model = YOLO(str(root / f'runs/pose-{version}/weights/best.pt'))
    scores = report['models'][version] = {}
    for split in ('newcheck', 'check', 'test'):
        gt_count = pred_count = matches = 0
        head_errors = []
        correct_heads = 0
        images = sorted((data / 'images' / split).iterdir())
        results = model.predict([str(p) for p in images], stream=True, batch=8,
                                conf=.35, imgsz=960, device=0, verbose=False)
        for path, result in zip(images, results, strict=True):
            assert Path(result.path).name == path.name
            labels = (data / 'labels' / split / f'{path.stem}.txt').read_text().splitlines()
            h, w = result.orig_shape
            targets = []
            for row in labels:
                cls, x, y, bw, bh, hx, hy, vis = map(float, row.split())
                box = np.array([(x-bw/2)*w, (y-bh/2)*h, (x+bw/2)*w, (y+bh/2)*h])
                targets.append((int(cls), box, np.array([hx*w, hy*h]), vis))
            boxes = result.boxes.xyxy.cpu().numpy()
            classes = result.boxes.cls.cpu().numpy()
            heads = result.keypoints.xy.cpu().numpy()
            confidence = result.keypoints.conf.cpu().numpy()
            candidates = sorted(((iou(b, t[1]), pi, gi)
                                 for pi, b in enumerate(boxes) for gi, t in enumerate(targets)
                                 if classes[pi] == t[0]), reverse=True)
            used_p, used_g = set(), set()
            for overlap, pi, gi in candidates:
                if overlap < .5 or pi in used_p or gi in used_g:
                    continue
                used_p.add(pi)
                used_g.add(gi)
                if targets[gi][3] > 0 and confidence[pi, 0] >= .55:
                    correct_heads += 1
                    head_errors.append(float(np.linalg.norm(heads[pi, 0] - targets[gi][2])))
            gt_count += len(targets)
            pred_count += len(boxes)
            matches += len(used_g)
        scores[split] = dict(images=len(images), gt=gt_count, predictions=pred_count,
                             matches=matches, false_positives=pred_count-matches,
                             heads_above_threshold=correct_heads,
                             median_head_error_px=float(np.median(head_errors)) if head_errors else None)
    print(version, scores, flush=True)
(root/'runs/pose-v10/comparison-v9.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
