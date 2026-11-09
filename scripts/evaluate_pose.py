"""Evaluate untouched test block, including matched-head error in original pixels."""
import json
import argparse
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parents[1]

def iou(a, b):
    lo, hi = np.maximum(a[:2], b[:2]), np.minimum(a[2:], b[2:])
    intersection = np.maximum(hi - lo, 0).prod()
    return float(intersection / max((a[2:] - a[:2]).prod() + (b[2:] - b[:2]).prod() - intersection, 1e-9))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', default='pose-baseline')
    parser.add_argument('--eval-name', default='pose-test')
    args = parser.parse_args()
    run = ROOT / 'runs' / args.run
    model = YOLO(str(run / 'weights/best.pt'))
    metrics = model.val(data=str(ROOT / 'configs/pose.yaml'), split='test',
                        device=0, imgsz=640, batch=8, workers=0,
                        project=str(ROOT / 'runs'), name=args.eval_name, plots=True)
    errors, normal_errors, details = [], [], []
    total_gt = total_pred = matched = 0
    sources = sorted((ROOT / 'data/pose-v1/images/test').glob('*.jpg'))
    for path in sources:
        labels = np.loadtxt(ROOT / 'data/pose-v1/labels/test' / f'{path.stem}.txt', ndmin=2)
        result = model.predict(str(path), device=0, imgsz=640, conf=.25, verbose=False)[0]
        h, w = result.orig_shape
        boxes = result.boxes.xyxy.cpu().numpy()
        classes = result.boxes.cls.cpu().numpy().astype(int)
        scores = result.boxes.conf.cpu().numpy()
        keypoints = result.keypoints.xy.cpu().numpy()[:, 0, :]
        ground = []
        for row in labels:
            cls, cx, cy, bw, bh, hx, hy, visible = row
            ground.append((int(cls), np.array([(cx-bw/2)*w,(cy-bh/2)*h,(cx+bw/2)*w,(cy+bh/2)*h]), np.array([hx*w,hy*h]), visible))
        candidates = sorted([(iou(b, g[1]), pi, gi) for pi,b in enumerate(boxes) for gi,g in enumerate(ground) if classes[pi] == g[0]], reverse=True)
        used_p, used_g = set(), set()
        for overlap, pi, gi in candidates:
            if overlap < .5 or pi in used_p or gi in used_g:
                continue
            used_p.add(pi); used_g.add(gi); matched += 1
            if ground[gi][3] > 0:
                error = float(np.linalg.norm(keypoints[pi] - ground[gi][2]))
                height = ground[gi][1][3] - ground[gi][1][1]
                errors.append(error); normal_errors.append(error / height)
        total_gt += len(ground); total_pred += len(boxes)
        details.append({'image': path.name, 'gt': len(ground), 'predictions': len(boxes), 'matches': len(used_g)})
        result.save(filename=str(run / f'test_{path.name}'))
    report = {'weights': str(run / 'weights/best.pt'), 'test_images': len(sources),
              'ultralytics': {k: float(v) for k,v in metrics.results_dict.items()},
              'fixed_thresholds': {'confidence': .25, 'class_matched_box_iou': .5},
              'ground_truth_objects': total_gt, 'predictions': total_pred, 'matched': matched,
              'precision': matched / total_pred if total_pred else 0.,
              'recall': matched / total_gt if total_gt else 0.,
              'head_error_on_matched_objects_only': {'count': len(errors),
                  'mean_pixels_at_1440x1080': float(np.mean(errors)) if errors else None,
                  'median_pixels': float(np.median(errors)) if errors else None,
                  'mean_fraction_of_body_height': float(np.mean(normal_errors)) if errors else None},
              'limitations': 'Small within-evening scene holdout. Conditional head error excludes missed or wrong-class objects; pose mAP uses default custom-keypoint OKS scaling. No live latency benchmark.',
              'images': details}
    (run / 'test-report.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    canvas = Image.new('RGB', (1280, 2 * 505), 'white')
    draw = ImageDraw.Draw(canvas)
    for row, prefix in enumerate(('178_', '180_', '184_', '189_')):
        src = next(p for p in sources if p.name.startswith(prefix))
        im = Image.open(run / f'test_{src.name}'); im.thumbnail((640, 480))
        x, y = (row % 2) * 640, (row // 2) * 505
        canvas.paste(im, (x, y))
        draw.text((x+5,y+482), src.name, fill='black')
    canvas.save(run / 'test-preview.jpg')
    print(json.dumps(report, indent=2))
