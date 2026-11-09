"""Compare frozen V3/V4 at identical thresholds on match-held-out frames."""
import json
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw
from ultralytics import YOLO
from evaluate_pose import iou

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT.parent/'2026-10-17/new-chat/outputs/fourth-training'

if __name__ == '__main__':
    OUT.mkdir(parents=True, exist_ok=True)
    report = {}
    for version in ('v3', 'v4'):
        model = YOLO(str(ROOT/f'runs/pose-{version}/weights/best.pt'))
        gt_count = pred_count = matched = 0
        errors, details = [], []
        canvas = Image.new('RGB', (1280, 4*385), 'white')
        draw = ImageDraw.Draw(canvas)
        for idx, path in enumerate(sorted((ROOT/'data/pose-v4/images/check').glob('*.png'))):
            rows = [list(map(float, line.split())) for line in (ROOT/'data/pose-v4/labels/check'/(path.stem+'.txt')).read_text().splitlines()]
            result = model.predict(str(path), device=0, imgsz=640, conf=.25, verbose=False)[0]
            h, w = result.orig_shape
            boxes = result.boxes.xyxy.cpu().numpy()
            classes = result.boxes.cls.cpu().numpy().astype(int)
            ground = [(int(c), np.array([(x-bw/2)*w, (y-bh/2)*h, (x+bw/2)*w, (y+bh/2)*h]), np.array([hx*w,hy*h]), vis) for c,x,y,bw,bh,hx,hy,vis in rows]
            candidates = sorted([(iou(b,g[1]),pi,gi) for pi,b in enumerate(boxes) for gi,g in enumerate(ground) if classes[pi]==g[0]], reverse=True)
            used_p, used_g = set(), set()
            for overlap,pi,gi in candidates:
                if overlap < .5 or pi in used_p or gi in used_g:
                    continue
                used_p.add(pi); used_g.add(gi)
                if ground[gi][3] > 0:
                    errors.append(float(np.linalg.norm(result.keypoints.xy.cpu().numpy()[pi,0]-ground[gi][2])))
            gt_count += len(ground); pred_count += len(boxes); matched += len(used_g)
            details.append({'image':path.name, 'gt':len(ground), 'predictions':len(boxes), 'matched':len(used_g)})
            im = Image.fromarray(result.plot()[...,::-1]); im.thumbnail((426,350))
            x,y=idx%3*426, idx//3*385
            canvas.paste(im,(x,y)); draw.text((x+4,y+352), f'{path.stem}\nGT {len(ground)}, predicted {len(boxes)}, matched {len(used_g)}',fill='black')
        canvas.save(OUT/f'check-{version}.jpg')
        report[version] = {'gt':gt_count, 'predictions':pred_count, 'matched':matched, 'precision':matched/pred_count if pred_count else 0, 'recall':matched/gt_count if gt_count else 0, 'conditional_head_mean_pixels':float(np.mean(errors)) if errors else None, 'head_matches':len(errors), 'images':details}
    report['protocol'] = {'confidence':.25, 'class_matched_iou':.5, 'imgsz':640, 'note':'12 frames, 2 held-out matches. Historical third-party training source overlap is unknown. Conditional head error excludes misses.'}
    (OUT/'active-check-comparison.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps(report,indent=2))
