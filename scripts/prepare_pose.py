"""Convert audited Label Studio JSON to one-head YOLO Pose data."""
import collections
import hashlib
import json
import re
import shutil
from pathlib import Path
import yaml
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
CLASS_IDS = {'T': 0, 'CT': 1}

def convert(results):
    boxes = {r['id']: r for r in results if r['type'] == 'rectanglelabels'}
    heads = [r for r in results if r['type'] == 'keypointlabels']
    assert all(h.get('parentID') in boxes for h in heads), 'Orphan keypoint'
    lines = []
    for bid, box in boxes.items():
        b = box['value']
        assert b.get('rotation', 0) == 0
        points = [h for h in heads if h.get('parentID') == bid]
        assert len(points) <= 1, 'Multiple head points for a box'
        cls = CLASS_IDS[b['rectanglelabels'][0]]
        x, y, w, h = (b[k] / 100 for k in ('x', 'y', 'width', 'height'))
        assert w > 0 and h > 0 and x >= 0 and y >= 0
        assert x + w <= 1.00001 and y + h <= 1.00001
        point = [0., 0., 0]
        if points:
            v = points[0]['value']
            point = [v['x'] / 100, v['y'] / 100, 2]
            assert x <= point[0] <= x + w and y <= point[1] <= y + h
        values = [x + w / 2, y + h / 2, w, h, *point]
        lines.append(str(cls) + ' ' + ' '.join(f'{n:.8f}' for n in values))
    return lines

if __name__ == '__main__':
    source = sorted((ROOT / 'data/exports').glob('*/ML_CS.json'))[-1]
    tasks = json.loads(source.read_text(encoding='utf-8'))
    dest = ROOT / 'data/pose-v1'
    if dest.exists():
        raise SystemExit(f'Refusing to overwrite existing dataset: {dest}')
    records = []
    seen_hashes = {}
    stats = {s: {'images': 0, 'objects': collections.Counter()} for s in ('train', 'val', 'test')}
    for task in tasks:
        image = ROOT / '.label-studio/media' / task['data']['image'].split('/data/')[1]
        stamp = re.search(r'(\d{14})_\d+\.', image.name).group(1)
        # Whole chronological scene blocks, never random individual frames.
        split = 'train' if stamp < '20261017191545' else 'val' if stamp < '20261017191616' else 'test'
        group = {'train': 'early_arena_and_dust_scenes', 'val': 'tunnel_episode', 'test': 'late_outdoor_episode'}[split]
        results = task['annotations'][0]['result']
        lines = convert(results)
        with Image.open(image) as im:
            assert im.size == (1440, 1080), f'Thumbnail or unexpected size: {image}'
        digest = hashlib.sha256(image.read_bytes()).hexdigest()
        assert digest not in seen_hashes, f'Duplicate image: {image}'
        seen_hashes[digest] = split
        name = f'{task["id"]}_{stamp}'
        img_dir, label_dir = dest / 'images' / split, dest / 'labels' / split
        img_dir.mkdir(parents=True, exist_ok=True)
        label_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(image, img_dir / f'{name}{image.suffix}')
        (label_dir / f'{name}.txt').write_text('\n'.join(lines) + ('\n' if lines else ''), encoding='utf-8')
        stats[split]['images'] += 1
        stats[split]['objects'].update(r['value']['rectanglelabels'][0] for r in results if r['type'] == 'rectanglelabels')
        records.append({'task': task['id'], 'source': str(image), 'split': split, 'group': group, 'sha256': digest})
    config = {'path': str(dest), 'train': 'images/train', 'val': 'images/val', 'test': 'images/test',
              'names': {0: 'T', 1: 'CT'}, 'kpt_shape': [1, 3], 'flip_idx': [0]}
    (ROOT / 'configs/pose.yaml').write_text(yaml.safe_dump(config, sort_keys=False), encoding='utf-8')
    manifest = {'source': str(source), 'limitation': 'Within-evening chronological scene holdout; not an independent-session test. Background, skins and recording conditions remain correlated.', 'stats': stats, 'images': records}
    (dest / 'manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    print(json.dumps(stats, indent=2))
