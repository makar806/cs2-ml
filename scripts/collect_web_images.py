"""Collect a reproducible unannotated CS2 image batch with source provenance."""
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from collections import defaultdict, Counter
import hashlib
import io
import json
import random
import re
import shutil
import time
import requests
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'data/raw/internet_batch_001'
REPO = 'fvossel/csgo-player-detection'
BASE = 'https://huggingface.co/datasets/' + REPO

def candidates(rows, cls, head):
    groups = defaultdict(list)
    for r in rows:
        cats, boxes = r['objects']['categories'], r['objects']['bbox']
        if cls not in cats or head not in cats or 4-cls in cats:
            continue
        if not any(c == head and min(b[2:]) >= 12 for c,b in zip(cats,boxes)):
            continue
        if not any(c == cls and 80 <= b[3] <= 450 and b[0] > 2 and b[1] > 2 and b[0]+b[2] < 638 and b[1]+b[3] < 638 for c,b in zip(cats,boxes)):
            continue
        groups[r['source_session']].append(r)
    rng = random.Random(42 + cls)
    for pool in groups.values():
        rng.shuffle(pool)
    selected = []
    used_frames = defaultdict(list)
    while any(groups.values()):
        for session in sorted(groups):
            while groups[session]:
                r = groups[session].pop()
                frame = int(re.search(r'_(\d+)\.png$',r['file_name']).group(1))
                if any(abs(frame - f) < 15 for f in used_frames[session]):
                    continue
                used_frames[session].append(frame)
                selected.append(r)
                break
    return selected

def dhash(im):
    values = list(im.convert('L').resize((9,8)).getdata())
    return sum((values[y*9+x] > values[y*9+x+1]) << (y*8+x) for y in range(8) for x in range(8))

def download(item):
    r, revision = item
    url = f'{BASE}/resolve/{revision}/data/train/{r["file_name"]}'
    for attempt in range(3):
        try:
            response = requests.get(url, timeout=45)
            response.raise_for_status()
            payload = response.content
            with Image.open(io.BytesIO(payload)) as im:
                im.load()
                assert im.size == (640,640)
                perceptual_hash = dhash(im)
            return r, url, payload, perceptual_hash
        except Exception:
            if attempt == 2:
                raise
            time.sleep(1)

if __name__ == '__main__':
    if OUT.exists():
        raise SystemExit(f'Refusing to overwrite {OUT}')
    metadata = requests.get(f'https://huggingface.co/api/datasets/{REPO}',timeout=30).json()
    revision = metadata['sha']
    response = requests.get(f'{BASE}/resolve/{revision}/data/train/metadata.jsonl',timeout=60)
    response.raise_for_status()
    rows = [json.loads(line) for line in response.text.splitlines() if line.strip()]
    OUT.mkdir(parents=True)
    provenance = []
    hashes = []
    sha_seen = set()
    for bucket,cls,head in [('CT',1,2),('T',3,4)]:
        pool = candidates(rows,cls,head)
        kept = 0
        with ThreadPoolExecutor(max_workers=6) as executor:
            for start in range(0,len(pool),20):
                if kept >= 50:
                    break
                for r,url,payload,ph in executor.map(download,[(r,revision) for r in pool[start:start+20]]):
                    sha = hashlib.sha256(payload).hexdigest()
                    if sha in sha_seen or any((ph ^ old).bit_count() < 9 for old in hashes):
                        continue
                    kept += 1
                    sha_seen.add(sha); hashes.append(ph)
                    name = f'web_{bucket}_{kept:03d}.png'
                    (OUT / name).write_bytes(payload)
                    provenance.append({'file':name,'suggested_team_from_source':bucket,
                        'source_url':url,'source_dataset':REPO,'revision':revision,
                        'source_split':'train','source_session':r['source_session'],
                        'original_file':r['file_name'],'sha256':sha,'width':640,'height':640,
                        'source_annotations_for_reference_only':r['objects']})
                    print(f'{bucket}: {kept}/50',flush=True)
                    if kept == 50:
                        break
        assert kept == 50, f'Not enough distinct images for {bucket}: {kept}'
    (OUT / 'manifest.json').write_text(json.dumps(provenance,indent=2),encoding='utf-8')
    card = requests.get(f'{BASE}/resolve/{revision}/README.md',timeout=30)
    card.raise_for_status()
    (OUT / 'SOURCE_DATASET_README.md').write_text(card.text,encoding='utf-8')
    for page in range(5):
        canvas = Image.new('RGB',(1400,4*305),'white')
        d = ImageDraw.Draw(canvas)
        for i,row in enumerate(provenance[page*20:(page+1)*20]):
            im = Image.open(OUT / row['file']); im.thumbnail((280,280))
            x,y = i%5*280,i//5*305
            canvas.paste(im,(x,y));d.text((x+3,y+282),row['file'],fill='black')
        canvas.save(ROOT / f'work/web-preview-{page+1}.jpg')
    print('Saved:',OUT)
    print('Sessions:',dict(Counter(r['source_session'] for r in provenance)))
