"""Mine diverse pre-engagement full-resolution frames, split by match before annotation."""
import collections
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
from pathlib import Path
import random
import time
from urllib.parse import quote
import zipfile
import cv2
import numpy as np
import polars as pl
import requests
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
DEST = ROOT/'data/raw/multimap_batch_001'
WORK = ROOT/'work/multimap-downloads'
OUT = ROOT.parent/'2026-10-17/new-chat/outputs/multimap-dataset'
EXCLUDE = {2392746,2392829,2392620,2392440,2392753,2392915}

def select():
    kills = pl.read_parquet(ROOT/'work/public-kills.parquet').filter(~pl.col('wallbang'))
    events = {}
    for row in kills.to_dicts():
        key = (row['match_id'],str(row['map_name']),row['round'],row['attacker_player_slot'])
        events.setdefault(key, []).append(row)
    candidates = []
    for row in pl.read_parquet(ROOT/'work/pov_rounds.parquet').to_dicts():
        key = (row['match_id'],str(row['map_name']),row['round'],row['player_slot'])
        ev = events.get(key, [])
        if row['match_id'] in EXCLUDE or len(ev)<3 or row['media_bytes']>85_000_000 or not 12<row['duration_s']<100:
            continue
        row.pop('created_at',None)
        row['events'] = ev
        candidates.append(row)
    rng = random.Random(20261019)
    rng.shuffle(candidates)
    selected = []
    for map_name in sorted({str(r['map_name']) for r in candidates}):
        matches = {}
        for row in candidates:
            if str(row['map_name']) == map_name:
                matches.setdefault(row['match_id'], []).append(row)
        # Two distinct rounds from each match, preferably opposite POV sides.
        valid = []
        for mid, rows in matches.items():
            pair = next(((a,b) for a in rows for b in rows if a['round']!=b['round'] and a['player_side']!=b['player_side']),None)
            if pair: valid.append((mid,pair))
        assert len(valid)>=10, (map_name,len(valid))
        for index,(mid,pair) in enumerate(valid[:10]):
            split = 'train' if index<7 else 'val' if index<8 else 'test'
            for row in pair:
                row['split'] = split
                selected.append(row)
    return selected

def extract(row):
    source = row['video']['path'].split('@main/')[1]
    url = 'https://huggingface.co/datasets/blanchon/opencs2_dataset/resolve/main/'+quote(source,safe='/=')
    video = WORK/(row['media_id']+'.mp4')
    if not video.exists() or video.stat().st_size!=row['media_bytes']:
        for attempt in range(3):
            try:
                with requests.get(url,stream=True,timeout=(25,90)) as response:
                    response.raise_for_status()
                    with video.open('wb') as handle:
                        for chunk in response.iter_content(1024*1024): handle.write(chunk)
                assert video.stat().st_size==row['media_bytes']
                break
            except Exception:
                if attempt==2: raise
                time.sleep(2)
    digest = hashlib.sha256(video.read_bytes()).hexdigest()
    cap = cv2.VideoCapture(str(video))
    fps = cap.get(cv2.CAP_PROP_FPS)
    assert fps>0
    times = []
    # At least 0.9s between frames; prioritize before kills, never post-kill corpse frames.
    for offset in (-.35,-1.6,-3.0,-4.5):
        for event in sorted(row['events'],key=lambda e:e['event_seconds']):
            sec = max(0,event['event_seconds']+offset)
            if sec<row['duration_s']-.1 and all(abs(sec-other)>.9 for other in times): times.append(sec)
    records, hashes = [], []
    for sec in times:
        cap.set(cv2.CAP_PROP_POS_FRAMES,round(sec*fps)); ok,frame=cap.read()
        if not ok: continue
        assert frame.shape[:2]==(720,1280)
        small=cv2.resize(cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY),(32,32))
        dh = cv2.resize(small,(17,16)); dh=dh[:,1:]>dh[:,:-1]
        if any(np.count_nonzero(dh!=old)<10 for old in hashes): continue
        if np.std(small)<7: continue
        hashes.append(dh)
        frame_id=round(sec*fps)
        filename=f'multi_{row["split"]}_{row["media_id"]}_f{frame_id:05d}.jpg'
        path=DEST/'images'/row['split']/filename
        assert cv2.imwrite(str(path),frame,[cv2.IMWRITE_JPEG_QUALITY,97])
        nearest=min(row['events'],key=lambda e:abs(e['event_seconds']-sec))
        records.append({'file':filename,'split':row['split'],'map':str(row['map_name']), 'match_id':row['match_id'], 'media_id':row['media_id'], 'pov_side':str(row['player_side']), 'frame':frame_id,'fps':fps,'seconds':frame_id/fps,'source_url':url,'source_video_sha256':digest,'sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'near_event_distance_metadata':nearest['distance'],'near_event_through_smoke_metadata':nearest['through_smoke'],'selection':'pre-kill timing, not proof that a target is visible'})
        if len(records)==8: break
    cap.release()
    assert video.resolve().parent==WORK.resolve()
    video.unlink()
    return records

if __name__=='__main__':
    for split in ('train','val','test'): (DEST/'images'/split).mkdir(parents=True,exist_ok=True)
    WORK.mkdir(parents=True,exist_ok=True);OUT.mkdir(parents=True,exist_ok=True)
    plan_path=DEST/'source-plan.json'
    plan=json.loads(plan_path.read_text()) if plan_path.exists() else select()
    assigned = {}
    for row in plan:
        row['split'] = assigned.setdefault(row['match_id'],row['split'])
    plan_path.write_text(json.dumps(plan,indent=2),encoding='utf-8')
    print('Selected',len(plan),'POV rounds; source download GiB',round(sum(r['media_bytes'] for r in plan)/1024**3,2),flush=True)
    records=[]; failures=[]
    with ThreadPoolExecutor(max_workers=4) as pool:
        jobs={pool.submit(extract,row):row for row in plan}
        for i,job in enumerate(as_completed(jobs),1):
            row=jobs[job]
            try: records.extend(job.result())
            except Exception as error: failures.append({'media_id':row['media_id'],'error':str(error)})
            (DEST/'progress.json').write_text(json.dumps({'completed_rounds':i,'total_rounds':len(plan),'images':len(records),'failures':failures},indent=2))
            print(f'{i}/{len(plan)} rounds, {len(records)} frames, {len(failures)} errors',flush=True)
    records.sort(key=lambda r:r['file'])
    retained={r['file'] for r in records}
    for path in (DEST/'images').glob('*/*.jpg'):
        if path.name not in retained:
            assert DEST.resolve() in path.resolve().parents
            path.unlink()
    for a in ('train','val','test'):
        for b in ('train','val','test'):
            if a!=b: assert not {r['match_id'] for r in records if r['split']==a}&{r['match_id'] for r in records if r['split']==b}
    assert len({r['sha256'] for r in records})==len(records)
    stats=collections.Counter(r['map'] for r in records)
    manifest={'source':'https://huggingface.co/datasets/blanchon/opencs2_dataset','author':'blanchon','license_declared_by_source':'CC BY 4.0','modifications':'Full resolution frames extracted, JPEG quality 97, temporal and perceptual deduplication within round. No model-generated labels.','split_policy':'7 training, 1 validation, 2 final test matches per map. No match crosses splits. Existing six active-video matches excluded. Historical still-image overlap unknown.','maps':dict(stats),'splits':dict(collections.Counter(r['split'] for r in records)),'images':records,'failures':failures}
    (DEST/'manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    (OUT/'manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    for split in ('train','val','test'):
        with zipfile.ZipFile(OUT/f'multimap-{split}.zip','w',zipfile.ZIP_STORED) as archive:
            for r in records:
                if r['split']==split: archive.write(DEST/'images'/split/r['file'],r['file'])
    for map_name in sorted(stats):
        subset=[r for r in records if r['map']==map_name]
        subset=subset[::max(1,len(subset)//20)][:20]
        canvas=Image.new('RGB',(1280,5*200),'white'); draw=ImageDraw.Draw(canvas)
        for i,r in enumerate(subset):
            im=Image.open(DEST/'images'/r['split']/r['file']); im.thumbnail((320,180))
            x,y=i%4*320,i//4*200;canvas.paste(im,(x,y));draw.text((x+3,y+181),r['media_id'],fill='black')
        canvas.save(OUT/f'preview-{map_name}.jpg')
    print(json.dumps({'maps':dict(stats),'splits':manifest['splits'],'failures':failures},indent=2),flush=True)
