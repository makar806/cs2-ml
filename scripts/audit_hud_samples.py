"""Inspect the four provided AK screenshots; all outputs remain private/local.

The first screenshot supplies templates; the other three are correlated
same-session probes, not independent validation or evidence for reload/zoom.
"""
import argparse
import hashlib
import json
from pathlib import Path
import statistics
import time
import cv2
import numpy as np
from mvp_weapon_visual import WeaponVisual
from mvp_killfeed import VisualKillFeed,calibrate


def audit(source,output):
    paths=sorted(Path(source).glob('*.jpg'))
    if len(paths)!=4:
        raise ValueError('Expected the four supplied screenshots')
    output=Path(output)
    if output.exists():
        raise FileExistsError('Choose a new output directory')
    icons=output/'weapons'
    icons.mkdir(parents=True)
    image=cv2.imread(str(paths[0]))
    if image.shape[:2]!=(1080,1440):
        raise ValueError('These crop annotations apply only to the supplied 1440x1080 screenshots')
    pixels=image[778:815,1288:1411]
    color=np.median(pixels[pixels.max(axis=2)>220],axis=0).tolist()
    (icons/'palette.json').write_text(json.dumps({'bgr':color}),encoding='utf-8')
    for name,(x1,y1,x2,y2) in {
        'ak47':(1288,778,1411,815),
        'unknown_pistol':(1335,854,1412,894),
        'knife':(1325,938,1411,965),
    }.items():
        mask=(image[y1:y2,x1:x2].max(axis=2)>190).astype('uint8')*255
        cv2.imwrite(str(icons/f'{name}.png'),mask)
    sensor=WeaponVisual(icons,interval=0.)
    feed=VisualKillFeed(calibrate(image))
    rows=[]
    for path in paths:
        image=cv2.imread(str(path))
        timings=[]
        for _ in range(20):
            start=time.perf_counter()
            name=sensor.classify(image)
            timings.append((time.perf_counter()-start)*1000)
        matches=feed.own_rows(image)
        rows.append(dict(image=path.name,sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                         recognized=name,median_ms=statistics.median(timings),
                         own_killfeed_rows=len(matches)))
    result=dict(calibration_image=paths[0].name,frames=rows,
                independent_validation=False,
                unavailable=['other active weapons','weapon transitions','reload','ammo changes','sniper zoom'],
                note='Templates derive from training screenshots; no detector retraining or holdout changes')
    (output/'report.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--source',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args()
    print(json.dumps(audit(args.source,args.output),indent=2))
