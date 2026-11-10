"""Build local HUD templates from explicitly annotated saved screenshots.

No capture, game file access or publishing. See docs/v3.32.md for the schema.
"""
import argparse
import hashlib
import json
from pathlib import Path
import cv2
import numpy as np
from mvp_hud import crop, white_mask


def build(spec_path, output):
    spec_path=Path(spec_path)
    spec=json.loads(spec_path.read_text(encoding='utf-8'))
    output=Path(output)
    if output.exists():
        raise FileExistsError('Choose a new directory; existing calibration is never overwritten')
    samples=[]
    palette=None
    names=set()
    for item in spec['samples']:
        name=item['name']
        valid=(name in ('ready','reloading') or name in {f'digit-{d}' for d in range(10)}
               or (name.startswith('weapon_') and name[7:].replace('_','').isalnum()))
        if not valid or name in names:
            raise ValueError(f'Invalid or duplicate template: {name}')
        path=spec_path.parent/item['image']
        image=cv2.imread(str(path))
        if image is None:
            raise ValueError(f'Cannot read screenshot for {name}')
        rect=item['rect']
        if not (len(rect)==4 and 0<=rect[0]<rect[2]<=1 and 0<=rect[1]<rect[3]<=1):
            raise ValueError(f'Invalid normalized crop for {name}')
        mask=white_mask(crop(image,rect))
        if name.startswith('weapon_') and palette is None:
            pixels=crop(image,rect)
            bright=pixels[pixels.max(axis=2)>220]
            if len(bright):palette=np.median(bright,axis=0).tolist()
        if mask.size<20 or not mask.any():
            raise ValueError(f'Empty/unusable HUD crop: {name}')
        digest=hashlib.sha256(path.read_bytes()).hexdigest()
        samples.append((name,mask,digest,rect))
        names.add(name)
    required={'ready','reloading'}|{f'digit-{d}' for d in range(10)}
    if not required<=names or sum(n.startswith('weapon_') for n in names)<2:
        raise ValueError('Need 10 digits, ready/reloading and at least 2 distinct weapon icons')
    for key in ('ammo_roi','state_roi'):
        r=spec[key]
        if not (len(r)==4 and 0<=r[0]<r[2]<=1 and 0<=r[1]<r[3]<=1):
            raise ValueError(f'Invalid {key}')
    (output/'weapons').mkdir(parents=True)
    if palette is not None:
        (output/'weapons/palette.json').write_text(json.dumps({'bgr':palette}),encoding='utf-8')
    provenance=[]
    for name,mask,digest,rect in samples:
        path=output/'weapons'/f'{name[7:]}.png' if name.startswith('weapon_') else output/f'{name}.png'
        if not cv2.imwrite(str(path),mask):
            raise OSError('Failed to write calibration')
        provenance.append(dict(template=str(path.relative_to(output)),source_sha256=digest,rect=rect))
    manifest=dict(schema_version=1,ammo_roi=spec['ammo_roi'],state_roi=spec['state_roi'],
                  provenance=provenance,validation='Local calibration; real HUD holdout still required')
    (output/'manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    return manifest


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--annotations',required=True,type=Path)
    p.add_argument('--output',default=Path('work/hud'),type=Path)
    args=p.parse_args()
    result=build(args.annotations,args.output)
    print(f"Created {len(result['provenance'])} local templates in {args.output}")
