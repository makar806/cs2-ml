"""Add web annotations while preserving the original test and session grouping."""
from pathlib import Path
import collections
import hashlib
import json
import shutil
import yaml
from PIL import Image
from prepare_pose import convert

ROOT = Path(__file__).resolve().parents[1]

if __name__ == '__main__':
    old = json.loads((ROOT/'data/pose-v1/manifest.json').read_text())
    web = {r['file']:r for r in json.loads((ROOT/'data/raw/internet_batch_001/manifest.json').read_text())}
    source = sorted((ROOT/'data/exports').glob('*/ML_CS.json'))[-1]
    tasks = json.loads(source.read_text(encoding='utf-8'))
    dest = ROOT/'data/pose-v2'
    assert not dest.exists(), 'Refusing to overwrite dataset'
    old_by_id = {r['task']:r for r in old['images']}
    val_session = '20260725-185119-fc031b'
    stats = {s:{'images':0,'classes':collections.Counter()} for s in ('train','val','test')}
    seen = set()
    manifest = []
    for t in tasks:
        path = ROOT/'.label-studio/media'/t['data']['image'].split('/data/')[1]
        if t['id'] in old_by_id:
            record = old_by_id[t['id']]
            split, group = record['split'], record['group']
            original = next((ROOT/'data/pose-v1/images'/split).glob(str(t['id'])+'_*.jpg'))
            name = original.stem
        else:
            filename = path.name[path.name.index('web_'):]
            record = web[filename]
            group = record['source_session']
            split = 'val' if group == val_session else 'train'
            name = filename[:-4]
            assert path.read_bytes() == (ROOT/'data/raw/internet_batch_001'/filename).read_bytes()
        lines = convert(t['annotations'][0]['result'])
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        assert digest not in seen
        seen.add(digest)
        with Image.open(path) as im:
            width,height = im.size
        assert (width,height) in ((1440,1080),(640,640))
        image_dir,label_dir = dest/'images'/split,dest/'labels'/split
        image_dir.mkdir(parents=True,exist_ok=True);label_dir.mkdir(parents=True,exist_ok=True)
        shutil.copy2(path,image_dir/(name+path.suffix))
        target_label = label_dir/(name+'.txt')
        target_label.write_text('\n'.join(lines)+ ('\n' if lines else ''),encoding='utf-8')
        if split == 'test':
            original_label = ROOT/'data/pose-v1/labels/test'/target_label.name
            assert target_label.read_bytes() == original_label.read_bytes(), 'Original test annotations changed'
            assert path.read_bytes() == original.read_bytes(), 'Original test image changed'
        stats[split]['images'] += 1
        stats[split]['classes'].update(r['value']['rectanglelabels'][0] for r in t['annotations'][0]['result'] if r['type']=='rectanglelabels')
        manifest.append({'task':t['id'],'source':str(path),'split':split,'group':group,'sha256':digest})
    assert stats['test']['images']==12
    train_groups = {r['group'] for r in manifest if r['split']=='train'}
    val_groups = {r['group'] for r in manifest if r['split']=='val'}
    assert not train_groups & val_groups
    config = {'path':str(dest),'train':'images/train','val':'images/val','test':'images/test','names':{0:'T',1:'CT'},'kpt_shape':[1,3],'flip_idx':[0]}
    (ROOT/'configs/pose-v2.yaml').write_text(yaml.safe_dump(config,sort_keys=False),encoding='utf-8')
    (dest/'manifest.json').write_text(json.dumps({'snapshot':str(source),'stats':stats,'web_val_session':val_session,'images':manifest},indent=2),encoding='utf-8')
    print(json.dumps(stats,indent=2))
