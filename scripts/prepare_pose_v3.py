"""Preserve v2 splits and add the entire hard-video batch to train only."""
import collections
import hashlib
import json
from pathlib import Path
import shutil
import yaml
from prepare_pose import convert

ROOT=Path(__file__).resolve().parents[1]

if __name__=='__main__':
    base=ROOT/'data/pose-v2'
    dest=ROOT/'data/pose-v3'
    assert not dest.exists(), 'Refusing to overwrite dataset'
    manifest=json.loads((base/'manifest.json').read_text())
    snapshot=sorted((ROOT/'data/exports').glob('*/ML_CS.json'))[-1]
    tasks=json.loads(snapshot.read_text(encoding='utf-8'))
    old={r['task']:r for r in manifest['images']}
    hard={r['file']:r for r in json.loads((ROOT/'data/raw/hard_batch_001/manifest.json').read_text())}
    shutil.copytree(base,dest)
    records=manifest['images']
    hashes={r['sha256'] for r in records}
    negatives=missing_heads=new_objects=new_images=0
    for task in tasks:
        results=task['annotations'][0]['result']
        lines=convert(results)
        if task['id'] in old:
            record=old[task['id']]
            matches=list((base/'labels'/record['split']).glob(str(task['id'])+'_*.txt'))
            if not matches:
                source_name=Path(record['source']).name
                name=source_name[source_name.index('web_'):]
                matches=[base/'labels'/record['split']/(Path(name).stem+'.txt')]
            assert matches[0].read_text(encoding='utf-8').splitlines()==lines, f'Existing annotation changed: {task["id"]}'
            continue
        path=ROOT/'.label-studio/media'/task['data']['image'].split('/data/')[1]
        name=path.name[path.name.index('hard_'):]
        source_record=hard[name]
        payload=path.read_bytes()
        digest=hashlib.sha256(payload).hexdigest()
        assert digest==source_record['sha256'] and digest not in hashes
        hashes.add(digest)
        shutil.copy2(path,dest/'images/train'/name)
        (dest/'labels/train'/(Path(name).stem+'.txt')).write_text('\n'.join(lines)+ ('\n' if lines else ''),encoding='utf-8')
        negatives+=not bool(lines);new_images+=1;new_objects+=len(lines)
        missing_heads+=sum(float(line.split()[-1])==0 for line in lines)
        records.append({'task':task['id'],'source':str(path),'split':'train','group':source_record['source_session'],'sha256':digest})
    assert new_images==40 and new_objects==52 and negatives==10 and missing_heads==1
    for split in ('val','test'):
        for folder in ('images','labels'):
            for original in (base/folder/split).iterdir():
                assert original.read_bytes()==(dest/folder/split/original.name).read_bytes()
    stats={}
    for split in ('train','val','test'):
        labels=list((dest/'labels'/split).glob('*.txt'))
        counter=collections.Counter()
        for file in labels:
            counter.update(int(line.split()[0]) for line in file.read_text().splitlines())
        stats[split]={'images':len(labels),'classes':{'T':counter[0],'CT':counter[1]},'empty':sum(not p.read_text().strip() for p in labels)}
    assert not {r['group'] for r in records if r['split']=='train'} & {r['group'] for r in records if r['split']!='train'}
    config=yaml.safe_load((ROOT/'configs/pose-v2.yaml').read_text())
    config['path']=str(dest)
    (ROOT/'configs/pose-v3.yaml').write_text(yaml.safe_dump(config,sort_keys=False),encoding='utf-8')
    (dest/'manifest.json').write_text(json.dumps({'snapshot':str(snapshot),'images':records,'stats':stats,'new_negative_images':negatives,'missing_head_points':missing_heads,'recording_independence':'Hard-video session is entirely train. Do not report performance on that recording as unseen-data performance.'},indent=2),encoding='utf-8')
    print(json.dumps(stats,indent=2))
