"""Merge reviewed FOCUS labels into train with hash and match leakage checks."""
import collections
import hashlib
import json
from pathlib import Path
import shutil
import yaml
from prepare_pose import convert

ROOT=Path(__file__).resolve().parents[1]
if __name__=='__main__':
    base=ROOT/'data/pose-v5';dest=ROOT/'data/pose-v7';assert not dest.exists()
    snapshot=sorted((ROOT/'data/exports').glob('*/project-4.json'))[-1]
    tasks=json.loads(snapshot.read_text(encoding='utf-8'));assert len(tasks)==40
    selected=json.loads((ROOT.parent/'2026-10-17/new-chat/outputs/sixth-experiment/focus-review/selected.json').read_text())
    lookup={r['source_task_id']:r for r in selected}
    manifest=json.loads((base/'manifest.json').read_text())
    excluded={int(k) for k,v in manifest['new_match_assignment'].items() if v!='train'}
    hashes={r['sha256'] for r in manifest['images']}
    records=manifest['images'];shutil.copytree(base,dest)
    added_objects=negatives=0
    for task in tasks:
        tid=task['data']['source_task_id'];source=lookup[tid]
        assert source['match_id'] not in excluded
        path=ROOT/'.label-studio/media'/task['data']['image'].split('/data/')[1]
        digest=hashlib.sha256(path.read_bytes()).hexdigest()
        assert digest==source['sha256'] and digest not in hashes
        hashes.add(digest)
        name=source['file'];lines=convert(task['annotations'][0]['result'])
        shutil.copy2(path,dest/'images/train'/name)
        (dest/'labels/train'/(Path(name).stem+'.txt')).write_text('\n'.join(lines)+('\n' if lines else ''),encoding='utf-8')
        records.append({'task':task['id'],'annotation_project_id':4,'source_task_id':tid,'source':str(path),'split':'train','group':'multi-match-'+str(source['match_id']),'sha256':digest,'map':source['map']})
        added_objects+=len(lines);negatives+=not bool(lines)
    for split in ('val','newcheck','test','check'):
        for folder in ('images','labels'):
            for path in (base/folder/split).iterdir():assert path.read_bytes()==(dest/folder/split/path.name).read_bytes()
    stats={}
    for split in ('train','val','newcheck','check','test'):
        labels=list((dest/'labels'/split).glob('*.txt'))
        counts=collections.Counter(int(line.split()[0]) for p in labels for line in p.read_text().splitlines())
        stats[split]={'images':len(labels),'T':counts[0],'CT':counts[1],'empty':sum(not p.read_text().strip() for p in labels)}
    config=yaml.safe_load((ROOT/'configs/pose-v5.yaml').read_text());config['path']=str(dest)
    (ROOT/'configs/pose-v7.yaml').write_text(yaml.safe_dump(config,sort_keys=False))
    (dest/'manifest.json').write_text(json.dumps({'images':records,'stats':stats,'focus_snapshot':str(snapshot),'base':'pose-v5','focus_added_objects':added_objects,'focus_negatives':negatives,'new_match_assignment':manifest['new_match_assignment'],'note':'FOCUS replaces no existing image: all 40 source hashes verified absent. Do not ingest original project copies later. All held-out labels and images preserved byte-for-byte.'},indent=2))
    print(json.dumps({'stats':stats,'added_objects':added_objects,'added_negatives':negatives},indent=2))
