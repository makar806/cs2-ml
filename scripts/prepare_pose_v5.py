"""Add submitted multimap labels, retaining whole matches for val and check."""
import collections
import hashlib
import json
from pathlib import Path
import shutil
import yaml
from prepare_pose import convert

ROOT=Path(__file__).resolve().parents[1]
if __name__=='__main__':
    base=ROOT/'data/pose-v4';dest=ROOT/'data/pose-v5'
    assert not dest.exists()
    snapshot=sorted((ROOT/'data/exports').glob('*/ML_CS.json'))[-1]
    tasks=json.loads(snapshot.read_text(encoding='utf-8'))
    old=json.loads((base/'manifest.json').read_text())
    known={r['task'] for r in old['images']}
    source=json.loads((ROOT/'data/raw/multimap_batch_001/manifest.json').read_text())
    lookup={r['file']:r for r in source['images']}
    clean=[];excluded=[]
    for task in tasks:
        if task['id'] in known:continue
        path=ROOT/'.label-studio/media'/task['data']['image'].split('/data/')[1]
        name=path.name[path.name.index('multi_'):]
        try: lines=convert(task['annotations'][0]['result'])
        except (AssertionError,KeyError) as error:
            excluded.append({'task':task['id'],'reason':str(error)});continue
        clean.append((task,path,lookup[name],lines))
    by_map=collections.defaultdict(set)
    for task,path,row,lines in clean:by_map[row['map']].add(row['match_id'])
    assignment={}
    for map_name,matches in sorted(by_map.items()):
        ordered=sorted(matches,key=lambda mid:hashlib.sha256(str(mid).encode()).hexdigest())
        assert len(ordered)>=3,(map_name,len(ordered))
        for idx,mid in enumerate(ordered):
            split='newcheck' if idx==0 else 'val' if idx==1 else 'train'
            if mid in assignment: assert assignment[mid]==split
            assignment[mid]=split
    shutil.copytree(base,dest)
    for folder in ('images','labels'):(dest/folder/'newcheck').mkdir(parents=True)
    records=old['images']
    for task,path,row,lines in clean:
        split=assignment[row['match_id']];name=row['file']
        assert hashlib.sha256(path.read_bytes()).hexdigest()==row['sha256']
        shutil.copy2(path,dest/'images'/split/name)
        (dest/'labels'/split/(Path(name).stem+'.txt')).write_text('\n'.join(lines)+('\n' if lines else ''),encoding='utf-8')
        records.append({'task':task['id'],'source':str(path),'split':split,'group':'multi-match-'+str(row['match_id']),'sha256':row['sha256'],'map':row['map']})
    stats={}
    for split in ('train','val','test','check','newcheck'):
        labels=list((dest/'labels'/split).glob('*.txt'))
        counts=collections.Counter(int(line.split()[0]) for p in labels for line in p.read_text().splitlines())
        stats[split]={'images':len(labels),'T':counts[0],'CT':counts[1],'empty':sum(not p.read_text().strip() for p in labels)}
    for split in ('test','check'):
        for folder in ('images','labels'):
            for p in (base/folder/split).iterdir():assert p.read_bytes()==(dest/folder/split/p.name).read_bytes()
    config=yaml.safe_load((ROOT/'configs/pose-v4.yaml').read_text());config['path']=str(dest)
    (ROOT/'configs/pose-v5.yaml').write_text(yaml.safe_dump(config,sort_keys=False))
    (dest/'manifest.json').write_text(json.dumps({'snapshot':str(snapshot),'images':records,'stats':stats,'excluded':excluded,'new_match_assignment':assignment,'note':'Only submitted manual annotations used. Match-level newcheck is excluded from training and checkpoint selection. Old third-party match identity remains unknown.'},indent=2))
    print(json.dumps({'stats':stats,'excluded':excluded,'new_images':len(clean),'map_matches':{k:len(v) for k,v in by_map.items()}},indent=2))
