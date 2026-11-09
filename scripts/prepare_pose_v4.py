"""Preserve v3 and add active frames with a separate match-level check split."""
import collections
import hashlib
import json
from pathlib import Path
import shutil
import yaml
from prepare_pose import convert

ROOT = Path(__file__).resolve().parents[1]

if __name__ == '__main__':
    base, dest = ROOT/'data/pose-v3', ROOT/'data/pose-v4'
    assert not dest.exists(), 'Refusing to overwrite dataset'
    manifest = json.loads((base/'manifest.json').read_text())
    snapshot = sorted((ROOT/'data/exports').glob('*/ML_CS.json'))[-1]
    tasks = json.loads(snapshot.read_text(encoding='utf-8'))
    batch = json.loads((ROOT/'data/raw/active_hard_batch_001/manifest.json').read_text())
    lookup = {r['file']: r for r in batch['images']}
    shutil.copytree(base, dest)
    records = manifest['images']
    old = {r['task'] for r in records}
    seen = set()
    for task in tasks:
        if task['id'] in old:
            continue
        path = ROOT/'.label-studio/media'/task['data']['image'].split('/data/')[1]
        name = path.name[path.name.index('active_'):]
        source = lookup[name]
        assert name not in seen
        seen.add(name)
        split = source['split']
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        assert digest == source['sha256']
        lines = convert(task['annotations'][0]['result'])
        for folder in ('images', 'labels'):
            (dest/folder/split).mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, dest/'images'/split/name)
        (dest/'labels'/split/(Path(name).stem+'.txt')).write_text('\n'.join(lines)+('\n' if lines else ''), encoding='utf-8')
        records.append({'task': task['id'], 'source': str(path), 'split': split,
                        'group': 'active-match-'+str(source['match_id']), 'sha256': digest})
    assert len(seen) == 40
    assert not {r['match_id'] for r in lookup.values() if r['split']=='train'} & {r['match_id'] for r in lookup.values() if r['split']=='check'}
    for split in ('train', 'val', 'test'):
        for folder in ('images', 'labels'):
            for original in (base/folder/split).iterdir():
                assert original.read_bytes() == (dest/folder/split/original.name).read_bytes()
    stats = {}
    for split in ('train', 'val', 'test', 'check'):
        labels = list((dest/'labels'/split).glob('*.txt'))
        counts = collections.Counter(int(line.split()[0]) for p in labels for line in p.read_text().splitlines())
        stats[split] = {'images': len(labels), 'classes': {'T': counts[0], 'CT': counts[1]}, 'empty': sum(not p.read_text().strip() for p in labels)}
    config = yaml.safe_load((ROOT/'configs/pose-v3.yaml').read_text())
    config['path'] = str(dest)
    (ROOT/'configs/pose-v4.yaml').write_text(yaml.safe_dump(config, sort_keys=False), encoding='utf-8')
    check_config = dict(config, test='images/check')
    (ROOT/'configs/pose-v4-check.yaml').write_text(yaml.safe_dump(check_config, sort_keys=False), encoding='utf-8')
    (dest/'manifest.json').write_text(json.dumps({'snapshot': str(snapshot), 'images': records, 'stats': stats, 'limitation': batch['limitation']}, indent=2), encoding='utf-8')
    print(json.dumps(stats, indent=2))
