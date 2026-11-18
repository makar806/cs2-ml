"""Deterministic, pixel-only holdout collection (never reads ticks or event tables)."""

import argparse
from concurrent.futures import ThreadPoolExecutor
from hashlib import sha256
import json
from pathlib import Path
from urllib.request import urlopen

from protocol import ROOT, fingerprint, load


API = 'https://huggingface.co/api/datasets/blanchon/opencs2_dataset/tree/main/'
BASE = 'https://huggingface.co/datasets/blanchon/opencs2_dataset/resolve/main/'


def directories(path):
    with urlopen(API+path+'?limit=1000', timeout=45) as response:
        return [row['path'] for row in json.load(response) if row['type'] == 'directory']


def fetch_row(row, destination):
    if not row['url'].startswith(BASE) or Path(row['file']).name != row['file']:
        raise ValueError('Unexpected source URL or filename')
    target = destination/row['file']
    if target.is_file() and target.stat().st_size == row['bytes'] and fingerprint(target) == row['sha256']:
        return
    temporary = target.with_suffix('.part')
    try:
        with urlopen(row['url'], timeout=60) as source, temporary.open('wb') as out:
            size = 0
            while chunk := source.read(1024*1024):
                size += len(chunk)
                if size > row['bytes']:
                    raise ValueError('Source exceeds recorded length')
                out.write(chunk)
        if size != row['bytes'] or fingerprint(temporary) != row['sha256']:
            raise ValueError('Source does not match saved manifest')
        temporary.replace(target)
    finally:
        temporary.unlink(missing_ok=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--destination', type=Path, required=True)
    parser.add_argument('--count', type=int, default=12)
    parser.add_argument('--manifest', type=Path, help='Replay exact URLs and hashes; do not rediscover sources')
    args = parser.parse_args()
    args.destination.mkdir(parents=True, exist_ok=True)
    if args.manifest:
        for row in json.loads(args.manifest.read_text()):
            fetch_row(row, args.destination)
            print('verified', row['case'], flush=True)
        return
    excluded = {row['match_id'] for row in load('sources.json')}
    candidates = [path for path in directories('rounds') if int(path.split('=')[-1]) not in excluded]
    candidates.sort(key=lambda path: sha256(('early-2026-11-18:'+path).encode()).hexdigest())
    selection = dict(seed='early-2026-11-18', rule='sha256 order, first alphabetical map, round 07, player 02',
                     excluded_published_matches=sorted(excluded), matches=candidates[:args.count],
                     historical_training_disjointness='unverifiable: full V18 historical training manifest unavailable')
    (args.destination/'selection.json').write_text(json.dumps(selection, indent=2)+'\n')
    def download(item):
        index, match = item
        maps = sorted(directories(match))
        filename = maps[0]+'/round=07/player=02/video.mp4'
        target = args.destination/f'{801+index}.mp4'
        if not target.is_file():
            temporary = target.with_suffix('.part')
            with urlopen(BASE+filename, timeout=60) as source, temporary.open('wb') as out:
                size = 0
                while chunk := source.read(1024*1024):
                    size += len(chunk)
                    if size > 250*1024*1024:
                        raise ValueError('Unexpectedly large source')
                    out.write(chunk)
            temporary.replace(target)
        row = dict(case=801+index, match_id=int(match.split('=')[-1]), map=maps[0].split('=')[-1],
                   url=BASE+filename, bytes=target.stat().st_size, sha256=fingerprint(target), file=target.name)
        print(row['case'], row['match_id'], row['map'], row['bytes'], flush=True)
        return row
    with ThreadPoolExecutor(max_workers=3) as pool:
        rows = list(pool.map(download, enumerate(candidates[:args.count])))
    (args.destination/'sources.json').write_text(json.dumps(rows, indent=2)+'\n')


if __name__ == '__main__':
    main()
