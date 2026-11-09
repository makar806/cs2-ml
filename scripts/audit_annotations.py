"""Read-only audit and JSON snapshot of the ML_CS annotations."""
import collections
import argparse
import json
import sqlite3
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
parser=argparse.ArgumentParser()
parser.add_argument('--project-id',type=int)
args=parser.parse_args()
db = sqlite3.connect(f'file:{(ROOT / ".label-studio/label_studio.sqlite3").as_posix()}?mode=ro', uri=True)
if args.project_id is None:
    project_id,title=db.execute('SELECT id,title FROM project WHERE title=?',('ML_CS',)).fetchone()
else:
    project_id,title=db.execute('SELECT id,title FROM project WHERE id=?',(args.project_id,)).fetchone()
rows = db.execute('SELECT task.id, task.data, task_completion.result FROM task JOIN task_completion ON task.id=task_completion.task_id WHERE task.project_id=? AND task_completion.was_cancelled=0', (project_id,)).fetchall()
issues = []
missing_heads = []
classes = collections.Counter()
snapshot = []
for task, data, raw in rows:
    results = json.loads(raw)
    snapshot.append({'id': task, 'data': json.loads(data), 'annotations': [{'result': results}]})
    boxes = {r['id']: r for r in results if r.get('type') == 'rectanglelabels'}
    heads = [r for r in results if r.get('type') == 'keypointlabels']
    for box in boxes.values():
        classes.update(box['value']['rectanglelabels'])
        count = sum(h.get('parentID') == box['id'] for h in heads)
        if count == 0:
            missing_heads.append({'task': task, 'box': box['id'], 'note': 'Allowed if head is not distinguishable; inspect visually.'})
        elif count != 1:
            issues.append({'task': task, 'box': box['id'], 'head_count': count})
    for head in heads:
        parent = boxes.get(head.get('parentID'))
        if parent is None:
            issues.append({'task': task, 'head': head['id'], 'problem': 'missing parent'})
            continue
        b, h = parent['value'], head['value']
        if not (b['x'] <= h['x'] <= b['x'] + b['width'] and b['y'] <= h['y'] <= b['y'] + b['height']):
            issues.append({'task': task, 'head': head['id'], 'problem': 'outside parent box'})
folder = ROOT / 'data/exports' / datetime.now().strftime('%Y%m%d-%H%M%S')
folder.mkdir(parents=True)
export_name='ML_CS.json' if title=='ML_CS' else f'project-{project_id}.json'
(folder / export_name).write_text(json.dumps(snapshot, ensure_ascii=False, indent=2), encoding='utf-8')
report = {'tasks_with_annotations': len(rows), 'classes': dict(classes), 'issues': issues, 'missing_head_points_for_review': missing_heads}
(folder / 'audit.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
print(json.dumps(report, indent=2))
print(f'Snapshot: {folder}')
