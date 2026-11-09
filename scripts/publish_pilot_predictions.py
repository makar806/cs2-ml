"""Publish pilot predictions through the local API, preserving human annotations."""
import json
import os
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
os.environ['LABEL_STUDIO_BASE_DATA_DIR']=str(ROOT/'.label-studio')
os.environ['DJANGO_SETTINGS_MODULE']='core.settings.label_studio'
os.environ['LABEL_STUDIO_DISABLE_ANALYTICS']='true'
os.environ['LATEST_VERSION_CHECK']='false'
import label_studio
sys.path.insert(0,str(Path(label_studio.__file__).parent))
import django
django.setup()
from django.contrib.auth import get_user_model
from rest_framework_simplejwt.tokens import AccessToken
import requests

OUT=ROOT.parent/'2026-10-17/new-chat/outputs/prelabel-pilot'
if __name__=='__main__':
    if '--all' in sys.argv:
        OUT=ROOT.parent/'2026-10-17/new-chat/outputs/prelabel-all'
    user=get_user_model().objects.filter(is_active=True).first()
    session=requests.Session()
    session.headers['Authorization']='Bearer '+str(AccessToken.for_user(user))
    response=session.get('http://127.0.0.1:8080/api/projects/3',timeout=20)
    assert response.status_code==200, f'Authentication failed: {response.status_code}'
    payload=json.loads((OUT/'predictions.json').read_text())
    published=[]
    for record in payload:
        task=session.get(f'http://127.0.0.1:8080/api/tasks/{record["task"]}/',timeout=20)
        task.raise_for_status(); task=task.json()
        assert task['project']==3
        if task.get('annotations'):
            print('Skipped human annotation:',record['task'],flush=True);continue
        if task.get('predictions'):
            continue
        existing=[p for p in task.get('predictions',[]) if p.get('model_version')==record['model_version']]
        if existing:continue
        response=session.post('http://127.0.0.1:8080/api/predictions/',json=record,timeout=20)
        assert response.status_code in (200,201),f'Prediction publish failed: {response.status_code}: {response.text[:150]}'
        published.append({'task':record['task'],'prediction_id':response.json()['id']})
        if len(published)%100==0: print('Published',len(published),flush=True)
    (OUT/'published.json').write_text(json.dumps(published,indent=2))
    print('PUBLISHED',len(published),flush=True)
