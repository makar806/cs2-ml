"""Repair two unambiguous structural annotation defects through API, with backups."""
import copy
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
from prepare_pose import convert

OUT=ROOT.parent/'2026-10-17/new-chat/outputs/sixth-experiment'
if __name__=='__main__':
    session=requests.Session();session.headers['Authorization']='Bearer '+str(AccessToken.for_user(get_user_model().objects.filter(is_active=True).first()))
    history=[]
    for tid in (575,687):
        response=session.get(f'http://127.0.0.1:8080/api/tasks/{tid}/',timeout=20);response.raise_for_status()
        task=response.json();assert task['project']==3
        annotations=[a for a in task['annotations'] if not a.get('was_cancelled')]
        assert len(annotations)==1
        annotation=annotations[0];results=copy.deepcopy(annotation['result'])
        heads=[r for r in results if r['type']=='keypointlabels']
        assert len(heads)==1
        # A Head was accidentally also tagged as a rectangle without height.
        invalid=[r for r in results if r['type']=='rectanglelabels' and 'height' not in r['value']]
        assert len(invalid)<=1
        results=[r for r in results if r not in invalid]
        boxes=[r for r in results if r['type']=='rectanglelabels']
        assert len(boxes)==1 and boxes[0]['value']['rectanglelabels']==['CT']
        box=boxes[0];point=heads[0];b=box['value'];p=point['value']
        assert b['x']<=p['x']<=b['x']+b['width'] and b['y']<=p['y']<=b['y']+b['height']
        box.pop('parentID',None);point['parentID']=box['id']
        convert(results)
        backup=OUT/f'annotation-{tid}-before.json'
        assert not backup.exists(),'Refusing to overwrite annotation backup'
        backup.write_text(json.dumps(annotation,indent=2),encoding='utf-8')
        response=session.patch(f'http://127.0.0.1:8080/api/annotations/{annotation["id"]}/',json={'result':results},timeout=20);response.raise_for_status()
        returned=response.json()['result'];convert(returned)
        history.append({'task':tid,'annotation_id':annotation['id'],'reason':'Removed malformed point-as-box duplicate where present; attached existing Head to the sole containing CT rectangle; cleared inverted parent relationship. Coordinates and CT class unchanged.','after':returned})
        print('REPAIRED',tid,flush=True)
    (OUT/'annotation-repairs.json').write_text(json.dumps(history,indent=2),encoding='utf-8')
