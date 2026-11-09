"""Create a separate consecutive review queue without modifying original tasks."""
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
OUT=ROOT.parent/'2026-10-17/new-chat/outputs/sixth-experiment/focus-review'

if __name__=='__main__':
    assert not (OUT/'published-project.json').exists(),'Queue already published'
    session=requests.Session();session.headers['Authorization']='Bearer '+str(AccessToken.for_user(get_user_model().objects.filter(is_active=True).first()))
    base='http://127.0.0.1:8080'
    original=session.get(base+'/api/projects/3/',timeout=20);original.raise_for_status()
    selected=json.loads((OUT/'selected.json').read_text())
    response=session.post(base+'/api/projects/',json={'title':'ML_CS_FOCUS_40','description':'40 training-only candidates on seven maps, selected from unsubmitted ML_CS frames. Validate live T/CT, all visible characters and Head; no jumping between original IDs. Predictions are hints, not ground truth. Original task IDs and hashes retained for deduplication.','label_config':original.json()['label_config']},timeout=20)
    response.raise_for_status();pid=response.json()['id']
    # Save project immediately to avoid creating another project after an interrupted import.
    state={'project_id':pid,'url':f'{base}/projects/{pid}/data','source_project_id':3,'imported':False,'rows':selected}
    (OUT/'published-project.json').write_text(json.dumps(state,indent=2))
    tasks=[]
    for row in selected:
        result=session.get(f'{base}/api/tasks/{row["source_task_id"]}/',timeout=20);result.raise_for_status()
        assert not result.json().get('annotations'),'Source task was annotated while queue was prepared'
        tasks.append({'data':{'image':row['source_image_url'],'source_task_id':row['source_task_id'],'source_project_id':3,'source_sha256':row['sha256'],'map_name':row['map'],'source_file':row['file']},'predictions':[{'model_version':'V6-focus-960','score':min(row['rank'],1),'result':row['result']}]})
    response=session.post(f'{base}/api/projects/{pid}/import',json=tasks,timeout=90);response.raise_for_status()
    state['imported']=True;state['import_response']=response.json()
    (OUT/'published-project.json').write_text(json.dumps(state,indent=2))
    print('PROJECT',pid,'IMPORTED',response.json().get('task_count'),flush=True)
