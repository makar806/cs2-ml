"""Package hard training frames and match-separated fixed evaluation frames."""
from pathlib import Path
import hashlib
import json
import shutil
import zipfile
import cv2
from PIL import Image,ImageDraw

ROOT=Path(__file__).resolve().parents[1]
WORK=ROOT/'work/active-label-candidates'
DEST=ROOT/'data/raw/active_hard_batch_001'
OUT=ROOT.parent/'2026-10-17/new-chat/outputs/active-hard-labels'
SELECT={1:[0,2,4,6,8,10,12],4:[0,2,5,6,8,10,13],5:[0,2,4,6,9,11,13],6:[0,2,4,6,8,10,13]}

if __name__=='__main__':
    assert not DEST.exists(), 'Refusing to overwrite'
    (DEST/'images').mkdir(parents=True)
    OUT.mkdir(parents=True,exist_ok=True)
    source_folder=ROOT.parent/'2026-10-17/new-chat/outputs/active-cs2-video'
    video=source_folder/'cs2-active-scenes.mp4'
    source=json.loads((source_folder/'sources.json').read_text())
    candidates=json.loads((WORK/'candidates.json').read_text())
    scenes={s['scene']:s for s in source['scenes']}
    records=[];hashes=set()
    def add(path,split,scene,frame,seconds):
        digest=hashlib.sha256(path.read_bytes()).hexdigest()
        assert digest not in hashes;hashes.add(digest)
        s=scenes[scene]
        name=f'active_{split}_scene{scene}_frame{frame:05d}.png'
        shutil.copy2(path,DEST/'images'/name)
        with Image.open(DEST/'images'/name) as im: assert im.size==(1280,720)
        records.append({'file':name,'split':split,'scene':scene,'frame':frame,'montage_seconds':seconds,
            'match_id':int(s['media_id'].split('-')[0]),'media_id':s['media_id'],'source_url':s['source_url'],
            'sha256':digest,'source_video_sha256':source['sha256']})
    for scene,indices in SELECT.items():
        for idx in indices:
            r=next(r for r in candidates if r['scene']==scene and r['candidate']==idx)
            add(WORK/r['filename'],'train',scene,r['frame'],r['output_seconds'])
    cap=cv2.VideoCapture(str(video))
    for scene,offsets in {2:[3,6,10,15,20,24],3:[2,5,9,13,17,21]}.items():
        for idx,off in enumerate(offsets):
            sec=scenes[scene]['output_start_seconds']+off;frame=round(sec*32)
            cap.set(cv2.CAP_PROP_POS_FRAMES,frame);ok,image=cap.read();assert ok
            p=WORK/f'fixed-check-scene{scene}-{idx}.png';cv2.imwrite(str(p),image)
            add(p,'check',scene,frame,sec)
    cap.release()
    assert len(records)==40
    assert not {r['match_id'] for r in records if r['split']=='train'} & {r['match_id'] for r in records if r['split']=='check'}
    metadata={'split_policy':'28 manually curated training images from 4 matches; 12 fixed-time check images from 2 other matches. Never move check to train while reporting check metrics.',
        'limitation':'Prior third-party still-image data did not include match IDs; independence from that historical source is not fully proven.',
        'new_training_performed':False,'images':records}
    (DEST/'manifest.json').write_text(json.dumps(metadata,indent=2),encoding='utf-8')
    guide='''# 40 кадров: тени, дальние цели, перекрытия и дым

Загрузите все 40 PNG из папки images в ML_CS. После импорта ожидается 243 кадра,
если ранее их было 203. Имена active_train_* и active_check_* не меняйте.
Явное разделение сохранено в manifest.json: 28 train и 12 check из других матчей.
check не будет использован в обучении; размечайте его по тем же правилам.
На кадрах нет предсказаний модели. Размер 1280×720, исходное соотношение сторон.

Размечайте всех различимых живых T и CT, в том числе маленьких и частично скрытых.
Рамка плотно охватывает видимую часть тела. Голова — одна точка Head,
привязанная к своему персонажу в Regions. Не угадывайте невидимую голову.
Увеличивайте изображение в Label Studio, чтобы размечать маленькие цели.
Если тело полностью скрыто непрозрачным дымом/стеной — не рисуйте его по памяти.
Если в кадре нет различимых живых персонажей, сохраните пустую разметку Submit.
Если персонаж виден, но команду невозможно определить, запишите номер задания
для совместной проверки; не угадывайте класс и не сохраняйте его как пустой фон.
Не размечайте трупы, своё оружие, вспышки, элементы интерфейса и фон.

Кадры в дыму не гарантируют наличие видимого противника. Иногда правильный ответ —
ничего не обнаружить. На основе одного непрозрачного кадра нельзя восстановить
положение скрытого человека.

Источник: blanchon / OpenCS2 POV Renders,
https://huggingface.co/datasets/blanchon/opencs2_dataset
Заявленная лицензия: CC BY 4.0, https://creativecommons.org/licenses/by/4.0/
Изменения: извлечены кадры из ранее подготовленных фрагментов, без новых визуальных наложений.
URL, match ID и номера кадров приведены в manifest.json.
'''
    (DEST/'START_HERE.md').write_text(guide,encoding='utf-8')
    (OUT/'START_HERE.md').write_text(guide,encoding='utf-8')
    (OUT/'manifest.json').write_text(json.dumps(metadata,indent=2),encoding='utf-8')
    with zipfile.ZipFile(OUT/'active-hard-40.zip','w',compression=zipfile.ZIP_STORED) as z:
        for p in (DEST/'images').glob('*.png'):z.write(p,'images/'+p.name)
        z.write(DEST/'START_HERE.md','START_HERE.md');z.write(DEST/'manifest.json','manifest.json')
    for page in range(2):
        canvas=Image.new('RGB',(1280,5*205),'white');d=ImageDraw.Draw(canvas)
        for i,r in enumerate(records[page*20:(page+1)*20]):
            im=Image.open(DEST/'images'/r['file']);im.thumbnail((320,180));x=i%4*320;y=i//4*205
            canvas.paste(im,(x,y));d.text((x+3,y+182),r['file'],fill='black')
        canvas.save(OUT/f'contact-{page+1}.jpg')
    print('Ready: 28 training + 12 check frames')
    print(DEST/'images')
