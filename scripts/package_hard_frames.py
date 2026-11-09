"""Package manually reviewed raw frames and resolution comparison artifacts."""
from pathlib import Path
import hashlib
import json
import shutil
import zipfile
from PIL import Image, ImageDraw

ROOT=Path(__file__).resolve().parents[1]
WORK=ROOT/'work/hard-frames'
DEST=ROOT/'data/raw/hard_batch_001'
OUT=ROOT.parent/'2026-10-17/new-chat/outputs/hard-batch-001'
SELECTED=[0,2,4,6,8,10,12,14,16,18,20,22,23,24,29,30,32,34,36,37,38,40,42,44,46,48,50,52,54,56,58,60,62,64,66,68,70,72,74,77]

if __name__=='__main__':
    assert len(SELECTED)==len(set(SELECTED))==40
    assert not DEST.exists(), 'Refusing to overwrite batch'
    (DEST/'images').mkdir(parents=True)
    OUT.mkdir(parents=True,exist_ok=True)
    comparison=json.loads((WORK/'comparison.json').read_text())
    source=Path(r'C:\Users\makar\Videos\Captures\Counter-Strike 2 2026-10-18 02-17-26.mp4')
    manifest=[];hashes=set()
    for i,s in enumerate(SELECTED,1):
        r=comparison[s]
        name=f'hard_{i:03d}_frame_{r["frame"]:05d}.png'
        src=WORK/f'sample-{s:03d}.png'
        digest=hashlib.sha256(src.read_bytes()).hexdigest()
        assert digest not in hashes
        hashes.add(digest)
        with Image.open(src) as im: assert im.size==(1440,1080)
        shutil.copy2(src,DEST/'images'/name)
        manifest.append({'file':name,'source':str(source),'source_session':'local_video_20261018_021726',
            'source_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),
            'frame':r['frame'],'approx_seconds':r['approx_seconds'],
            'source_position_ms':r['source_position_ms'],'sha256':digest,
            'intended_use':'training_only_after_manual_annotation','sample_index':s})
    (DEST/'manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    guide='''# 40 сложных кадров из новой записи

Оригиналы для импорта находятся только в папке images: 40 PNG 1440×1080.
На них нет рамок или точек от модели. Исходная запись не изменена.
Добавьте эти 40 PNG в ML_CS через Import → Upload Files.
После импорта в проекте должно быть 203 кадра (если до импорта было 163).
Рекомендуется сначала разметить первые 5 кадров и проверить привязку Head.

Размечайте всех различимых живых T и CT, включая дальних и частично скрытых.
Head ставьте только если положение головы можно определить; скрытое не угадывайте.
Привязывайте точку к соответствующей рамке в Regions.
Не размечайте аватарки сверху, миникарту, значки, своё оружие, трупы, стены и вывески.
На кадре без живых персонажей сохраните пустую разметку через Submit.
Если голова различима плохо, увеличьте изображение в интерфейсе; недостающие детали не придумывайте.
Сохраняйте исправления через Update.

Вся запись — одна исходная сессия. Эти кадры уже отобраны по ошибкам модели:
после разметки используем их для обучения, а не для доказательства независимого качества.
Прежние 12 test-кадров остаются прежними; для новой независимой оценки нужна другая запись.
Соседние кадры этой записи не должны случайно попасть в train и test одновременно.
В manifest.json сохранены исходная запись, номера кадров и контрольные суммы.
'''
    (DEST/'START_HERE.md').write_text(guide,encoding='utf-8')
    (OUT/'START_HERE.md').write_text(guide,encoding='utf-8')
    with zipfile.ZipFile(OUT/'hard-40-images.zip','w',compression=zipfile.ZIP_STORED) as z:
        for p in sorted((DEST/'images').glob('*.png')):z.write(p,'images/'+p.name)
        z.write(DEST/'manifest.json','manifest.json');z.write(DEST/'START_HERE.md','START_HERE.md')
    for page in range(2):
        canvas=Image.new('RGB',(1280,5*265),'white');d=ImageDraw.Draw(canvas)
        for i,r in enumerate(manifest[page*20:(page+1)*20]):
            im=Image.open(DEST/'images'/r['file']);im.thumbnail((320,240))
            x,y=i%4*320,i//4*265;canvas.paste(im,(x,y));d.text((x+3,y+243),r['file'],fill='black')
        canvas.save(OUT/f'contact-{page+1}.jpg')
    examples=[0,34,44,60]
    canvas=Image.new('RGB',(1440,len(examples)*385+25),'white');d=ImageDraw.Draw(canvas)
    for col,size in enumerate((640,960,1280)):d.text((col*480+5,5),f'imgsz={size}',fill='black')
    for row,s in enumerate(examples):
        for col,size in enumerate((640,960,1280)):
            im=Image.open(WORK/f'pred-{s:03d}-{size}.jpg');im.thumbnail((480,360))
            x,y=col*480,row*385+25;canvas.paste(im,(x,y));d.text((x+3,y+362),f'frame {comparison[s]["frame"]}',fill='black')
    canvas.save(OUT/'resolution-comparison.jpg')
    shutil.copy2(WORK/'summary.json',OUT/'resolution-summary.json')
    report='''# Проверка разрешений и подготовка следующей разметки

Одна и та же модель pose-v2, confidence=0.25, 82 одинаковых кадра из видео.
Около одного кадра на секунду, номеров кадров одинаковые для всех размеров.
Прогрев по 3 вызова на размер. GPU синхронизирован вокруг predict().

| Размер входа | Всего предсказанных рамок | Медиана вызова predict |
|---|---:|---:|
| 640 | 73 | 18.3 мс |
| 960 | 155 | 25.2 мс |
| 1280 | 256 | 40.5 мс |

Количество рамок не является количеством правильных обнаружений.
На 1280 модель выделяет аватарки верхней панели как T; на примере ~34 с
также выделяется фонарь как CT. На 640 аватарки в просмотренном кадре не выделяются.
Обнаружение центрального CT на примере ~40 с есть и на 640, и на 1280.
Без ручной разметки и подсчёта precision/recall утверждать, что 1280 лучше, нельзя.
Времена — наблюдение конкретного прогретого offline-вызова с постобработкой,
а не end-to-end latency или гарантия игрового FPS; системная нагрузка не контролировалась.

Отобраны 40 полных кадров без предсказаний модели: дальние цели, перекрытия,
неоднозначные сцены и кадры с фоном для отрицательных примеров.
Проверены размеры 1440×1080, отсутствие точных дубликатов и контрольные суммы.
Исходная запись и проекты Label Studio не изменены. Новое обучение не запускалось.

Следующий шаг: ручная разметка 40 PNG, проверка качества и обучение с отрицательными
примерами. Отдельно можно исследовать маскирование HUD, но в этом эксперименте оно
не применялось, чтобы сравнивался только размер входа.
'''
    (OUT/'report.md').write_text(report,encoding='utf-8')
    (ROOT/'docs/resolution-and-hard-frames.md').write_text(report,encoding='utf-8')
    print('Prepared 40 raw frames:',DEST/'images')
    print('Deliverables:',OUT)
