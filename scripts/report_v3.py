"""Package v2/v3 comparison on the identical original test, with limitations."""
import json
from pathlib import Path
import shutil
from PIL import Image, ImageDraw

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT.parent/'2026-10-17/new-chat/outputs/third-training'

if __name__=='__main__':
    old=json.loads((ROOT/'runs/pose-v2/test-report.json').read_text())
    new=json.loads((ROOT/'runs/pose-v3/test-report.json').read_text())
    assert old['fixed_thresholds']==new['fixed_thresholds']
    assert [r['image'] for r in old['images']]==[r['image'] for r in new['images']]
    stats=json.loads((ROOT/'data/pose-v3/manifest.json').read_text())['stats']
    OUT.mkdir(parents=True,exist_ok=True)
    lines=['# Третье обучение — 19 октября 2026','',
        'Добавлены 40 кадров новой видеосессии: 52 размеченных персонажа, 10 пустых изображений.',
        'Одна голова неразличима: экспортирована как (0,0,0), без угаданной координаты.',
        'Вся новая видеосессия целиком используется только в train.',
        'Старые train/val/test-метки проверены на совпадение; val/test-изображения и метки сохранены побайтово.',
        'Train 167 кадров (158 T/106 CT), val 24 (15 T/22 CT), test 12 (9 T/16 CT).','',
        'YOLO11n-pose, те же исходные pretrained веса, вход 640, максимум 60 эпох, batch 8, seed 42.',
        'Сравнение использует confidence≥0.25, совпадение класса и IoU≥0.5.','',
        '| Метрика на прежнем test | V2 | V3 |','|---|---:|---:|']
    for label,a,b in [
        ('Правильно найдено и классифицировано из 25',old['matched'],new['matched']),
        ('Всего выдано рамок',old['predictions'],new['predictions']),
        ('Precision',f"{old['precision']:.1%}",f"{new['precision']:.1%}"),
        ('Recall',f"{old['recall']:.1%}",f"{new['recall']:.1%}"),
        ('Box mAP@0.5',f"{old['ultralytics']['metrics/mAP50(B)']:.3f}",f"{new['ultralytics']['metrics/mAP50(B)']:.3f}"),
        ('Box mAP@0.5:0.95',f"{old['ultralytics']['metrics/mAP50-95(B)']:.3f}",f"{new['ultralytics']['metrics/mAP50-95(B)']:.3f}"),
        ('Средняя ошибка головы, пикселей, только сопоставленные объекты',
         f"{old['head_error_on_matched_objects_only']['mean_pixels_at_1440x1080']:.1f}",
         f"{new['head_error_on_matched_objects_only']['mean_pixels_at_1440x1080']:.1f}")]:
        lines.append(f'| {label} | {a} | {b} |')
    lines+=['','## Ограничения','',
        'Test маленький и уже многократно просмотрен при разработке. Это сравнительный benchmark, а не финальная независимая оценка.',
        'Новая видеозапись теперь обучающая: её кадры или соседние кадры нельзя использовать как независимый test.',
        'Потребуется новая игровая сессия. Метрики старого test не доказывают устранения ложных срабатываний на пустом фоне: там нет пустых кадров.',
        'Ошибку головы считаем только по правильно сопоставленным объектам, и набор таких объектов может различаться между моделями.',
        'При одинаковом числе эпох больший train даёт больше шагов оптимизации; это не чистое сравнение при одинаковом вычислительном бюджете.',
        'Наведение, стрельба и end-to-end задержка этим экспериментом не проверялись.','',
        '## Артефакты и воспроизведение','',
        'Веса: runs/pose-v3/weights/best.pt. Параметры: runs/pose-v3/args.yaml.',
        'Исходные кадры и split: data/pose-v3/manifest.json.',
        'Обучение: train-pose.cmd --data configs/pose-v3.yaml --name pose-v3-repeat',
        'Оценка: .venv/Scripts/python.exe scripts/evaluate_pose.py --run pose-v3 --eval-name pose-v3-test',
        'Подробный результат: runs/pose-v3/test-report.json.']
    report='\n'.join(lines)+'\n'
    (ROOT/'docs/third-pose-run.md').write_text(report,encoding='utf-8')
    (OUT/'report.md').write_text(report,encoding='utf-8')
    for src,target in [('weights/best.pt','cs2-head-v3.pt'),('test-report.json','metrics.json'),('results.png','training-curves.png')]:
        shutil.copy2(ROOT/'runs/pose-v3'/src,OUT/target)
    canvas=Image.new('RGB',(1280,4*505+25),'white');d=ImageDraw.Draw(canvas)
    d.text((5,5),'V2',fill='black');d.text((645,5),'V3: hard frames + negative examples',fill='black')
    for row,prefix in enumerate(('178_','180_','184_','189_')):
        for col,run in enumerate(('pose-v2','pose-v3')):
            p=next((ROOT/'runs'/run).glob('test_'+prefix+'*.jpg'))
            im=Image.open(p);im.thumbnail((640,480));x,y=col*640,row*505+25
            canvas.paste(im,(x,y));d.text((x+3,y+482),p.name,fill='black')
    canvas.save(OUT/'comparison.jpg')
    print('Saved comparison:',OUT)
