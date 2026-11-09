"""Package an evaluated v2 run and a direct comparison with the first run."""
import json
import shutil
from pathlib import Path
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT.parent/'2026-10-17/new-chat/outputs/second-training'

if __name__ == '__main__':
    before = json.loads((ROOT/'runs/pose-baseline/test-report.json').read_text())
    after = json.loads((ROOT/'runs/pose-v2/test-report.json').read_text())
    assert before['test_images'] == after['test_images'] == 12
    assert before['ground_truth_objects'] == after['ground_truth_objects'] == 25
    assert before['fixed_thresholds'] == after['fixed_thresholds']
    assert [r['image'] for r in before['images']] == [r['image'] for r in after['images']]
    OUT.mkdir(parents=True,exist_ok=True)
    lines = ['# Второе обучение CS2 T/CT + голова — 18 октября 2026', '',
        'ML_CS: 163 кадра, 274 персонажа (167 T, 107 CT). Все привязки Head→bbox прошли проверку.',
        'Новые координаты преобразованы с учётом размеров 640×640 и 1440×1080.',
        'Точные дубликаты проверены. Исходные изображения и Label Studio не изменялись.', '',
        'Train: 127 кадров, 143 T/69 CT. Val: 24 кадра, 15 T/22 CT.',
        'Одна исходная веб-сессия целиком оставлена для val; её кадров нет в train.',
        'Test: прежние 12 кадров, 9 T/16 CT. Проверено побайтовое совпадение изображений и разметки с v1.',
        'Обе модели стартовали с одинаковых pretrained YOLO11n-pose весов.', '',
        '| Метрика на одинаковых test-кадрах | Первая модель | Вторая модель |',
        '|---|---:|---:|']
    for label, a, b in [
        ('Правильно найдено с верным классом из 25',before['matched'],after['matched']),
        ('Выдано рамок при confidence 0.25',before['predictions'],after['predictions']),
        ('Precision',f"{before['precision']:.1%}",f"{after['precision']:.1%}"),
        ('Recall',f"{before['recall']:.1%}",f"{after['recall']:.1%}"),
        ('Box mAP@0.5',f"{before['ultralytics']['metrics/mAP50(B)']:.3f}",f"{after['ultralytics']['metrics/mAP50(B)']:.3f}"),
        ('Box mAP@0.5:0.95',f"{before['ultralytics']['metrics/mAP50-95(B)']:.3f}",f"{after['ultralytics']['metrics/mAP50-95(B)']:.3f}"),
        ('Ошибка головы, пикселей, только правильно сопоставленные персонажи',
         f"{before['head_error_on_matched_objects_only']['mean_pixels_at_1440x1080']:.1f}",
         f"{after['head_error_on_matched_objects_only']['mean_pixels_at_1440x1080']:.1f}")]:
        lines.append(f'| {label} | {a} | {b} |')
    lines += ['', 'Сопоставление: совпадает класс, IoU≥0.5, confidence≥0.25.',
        'Ошибка головы не учитывает пропущенных или неверно классифицированных персонажей; набор совпавших объектов у моделей может различаться.',
        'mAP вычисляет Ultralytics по диапазону confidence и усредняет по классам; это другая метрика, чем fixed-threshold precision/recall.', '',
        '## Ограничения', '',
        'Всего 12 test-кадров; они из той же локальной съёмки, что часть train, а не из новой независимой сессии.',
        'Эти кадры уже использованы для сравнения версий, поэтому для окончательного отчёта нужен новый test.',
        'Эксперимент не изолирует только эффект данных: расширена val, а при том же числе эпох стало больше шагов оптимизации.',
        'Нет пустых кадров. Не проверены автономное наведение, стрельба и end-to-end задержка.',
        'Результат — промежуточный CV-прототип.', '',
        '## Воспроизведение', '',
        'Подготовка: scripts/prepare_pose_v2.py (не перезаписывает существующий датасет).',
        'Обучение: train-pose.cmd --data configs/pose-v2.yaml --name pose-v2-repeat',
        'Оценка на старом test: .venv/Scripts/python.exe scripts/evaluate_pose.py --run pose-v2 --eval-name pose-v2-test',
        'Фактические параметры: runs/pose-v2/args.yaml.',
        'Происхождение данных и split: data/pose-v2/manifest.json.',
        'Веса: runs/pose-v2/weights/best.pt.',
        'Параметры окружения закреплены в requirements-lock.txt.']
    report = '\n'.join(lines)+'\n'
    (OUT/'report.md').write_text(report,encoding='utf-8')
    (ROOT/'docs/second-pose-run.md').write_text(report,encoding='utf-8')
    for src,name in [('weights/best.pt','cs2-head-v2.pt'),('test-report.json','metrics.json'),('results.png','training-curves.png')]:
        shutil.copy2(ROOT/'runs/pose-v2'/src,OUT/name)
    canvas = Image.new('RGB',(1280,4*505+25),'white')
    draw = ImageDraw.Draw(canvas)
    draw.text((10,5),'V1: first training',fill='black');draw.text((650,5),'V2: extended dataset',fill='black')
    for row,prefix in enumerate(('178_','180_','184_','189_')):
        for col,run in enumerate(('pose-baseline','pose-v2')):
            p = next((ROOT/'runs'/run).glob('test_'+prefix+'*.jpg'))
            im=Image.open(p); im.thumbnail((640,480));x,y=col*640,row*505+25
            canvas.paste(im,(x,y));draw.text((x+3,y+482),p.name,fill='black')
    canvas.save(OUT/'comparison.jpg')
    print(f'Comparison saved: {OUT}')
