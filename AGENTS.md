# Project instructions

Communicate with the user in Russian. This is a university ML project.
Use .venv for training, .venv-labeling for annotation. Do not change GPU drivers.
Experiments involving aiming/firing belong in a simulator or controlled local bot environment.
Class mapping: 0=T, 1=CT. Verify exported label mappings before training.
Split data by recording/session; never randomly split adjacent frames across train/test.
Do not claim pretrained COCO weights recognize T/CT, or invent training metrics.
Prefer simple components and document significant changes.

После каждого нового выпуска с улучшениями увеличивать последний компонент версии на 1 (3.10 → 3.11 → 3.12). Согласованно обновлять заголовок панели, надписи интерфейса и версии в status/settings.

SAMEO отменён пользователем: не включать амодальное восстановление в живое управление. Для GUI v3.18 пользователь выбрал фиксированный интервал выстрелов 0.25 с, динамическую зону и отключённую по умолчанию программную паузу по RAM. GSI не подключать.

В режиме зоны не освобождать удерживаемую живую цель по таймеру/наличию другой цели. Плашка, замеченная после acquired_at, не должна теряться из-за более позднего повторного выстрела; дедупликация sequence остаётся обязательной.
Уточнение пользователя для v3.20: после отвода камеры от цели разрешать выбор новых противников без убийства прежнего. Освобождать по оценённому движению фона, выводящему цель за кадр, или подтверждённой резкой смене сцены. Не подменять это таймером пропуска детекции. GSI и дополнительные GPU-модели для этого не подключать.
