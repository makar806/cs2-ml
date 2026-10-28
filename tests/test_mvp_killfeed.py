import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from mvp_killfeed import VisualKillFeed, calibrate


def scene(rows=(), death=False, other=False):
    image = np.zeros((1080, 1920, 3), np.uint8)
    for i, name in enumerate(rows):
        x, y = 1510, 90 + i * 40
        cv2.rectangle(image, (x, y), (1905, y + 32), (30, 20, 230), 2)
        killer = "OTHER" if other else ("BOT" if death else "SELF")
        victim = "SELF" if death else name
        cv2.putText(
            image,
            killer,
            (x + 8, y + 23),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (210, 150, 60) if not death else (40, 190, 230),
            1,
        )
        cv2.putText(
            image,
            victim,
            (x + 200, y + 23),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (210, 150, 60) if death else (40, 190, 230),
            1,
        )
    return image


def sensor():
    return VisualKillFeed(calibrate(scene(["ZED"])))


def test_new_own_banner_counts_once_and_old_feed_is_baselined():
    s = sensor()
    image = scene(["ZED"])
    s.update(image, 0, False)
    s.update(image, 0.11, True)
    s.update(image, 0.22, True)
    assert s.sequence == 0
    image = scene(["ZED", "MURPHY"])
    s.update(image, 0.33, True)
    assert s.sequence == 0
    s.update(image, 0.44, True)
    assert s.sequence == 1
    for t in [0.55, 0.66, 0.77, 0.88]:
        s.update(image, t, True)
    assert s.sequence == 1


def test_own_death_and_another_killers_row_never_confirm_kill():
    s = sensor()
    assert not s.own_rows(scene(["ZED"], death=True))
    assert not s.own_rows(scene(["ZED"], other=True))


def test_same_victim_can_count_again_and_scrolling_does_not_duplicate():
    s = sensor()
    s.update(scene(), 0, False)
    for t in [0.11, 0.22]:
        s.update(scene(["ZED"]), t, True)
    assert s.sequence == 1
    for t in [0.33, 0.44]:
        s.update(scene(["ZED", "ZED"]), t, True)
    assert s.sequence == 2
    for t in [0.55, 0.66]:
        s.update(scene(["ZED"]), t, True)
    assert s.sequence == 2


def test_one_frame_flash_or_disappearance_does_not_create_duplicate_event():
    s = sensor()
    s.update(scene(), 0, False)
    s.update(scene(["ZED"]), 0.11, True)
    s.update(scene(), 0.22, True)
    assert s.sequence == 0
    for t in [0.33, 0.44]:
        s.update(scene(["ZED"]), t, True)
    assert s.sequence == 1
    s.update(scene(), 0.55, True)
    for t in [0.66, 0.77]:
        s.update(scene(["ZED"]), t, True)
    assert s.sequence == 1


def test_focus_gap_rebaselines_instead_of_replaying_unseen_kills():
    s = sensor()
    s.update(scene(), 0, False)
    s.update(scene(["ZED"]), 2, True)
    s.update(scene(["ZED"]), 2.11, True)
    assert s.sequence == 0
