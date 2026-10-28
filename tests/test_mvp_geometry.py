import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from mvp_core import Controller, Detection
from mvp_geometry import rotate_geometry


def test_45_degree_yaw_centers_ray_and_changes_body_scale():
    box, head = rotate_geometry(
        (1110, 350, 1130, 450), (1120, 360), 45 / 0.022, 0, (1280, 720), 1, 90
    )
    assert abs(head[0] - 640) < 1e-8 and abs(head[1] - 360) < 1e-8
    assert 65 < box[3] - box[1] < 80


def test_yaw_rotation_and_its_inverse_restore_head():
    original = ((900, 200, 1000, 500), (940, 220))
    for counts in [-1500, 1500]:
        geometry = rotate_geometry(*original, counts, 0, (1920, 1080), 0.97, 90)
        restored = rotate_geometry(*geometry, -counts, 0, (1920, 1080), 0.97, 90)
        assert math.dist(original[1], restored[1]) < 1e-8


def test_large_edge_flick_retains_identity_after_rendered_perspective_change():
    c = Controller()
    old = Detection(0, 0.99, (1150, 220, 1250, 550), (1200, 240), 0.99)
    a = c.step([old], 1280, 720, 1, 1, enabled=True, foreground=True, fire=False)
    c.acknowledge(a["dx"], a["dy"], 1280, 720)
    box, head = rotate_geometry(old.box, old.head, a["dx"], a["dy"], (1280, 720), 1, 90)
    observed = Detection(0, 0.99, box, head, 0.99)
    assert (box[3] - box[1]) / (old.box[3] - old.box[1]) < 0.75
    b = c.step([observed], 1280, 720, 1.05, 1.05, enabled=True, foreground=True, fire=False)
    assert b["target"] is not None and b["tracking"]
    assert math.hypot(*b["error"]) < 1
