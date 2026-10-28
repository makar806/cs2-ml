"""Perspective camera-motion approximation; no image/GPU/Windows dependency."""

import math


def rotate_geometry(box, head, dx, dy, size, sensitivity, fov_4_3):
    width, height = size
    cx, cy = width / 2, height / 2
    focal = height / (2 * math.tan(math.radians(fov_4_3) / 2) * 0.75)
    yaw = math.radians(dx * sensitivity * 0.022)
    pitch = math.radians(dy * sensitivity * 0.022)
    sy, cyaw = math.sin(yaw), math.cos(yaw)
    sp, cp = math.sin(pitch), math.cos(pitch)

    def point(px, py):
        x, y = (px - cx) / focal, (py - cy) / focal
        horizontal = cyaw * x - sy
        depth = sy * x + cyaw
        vertical = cp * y - sp * depth
        depth = sp * y + cp * depth
        if depth <= 0.01:
            return None
        return cx + focal * horizontal / depth, cy + focal * vertical / depth

    corners = [point(x, y) for x in (box[0], box[2]) for y in (box[1], box[3])]
    projected_head = point(*head)
    if projected_head is None or any(p is None for p in corners):
        return None
    projected_box = (
        min(p[0] for p in corners),
        min(p[1] for p in corners),
        max(p[0] for p in corners),
        max(p[1] for p in corners),
    )
    return projected_box, projected_head
