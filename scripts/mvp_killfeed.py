"""Small visual kill-feed sensor for the CT local-bot demonstration.

Recognizes a calibrated self-name at the LEFT of a red highlighted feed row.
No OCR, GPU model, game data, extra desktop capture, or mouse input.
It confirms a new own-kill notification, not a victim's world identity.
"""

from collections import Counter

import cv2
import numpy as np


def blue_ink(pixels):
    b, g, r = cv2.split(pixels.astype(np.int16))
    return ((b > r + 25) & (b > g + 12) & (g > r + 10) & (b > 65)).astype(np.uint8)


def yellow_ink(pixels):
    b, g, r = cv2.split(pixels.astype(np.int16))
    return ((r > b + 35) & (g > b + 25) & (r > 85) & (g > 70)).astype(np.uint8)


def tight(mask):
    y, x = np.where(mask)
    if len(x) < 12:
        return None
    return mask[y.min() : y.max() + 1, x.min() : x.max() + 1], (int(x.min()), int(y.min()))


def signature(mask):
    return cv2.resize(mask, (112, 16), interpolation=cv2.INTER_NEAREST).astype(bool)


def similarity(a, b):
    denominator = int(a.sum()) + int(b.sum())
    return 2 * int(np.logical_and(a, b).sum()) / denominator if denominator else 0.0


def name_similarity(a, b):
    return max(
        similarity(np.roll(np.roll(a, dy, axis=0), dx, axis=1), b)
        for dy in range(-2, 3)
        for dx in range(-2, 3)
    )


def highlighted_rows(image):
    h, w = image.shape[:2]
    scale = h / 1080
    ox, oy = round(w * 0.50), round(h * 0.04)
    roi = image[oy : round(h * 0.38), ox:]
    b, g, r = cv2.split(roi.astype(np.int16))
    red = ((r > 90) & (r > g * 1.7) & (r > b * 1.2)).astype(np.uint8)
    red = cv2.morphologyEx(red, cv2.MORPH_CLOSE, np.ones((1, 3), np.uint8))
    horizontal = cv2.morphologyEx(
        red, cv2.MORPH_OPEN, np.ones((1, max(20, round(60 * scale))), np.uint8)
    )
    contours, _ = cv2.findContours(horizontal, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    lines = []
    for contour in contours:
        x, y, cw, ch = cv2.boundingRect(contour)
        if 90 * scale <= cw <= w * 0.5 and ch <= 5 * scale:
            lines.append((x, y, cw, ch))
    lines.sort(key=lambda line: line[1])
    used = set()
    boxes = []
    for i, (x, y, cw, ch) in enumerate(lines):
        if i in used:
            continue
        for j in range(i + 1, len(lines)):
            if j in used:
                continue
            bx, by, bw, bh = lines[j]
            if by - y > 55 * scale:
                break
            if (
                14 * scale <= by - y <= 55 * scale
                and abs(bx - x) <= 5 * scale
                and abs(bx + bw - x - cw) <= 5 * scale
            ):
                boxes.append((min(x, bx) + ox, y + oy, max(x + cw, bx + bw) + ox, by + bh + oy))
                used.update((i, j))
                break
    return sorted(boxes, key=lambda box: box[1])


class VisualKillFeed:
    def __init__(self, name_template, interval=0.10):
        if name_template is None or name_template.ndim != 2:
            raise ValueError("Calibrate a CT self-name template first")
        self.name = signature((name_template > 0).astype(np.uint8))
        self.name_aspect = name_template.shape[1] / name_template.shape[0]
        self.interval = interval
        self.last_at = None
        self.geometry = None
        self.sequence = 0
        self.prototypes = []
        self.stable = None
        self.candidate = None
        self.repeats = 0
        self.rows = 0
        self.last_event = None

    @classmethod
    def load(cls, path):
        image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise FileNotFoundError(path)
        return cls(image)

    def own_rows(self, image):
        result = []
        scale = image.shape[0] / 1080
        for box in highlighted_rows(image):
            x, y, x2, y2 = box
            row = image[y + 2 : y2 - 2, x + 3 : x2 - 3]
            found = tight(blue_ink(row))
            if found is None:
                continue
            name, (nx, ny) = found
            aspect = name.shape[1] / name.shape[0]

            if nx > 20 * scale or not 0.7 <= aspect / self.name_aspect <= 1.4:
                continue
            score = name_similarity(signature(name), self.name)
            if score < 0.82:
                continue
            victim = tight(yellow_ink(row[:, nx + name.shape[1] :]))
            if victim is None:
                continue
            victim_mask, _ = victim
            result.append(dict(box=box, name_score=score, victim=signature(victim_mask)))
        return result

    def _identity(self, ink):
        scores = [similarity(ink, old) for old in self.prototypes]
        if scores and max(scores) >= 0.78:
            return int(np.argmax(scores))
        if len(self.prototypes) >= 128:
            self.prototypes = []
            self.stable = None
        self.prototypes.append(ink.copy())
        return len(self.prototypes) - 1

    def update(self, image, at, enabled):
        if self.last_at is not None and at - self.last_at < self.interval:
            return self.snapshot()
        gap = at - self.last_at if self.last_at is not None else None
        self.last_at = at
        rows = self.own_rows(image)
        self.rows = len(rows)
        counts = Counter(self._identity(row["victim"]) for row in rows)
        geometry = image.shape[:2]
        if (
            not enabled
            or self.stable is None
            or geometry != self.geometry
            or gap is None
            or gap > 1
        ):
            self.stable = counts
            self.candidate = None
            self.repeats = 0
        else:
            if counts == self.candidate:
                self.repeats += 1
            else:
                self.candidate = counts
                self.repeats = 1
            if self.repeats >= 2:
                added = sum((counts - self.stable).values())
                if added:
                    self.sequence += added
                    self.last_event = at
                self.stable = counts
        self.geometry = geometry
        return self.snapshot()

    def snapshot(self):
        return dict(
            sequence=self.sequence,
            last_event=self.last_event,
            rows=self.rows,
            sampled_at=self.last_at,
        )


def calibrate(image, row_index=0):
    boxes = highlighted_rows(image)
    if not 0 <= row_index < len(boxes):
        raise ValueError("No highlighted row at requested index")
    x, y, x2, y2 = boxes[row_index]
    row = image[y + 2 : y2 - 2, x + 3 : x2 - 3]
    found = tight(blue_ink(row))
    if found is None or found[1][0] > 20 * image.shape[0] / 1080:
        raise ValueError("CT killer name must be on the left")
    return found[0] * 255
