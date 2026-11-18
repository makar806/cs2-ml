"""Measure local appearance changes after registering the preceding frame."""

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass(frozen=True)
class Change:
    residual: np.ndarray
    changed: np.ndarray
    coverage: np.ndarray
    spatial_error: float
    inliers: int


def prepare(image):
    return cv2.GaussianBlur(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY), (3, 3), .55)


def register(previous, current):
    prev_small = cv2.resize(previous, None, fx=.5, fy=.5, interpolation=cv2.INTER_AREA)
    curr_small = cv2.resize(current, None, fx=.5, fy=.5, interpolation=cv2.INTER_AREA)
    height, width = prev_small.shape
    # Avoid the first-person view model and obtain anchors across the scene.
    mask = np.zeros_like(prev_small)
    mask[11:round(.69*height), 8:width-8] = 255
    points = cv2.goodFeaturesToTrack(prev_small, 150, .01, 8, mask=mask)
    if points is None or len(points) < 12:
        return None
    opts = dict(winSize=(15, 15), maxLevel=2)
    found, status, _ = cv2.calcOpticalFlowPyrLK(prev_small, curr_small, points, None, **opts)
    if found is None:
        return None
    back, valid, _ = cv2.calcOpticalFlowPyrLK(curr_small, prev_small, found, None, **opts)
    if back is None:
        return None
    error = np.linalg.norm(back[:, 0]-points[:, 0], axis=1)
    good = status[:, 0].astype(bool) & valid[:, 0].astype(bool) & (error < .65)
    src, dst = points[:, 0][good], found[:, 0][good]
    if len(src) < 12:
        return None
    matrix, inliers = cv2.estimateAffinePartial2D(src, dst, ransacReprojThreshold=1.,
                                                 maxIters=600, confidence=.98)
    if matrix is None or inliers is None:
        return None
    active = inliers[:, 0].astype(bool)
    count = int(active.sum())
    if count < 12 or count / len(src) < .55:
        return None
    if np.ptp(src[active, 0]) < width*.3 or np.ptp(src[active, 1]) < height*.23:
        return None
    delta = src[active] @ matrix[:, :2].T + matrix[:, 2] - dst[active]
    spatial_error = float(np.median(np.linalg.norm(delta, axis=1))*2)
    matrix[:, 2] *= 2
    return matrix, spatial_error, count


def changes(previous, current):
    alignment = register(previous, current)
    if alignment is None:
        return None
    matrix, error, inliers = alignment
    height, width = current.shape
    warped = cv2.warpAffine(previous, matrix, (width, height))
    coverage = cv2.warpAffine(np.full((height, width), 1, dtype=np.uint8), matrix, (width, height),
                              flags=cv2.INTER_NEAREST)
    coverage = cv2.erode(coverage, np.ones((3, 3), np.uint8))
    diff = current.astype(np.float32) - warped.astype(np.float32)
    sample = diff[::8, ::8][coverage[::8, ::8].astype(bool)]
    if not sample.size:
        return None
    offset = np.median(sample)
    if abs(offset) > 18 or np.median(np.abs(sample-offset)) > 12:
        return None  # Scene cut, full flash or an alignment failure.
    # Downweight high-gradient edges that are sensitive to sub-pixel registration.
    bounds = np.ones((3, 3), np.uint8)
    gradient = cv2.dilate(warped, bounds).astype(np.float32) - cv2.erode(warped, bounds)
    residual = np.maximum(np.abs(diff-offset) - .22*gradient - 3, 0)
    residual *= coverage
    return Change(cv2.integral(np.minimum(residual, 50), sdepth=cv2.CV_32F),
                  cv2.integral((residual >= 9).astype(np.uint8), sdepth=cv2.CV_32S),
                  cv2.integral(coverage, sdepth=cv2.CV_32S), error, inliers)


def box_features(change, boxes):
    """Return mean change, changed fraction and the same features in the nearby background."""
    if not len(boxes):
        return np.empty((0, 5), dtype=np.float32)
    height, width = np.subtract(change.residual.shape, 1)
    limits = (width, height, width, height)
    rect = np.clip(np.hstack((np.floor(boxes[:, :2]), np.ceil(boxes[:, 2:]))), 0, limits).astype(np.int32)
    extent = np.maximum(rect[:, 2:] - rect[:, :2], 0)
    margin = np.maximum(4, np.round(extent*.3)).astype(np.int32)
    surround = np.clip(rect + np.hstack((-margin, margin)), 0, limits)

    def totals(field, coordinates):
        x1, y1, x2, y2 = coordinates.T
        return field[y2, x2]-field[y1, x2]-field[y2, x1]+field[y1, x1]

    inner = totals(change.coverage, rect)
    outer = np.maximum(totals(change.coverage, surround)-inner, 1)
    area = np.maximum(extent[:, 0]*extent[:, 1], 1)
    amount = totals(change.residual, rect)
    count = totals(change.changed, rect)
    return np.column_stack((amount/np.maximum(inner, 1), count/np.maximum(inner, 1),
                            (totals(change.residual, surround)-amount)/outer,
                            (totals(change.changed, surround)-count)/outer, inner/area))
