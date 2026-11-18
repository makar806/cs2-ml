"""Portable helpers for exact-frame offline comparison."""

from contextlib import contextmanager
from hashlib import sha256
import json
from pathlib import Path
from urllib.request import Request, urlopen

import cv2
import numpy as np
import torch
from ultralytics import YOLO


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / 'data'


def load(name):
    return json.loads((DATA / name).read_text(encoding='utf-8'))


def fingerprint(path):
    with path.open('rb') as stream:
        digest = sha256()
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def materialize(row, cache):
    short = row.get('excerpt')
    if short:
        clip = DATA / short['file']
        if clip.is_file() and fingerprint(clip) == short['sha256']:
            return clip, short['first_frame']
    path = cache / f"{row['case']}-original.mp4"
    if path.is_file() and path.stat().st_size == row['bytes'] and fingerprint(path) == row['sha256']:
        return path, 0
    if not row['url'].startswith('https://huggingface.co/datasets/blanchon/opencs2_dataset/resolve/'):
        raise ValueError(f"Unexpected source URL for {row['case']}")
    cache.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix('.part')
    try:
        request = Request(row['url'], headers={'User-Agent': 'early-occlusion-benchmark/1.0'})
        with urlopen(request, timeout=45) as source, temporary.open('wb') as output:
            length = 0
            for chunk in iter(lambda: source.read(1024*1024), b''):
                length += len(chunk)
                if length > row['bytes']:
                    raise IOError(f"Source grew unexpectedly for {row['case']}")
                output.write(chunk)
        if length != row['bytes'] or fingerprint(temporary) != row['sha256']:
            raise IOError(f"Unexpected source bytes for {row['case']}")
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)
    return path, 0


def associated(a, b):
    ax, ay = (a[0]+a[2])/2, (a[1]+a[3])/2
    bx, by = (b[0]+b[2])/2, (b[1]+b[3])/2
    height = max(a[3]-a[1], b[3]-b[1])
    return (ax-bx)**2 + (ay-by)**2 <= max(36., .85*height)**2


@contextmanager
def network(weights, device):
    model = YOLO(str(weights), task='pose').model.eval().to(device)
    model.half() if device.type == 'cuda' else model.float()
    yield model


@torch.inference_mode()
def raw_prediction(model, image):
    # Keep both resize stages of the recorded V18 comparisons.
    if image.shape != (720, 1280, 3):
        raise ValueError(image.shape)
    prepared = cv2.resize(image[:, 160:1120], (1440, 1080), interpolation=cv2.INTER_LINEAR)
    height, width = prepared.shape[:2]
    ratio = 1280/max(height, width)
    rw, rh = round(width*ratio), round(height*ratio)
    padded_w, padded_h = (rw+31)//32*32, (rh+31)//32*32
    left, top = round((padded_w-rw)/2-.1), round((padded_h-rh)/2-.1)
    resized = cv2.resize(prepared, (rw, rh), interpolation=cv2.INTER_LINEAR)
    tensor = np.full((1, 3, padded_h, padded_w), 114/255, dtype=np.float32)
    np.multiply(resized[:, :, ::-1].transpose(2, 0, 1), 1/255,
                out=tensor[0, :, top:top+rh, left:left+rw], casting='unsafe')
    parameter = next(model.parameters())
    result = model(torch.from_numpy(tensor).to(device=parameter.device, dtype=parameter.dtype))[0]
    result = result.float().cpu().numpy()
    if result.shape[1] != 10:
        raise ValueError(f'Expected the V18/V19 pose layout with ten channels. Got {result.shape}')
    return result, (width, height), ratio, (left, top)


def boxes_for(output, original, ratio, pad):
    rows = np.asarray(output)[0].T
    width, height = original
    left, top = pad
    boxes = np.empty((len(rows), 4), dtype=np.float32)
    boxes[:, 0] = np.clip((rows[:, 0]-rows[:, 2]*.5-left)/ratio, 0, width)
    boxes[:, 1] = np.clip((rows[:, 1]-rows[:, 3]*.5-top)/ratio, 0, height)
    boxes[:, 2] = np.clip((rows[:, 0]+rows[:, 2]*.5-left)/ratio, 0, width)
    boxes[:, 3] = np.clip((rows[:, 1]+rows[:, 3]*.5-top)/ratio, 0, height)
    return boxes
