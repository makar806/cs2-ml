"""Export the 83 public annotated frames as additional YOLO pose training samples."""

import argparse
from hashlib import sha256
from pathlib import Path

import cv2

from protocol import ROOT, load, materialize
from replay import read_frames


def pose_label(annotation, side):
    if annotation is None:
        return ''
    x1, y1, x2, y2 = annotation['box']
    if side not in (0, 1) or not (160 <= x1 < x2 <= 1120 and 0 <= y1 < y2 <= 720):
        raise ValueError(f'Invalid box or side for {annotation}')
    head = annotation.get('head')
    if head and not (x1 <= head[0] <= x2 and y1 <= head[1] <= y2):
        raise ValueError(f'Visible head outside the visible box for {annotation}')
    keypoint = ((head[0]-160)/960, head[1]/720, 2) if head else (0, 0, 0)
    values = (((x1+x2)/2-160)/960, (y1+y2)/1440, (x2-x1)/960, (y2-y1)/720, *keypoint)
    return f'{side} ' + ' '.join(f'{value:.8f}' for value in values) + '\n'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('destination', type=Path)
    parser.add_argument('--cache', type=Path, default=ROOT/'.cache')
    args = parser.parse_args()
    sources = {row['case']: row for row in load('sources.json')}
    samples = load('calibration.json')
    episodes = {}
    for row in samples:
        episodes.setdefault(row['case'], {})[row['frame']] = row
    payload = []
    for case, marks in episodes.items():
        path, frame_offset = materialize(sources[case], args.cache)
        for frame, _, image in read_frames(path, marks, frame_offset):
            row = marks[frame]
            ok, compressed = cv2.imencode('.jpg', image[:, 160:1120], [cv2.IMWRITE_JPEG_QUALITY, 98])
            if not ok:
                raise OSError(f'Image encode failed for {case}/{frame}')
            data = compressed.tobytes()
            if sha256(data).hexdigest() != row['sha256']:
                raise ValueError(f'Encoded image bytes differ from the V19 export for {case}/{frame}')
            name = f"early_match{row['match_id']}_c{case}_f{frame:04d}"
            payload.append((name, data, pose_label(row['box_source'], row['expected_class'])))
    args.destination.mkdir(parents=True)
    images, labels = (args.destination/f'{kind}/train' for kind in ('images', 'labels'))
    images.mkdir(parents=True)
    labels.mkdir(parents=True)
    for name, image, label in payload:
        (images/f'{name}.jpg').write_bytes(image)
        (labels/f'{name}.txt').write_text(label, encoding='utf-8')
    print(f'{len(payload)} digest-verified annotated samples exported.')


if __name__ == '__main__':
    main()
