"""Compare detector-only and amodal preview in a local, read-only experiment."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time
import winsound

import cv2
import numpy as np

from mvp_runtime import load_model


ROOT = Path(__file__).resolve().parents[1]
VENDOR = ROOT / 'work/sameo/kins'
CHECKPOINT_HASH = '6a26266a44ce6c8edf69058a2313bd39fd21de936d9e0e12d5e876aab97b1634'
VERSION = '3.22'
WAIT_SECONDS = 120


def signal(sound, asynchronous=True):
    flags = winsound.SND_ALIAS | winsound.SND_NODEFAULT
    if asynchronous:
        flags |= winsound.SND_ASYNC
    try:
        winsound.PlaySound(sound, flags)
    except RuntimeError:
        pass  # The console still announces the stage if system sounds are muted.


class AmodalPreview:
    def __init__(self):
        import torch

        checkpoint_path = VENDOR / 'checkpoints/best.pt'
        if hashlib.sha256(checkpoint_path.read_bytes()).hexdigest() != CHECKPOINT_HASH:
            raise RuntimeError('Unexpected SAMEO-KINS checkpoint')
        sys.path.insert(0, str(VENDOR))
        from efficient_sam.build_efficient_sam import build_efficient_sam_vitt

        torch.set_num_threads(2)
        checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=True)
        model = build_efficient_sam_vitt(checkpoint=None)
        model.load_state_dict(checkpoint['model_state_dict'], strict=True)
        self.torch = torch
        self.model = model.to('cuda').eval()

    def predict(self, image, box):
        torch = self.torch
        with torch.inference_mode():
            rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image_tensor = torch.from_numpy(rgb).permute(2, 0, 1)[None].to('cuda', torch.float32) / 255
            x1, y1, x2, y2 = box
            points = torch.tensor([[[[x1, y1], [x2, y2]]]], device='cuda', dtype=torch.float32)
            labels = torch.tensor([[[2, 3]]], device='cuda', dtype=torch.int64)
            with torch.autocast(device_type='cuda', dtype=torch.float16):
                masks, _ = self.model(image_tensor, points, labels)
            return (masks[0, 0, 0].sigmoid() > .3).cpu().numpy()


def preview_image(image, detection, mask):
    original = image.copy()
    x1, y1, x2, y2 = map(round, detection.box)
    cv2.rectangle(original, (x1, y1), (x2, y2), (255, 160, 40), 2)
    amodal = original.copy()
    amodal[mask] = (amodal[mask] * .55 + np.array([0, 195, 255]) * .45).astype(np.uint8)
    for panel, label in ((original, 'Detector box'), (amodal, 'SAMEO-KINS estimated mask')):
        cv2.putText(panel, label, (15, 30), cv2.FONT_HERSHEY_SIMPLEX, .7, (255, 255, 255), 2)
    return np.hstack((original, amodal))


def stats(samples):
    if not samples:
        return None
    return {key: float(np.percentile(samples, percentile))
            for key, percentile in [('p50', 50), ('p95', 95), ('max', 100)]}


def run_stage(capture, detector, amodal, out, seconds):
    detector_ms, amodal_ms = [], []
    previews = []
    last_id = None
    start = next_mask = next_detection = time.perf_counter()
    while (now := time.perf_counter()) - start < seconds:
        if capture.error:
            raise RuntimeError(capture.error)
        frame = capture.get()
        if frame is None:
            raise RuntimeError('Lost CS2 focus or capture. Keep the local match in the foreground.')
        if frame[0] == last_id or now < next_detection:
            time.sleep(.003)
            continue
        last_id, _, image, _ = frame
        next_detection = now + 1 / 30
        detections = detector.predict(image)
        detector_ms.append((time.perf_counter() - now) * 1000)
        if amodal and detections and now >= next_mask:
            next_mask = now + .25
            target = max(detections, key=lambda d: d.score)
            before = time.perf_counter()
            mask = amodal.predict(image, target.box)
            amodal_ms.append((time.perf_counter() - before) * 1000)
            # Defer disk writes until the timed stage is over.
            if len(previews) < 6 and now - start >= len(previews) * seconds / 6:
                previews.append(preview_image(image, target, mask))
    elapsed = time.perf_counter() - start
    for index, image in enumerate(previews, 1):
        cv2.imwrite(str(out / f'preview-{index}.jpg'), image)
    return dict(seconds=elapsed, detector_frames=len(detector_ms), detector_hz=len(detector_ms)/elapsed,
                detector_ms=stats(detector_ms), masks=len(amodal_ms), amodal_ms=stats(amodal_ms),
                peak_torch_reserved_mb=amodal.torch.cuda.max_memory_reserved()/1048576 if amodal else None)


def main():
    parser = argparse.ArgumentParser(description='SAMEO-KINS read-only local performance preview')
    parser.add_argument('--image', type=Path, help='Use a still image for a smoke test instead of game capture')
    parser.add_argument('--seconds', type=int, default=15, choices=range(5, 61), metavar='5..60')
    args = parser.parse_args()
    from profile_stutter import panel_running, wait_for_game
    from mvp_capture import create_capture
    from mvp_windows import find_window, foreground, key_down

    if panel_running():
        raise RuntimeError('Close the regular MVP panel before running this experiment.')
    out = ROOT / 'runs/amodal-preview' / time.strftime('%Y%m%d-%H%M%S')
    out.mkdir(parents=True)
    print(f'Amodal preview v{VERSION}. Results: {out}', flush=True)
    detector = None
    capture = None
    summary = dict(version=VERSION, model='SAMEO-KINS third-party reproduction',
                   precision='AMP FP16', mask_limit_hz=4)
    try:
        if args.image:
            detector, _ = load_model('native', 1280)
            image = cv2.imread(str(args.image))
            if image is None:
                raise RuntimeError('Could not read test image')
            detections = detector.predict(image)
            if not detections:
                raise RuntimeError('The detector found no character in the test image')
            target = max(detections, key=lambda d: d.score)
            amodal = AmodalPreview()
            amodal.predict(image, target.box)
            start = time.perf_counter()
            mask = amodal.predict(image, target.box)
            summary.update(mode='still_image', detections=len(detections), mask_pixels=int(mask.sum()),
                           amodal_ms=(time.perf_counter()-start)*1000)
            cv2.imwrite(str(out/'preview-1.jpg'), preview_image(image, target, mask))
        else:
            print(f'Open a local match within {WAIT_SECONDS} seconds. Press F8 in CS2 when ready.', flush=True)
            hwnd = wait_for_game(find_window, lambda window: foreground(window) and key_down(0x77),
                                 timeout=WAIT_SECONDS, sleep=lambda seconds: time.sleep(min(seconds, .05)))
            print('F8 received. Loading the detector.', flush=True)
            detector, _ = load_model('native', 1280)
            capture = create_capture(hwnd, 30, 'dxgi')
            deadline = time.perf_counter() + 10
            while (frame := capture.get()) is None:
                if capture.error or time.perf_counter() >= deadline:
                    raise RuntimeError(capture.error or 'No game frames received')
                time.sleep(.05)
            for _ in range(3):
                detector.predict(frame[2])
            print('Stage 1: detector only.', flush=True)
            signal('SystemAsterisk')
            summary['detector_only'] = run_stage(capture, detector, None, out, args.seconds)
            print('Loading amodal weights and warming up.', flush=True)
            amodal = AmodalPreview()
            image = frame[2]
            h, w = image.shape[:2]
            for _ in range(2):
                amodal.predict(image, (w*.4, h*.3, w*.6, h*.7))
            amodal.torch.cuda.reset_peak_memory_stats()
            print('Stage 2: detector and amodal masks.', flush=True)
            summary['with_amodal'] = run_stage(capture, detector, amodal, out, args.seconds)
    finally:
        if capture is not None:
            capture.close()
        if detector is not None:
            detector.close()
        (out/'summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    print(json.dumps(summary, indent=2), flush=True)
    if not args.image and not summary['with_amodal']['masks']:
        print('No masks measured. Repeat with at least one visible bot.', flush=True)
    if not args.image:
        signal('SystemExclamation', asynchronous=False)


if __name__ == '__main__':
    main()
