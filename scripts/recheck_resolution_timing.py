"""Interleaved warm timing blocks, recording actual model input tensors."""
import json
from pathlib import Path
import random
import time
import cv2
import numpy as np
import torch
from ultralytics import YOLO

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT.parent/'2026-10-17/new-chat/outputs/resolution-comparison'
if __name__=='__main__':
    frames=[cv2.imread(str(p)) for p in sorted((ROOT/'data/pose-v5/images/newcheck').glob('*'))[::4]]
    models={v:YOLO(str(ROOT/f'runs/pose-{v}/weights/best.pt')) for v in ('v4','v5')}
    samples={};blocks=[];shapes={}
    for repeat in range(3):
        configs=[(v,s) for v in models for s in (640,960,1280)]
        random.Random(123+repeat).shuffle(configs)
        for version,size in configs:
            model=models[version];key=f'{version}-{size}'
            params={'imgsz':size,'device':0,'conf':.25,'verbose':False}
            for _ in range(6):model.predict(frames[0],**params)
            def capture(module,inputs):shapes[key]=list(inputs[0].shape)
            hook=model.predictor.model.model.register_forward_pre_hook(capture)
            values=[];inference=[]
            for frame in frames:
                torch.cuda.synchronize();start=time.perf_counter();result=model.predict(frame,**params)[0];torch.cuda.synchronize()
                values.append((time.perf_counter()-start)*1000);inference.append(result.speed['inference'])
            hook.remove()
            samples.setdefault(key,[]).extend(values)
            blocks.append({'repeat':repeat,'key':key,'median_ms':float(np.median(values)),'median_network_inference_ms':float(np.median(inference)),'actual_input_tensor':shapes[key]})
    result={'note':'Three randomized configuration-block passes; six warm predictions before each block, 24 same predecoded 720p images per block; FP32 batch1. Offline timing without game load. Shape hook overhead included equally. This is not a live-game FPS or capture benchmark.','summary':{key:{'median_predict_ms':float(np.median(values)),'p95_predict_ms':float(np.percentile(values,95)),'actual_tensor':shapes[key],'samples':len(values)} for key,values in samples.items()},'blocks':blocks}
    (OUT/'timing-recheck.json').write_text(json.dumps(result,indent=2))
    print(json.dumps(result['summary'],indent=2))
