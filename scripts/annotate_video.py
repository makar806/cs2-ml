"""Offline CV-only video annotation, without game input or control."""
import argparse
import collections
import json
from pathlib import Path
import subprocess
import time
import cv2
import imageio_ffmpeg
import numpy as np
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parents[1]

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('source',type=Path)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--weights',type=Path,default=ROOT/'runs/pose-v2/weights/best.pt')
    parser.add_argument('--filename',default='cs2-v2-annotated.mp4')
    parser.add_argument('--model-label',default='V2')
    parser.add_argument('--imgsz',type=int,default=640)
    parser.add_argument('--keep-audio',action='store_true')
    args=parser.parse_args()
    args.output.mkdir(parents=True,exist_ok=True)
    assert Path(args.filename).name==args.filename
    target=args.output/args.filename
    if target.exists():
        raise SystemExit(f'Refusing to overwrite {target}')
    cap=cv2.VideoCapture(str(args.source))
    assert cap.isOpened()
    fps=cap.get(cv2.CAP_PROP_FPS)
    width,height=int(cap.get(3)),int(cap.get(4))
    expected=int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    model=YOLO(str(args.weights))
    command=[imageio_ffmpeg.get_ffmpeg_exe(),'-hide_banner','-loglevel','error',
        '-f','rawvideo','-vcodec','rawvideo','-pix_fmt','bgr24','-s',f'{width}x{height}',
        '-r',str(fps),'-i','-']
    if args.keep_audio:
        command+=['-i',str(args.source),'-map','0:v:0','-map','1:a:0?','-c:a','copy','-shortest']
    else:
        command+=['-an']
    command+=['-c:v','libx264','-preset','fast','-crf','22',
        '-pix_fmt','yuv420p','-movflags','+faststart',str(target)]
    encoder=subprocess.Popen(command,stdin=subprocess.PIPE)
    count=with_detection=near_duplicate=0
    classes=collections.Counter()
    previous=None
    duplicate_run=max_duplicate_run=0
    timings=[]
    samples=[]
    started=time.perf_counter()
    try:
        with (args.output/'predictions.jsonl').open('w',encoding='utf-8') as out:
            while True:
                ok,frame=cap.read()
                if not ok:break
                gray=cv2.resize(cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY),(160,120))
                if previous is not None:
                    difference=float(np.mean(cv2.absdiff(gray,previous)))
                    if difference < .5:
                        near_duplicate+=1;duplicate_run+=1
                        max_duplicate_run=max(max_duplicate_run,duplicate_run)
                    else:duplicate_run=0
                previous=gray
                before=time.perf_counter()
                result=model.predict(frame,device=0,imgsz=args.imgsz,conf=.25,verbose=False)[0]
                elapsed=(time.perf_counter()-before)*1000
                timings.append(elapsed)
                predictions=[]
                for i,box in enumerate(result.boxes):
                    cls=int(box.cls.item());label=result.names[cls]
                    classes[label]+=1
                    predictions.append({'class':label,'confidence':float(box.conf.item()),
                        'bbox_xyxy':box.xyxy[0].cpu().tolist(),
                        'head_xy':result.keypoints.xy[i,0].cpu().tolist(),
                        'head_confidence':float(result.keypoints.conf[i,0].item()) if result.keypoints.conf is not None else None})
                with_detection+=bool(predictions)
                annotated=result.plot(kpt_radius=5,line_width=2)
                cv2.putText(annotated,f'{args.model_label} | frame {count} | {count/fps:.1f}s | conf >= 0.25',
                    (15,height-18),cv2.FONT_HERSHEY_SIMPLEX,.65,(255,255,255),2,cv2.LINE_AA)
                encoder.stdin.write(annotated.tobytes())
                out.write(json.dumps({'frame':count,'time_seconds':count/fps,'detections':predictions})+'\n')
                if count % max(1,int(fps*10)) == 0:
                    cv2.imwrite(str(args.output/f'frame-{count:05d}.jpg'),annotated)
                    samples.append(count)
                count+=1
                if count%200==0:print(f'{count}/{expected} frames processed',flush=True)
    finally:
        cap.release()
        encoder.stdin.close()
        status=encoder.wait()
    assert status==0 and count==expected,(status,count,expected)
    summary={'source':str(args.source),'weights':str(args.weights),'imgsz':args.imgsz,'audio_requested':args.keep_audio,
        'frames_processed':count,'width':width,'height':height,'nominal_fps':fps,'duration_seconds':count/fps,
        'frames_with_predictions':with_detection,'predicted_instances_by_class':dict(classes),
        'adjacent_near_identical_pairs':near_duplicate,'longest_near_identical_run_seconds':max_duplicate_run/fps,
        'offline_processing_seconds':time.perf_counter()-started,
        'median_predict_call_ms_excluding_first_10':float(np.median(timings[10:])),
        'notes':['Predictions are not ground truth; no precision or recall without manual labels.',
                 'Near-identical frames may reflect a static scene as well as a recording stall.',
                 'Output uses average source frame rate; no interpolation or skipped frames. Audio copied only if requested and available.',
                 'Offline call timing is not an end-to-end real-time benchmark.']}
    (args.output/'summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
    print(json.dumps(summary,indent=2))
