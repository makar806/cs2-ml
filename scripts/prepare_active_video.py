"""Download and retain only active excerpts from CC-BY OpenCS2 rounds."""
import hashlib
import json
from pathlib import Path
import subprocess
from urllib.parse import quote
import requests
import imageio_ffmpeg
import cv2
from PIL import Image,ImageDraw

ROOT=Path(__file__).resolve().parents[1]
WORK=ROOT/'work/active-video'
OUT=ROOT.parent/'2026-10-17/new-chat/outputs/active-cs2-video'

if __name__=='__main__':
    WORK.mkdir(parents=True,exist_ok=True);OUT.mkdir(parents=True,exist_ok=True)
    target=OUT/'cs2-active-scenes.mp4'
    assert not target.exists()
    rows=json.loads((ROOT/'work/active-selected.json').read_text())
    ffmpeg=imageio_ffmpeg.get_ffmpeg_exe()
    clips=[];manifest=[]
    canvas=Image.new('RGB',(1280,3*385),'white');draw=ImageDraw.Draw(canvas)
    total_seconds=0
    for i,row in enumerate(rows):
        filename=row['video']['path'].split('@main/')[1]
        url='https://huggingface.co/datasets/blanchon/opencs2_dataset/resolve/main/'+quote(filename,safe='/=')
        original=WORK/f'original-{i}.mp4';clip=WORK/f'clip-{i}.mp4'
        with requests.get(url,stream=True,timeout=60) as response:
            response.raise_for_status()
            with original.open('wb') as f:
                for chunk in response.iter_content(1024*1024):f.write(chunk)
        assert original.stat().st_size==row['media_bytes']
        digest=hashlib.sha256(original.read_bytes()).hexdigest()
        start=max(0,row['first_kill']-6);end=min(row['duration_s'],row['last_kill']+3)
        duration=end-start
        subprocess.run([ffmpeg,'-hide_banner','-loglevel','error','-ss',str(start),'-i',str(original),
            '-t',str(duration),'-c:v','libx264','-preset','fast','-crf','18','-pix_fmt','yuv420p',
            '-r','32','-c:a','aac','-b:a','96k','-movflags','+faststart',str(clip)],check=True)
        cap=cv2.VideoCapture(str(original));cap.set(cv2.CAP_PROP_POS_MSEC,(row['first_kill']-.2)*1000)
        ok,frame=cap.read();assert ok;cap.release()
        im=Image.fromarray(cv2.cvtColor(frame,cv2.COLOR_BGR2RGB));im.thumbnail((640,360))
        x,y=i%2*640,i//2*385;canvas.paste(im,(x,y));draw.text((x+3,y+362),f'Scene {i+1}: {row["player_side"]}, 4 kills, {duration:.1f}s',fill='black')
        clips.append(clip)
        manifest.append({'scene':i+1,'media_id':row['media_id'],'source_url':url,'source_sha256':digest,
            'source_start_seconds':start,'source_end_seconds':end,'output_start_seconds':total_seconds,
            'duration_seconds':duration,'player_side':row['player_side'],'kills_in_source_round':row['kills']})
        total_seconds+=duration
        assert original.resolve().parent==WORK.resolve()
        original.unlink()
        print(f'Prepared scene {i+1}/6 ({duration:.1f}s); removed full source round',flush=True)
    listing=WORK/'concat.txt'
    listing.write_text(''.join(f"file '{p.as_posix()}'\n" for p in clips),encoding='utf-8')
    subprocess.run([ffmpeg,'-hide_banner','-loglevel','error','-f','concat','-safe','0','-i',str(listing),
        '-c','copy','-movflags','+faststart',str(target)],check=True)
    cap=cv2.VideoCapture(str(target));fps=cap.get(cv2.CAP_PROP_FPS);expected=int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    count=0
    while True:
        ok,frame=cap.read()
        if not ok:break
        assert frame.shape[:2]==(720,1280)
        count+=1
    cap.release();assert count==expected and fps==32
    for clip in clips:
        assert clip.resolve().parent==WORK.resolve()
        clip.unlink()
    canvas.save(OUT/'preview.jpg')
    (OUT/'sources.json').write_text(json.dumps({'dataset':'https://huggingface.co/datasets/blanchon/opencs2_dataset',
        'license_declared_by_source':'CC-BY-4.0','author':'blanchon','created':'2026-10-19',
        'modifications':'Six excerpts trimmed, encoded and concatenated; no AI detections added.',
        'frames':count,'fps':fps,'duration_seconds':count/fps,'sha256':hashlib.sha256(target.read_bytes()).hexdigest(),
        'scenes':manifest},indent=2),encoding='utf-8')
    print('Verified output:',count/fps,'seconds;',round(target.stat().st_size/1024**2,1),'MiB')
