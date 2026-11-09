"""Inference-only isolation must never create screen capture or send inputs."""
import json
import sys
import threading
from pathlib import Path
from queue import Queue
from types import SimpleNamespace
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
import profile_stutter as profiler
import mvp_runtime


def test_replay_uses_saved_image_and_closes_model(monkeypatch,tmp_path):
    source=tmp_path/'data/pose-v7/images/val';source.mkdir(parents=True)
    (source/'example.jpg').touch();out=tmp_path/'out';out.mkdir()
    import cv2
    monkeypatch.setattr(profiler,'ROOT',tmp_path)
    monkeypatch.setattr(cv2,'imread',lambda _:np.zeros((10,10,3),np.uint8))
    monkeypatch.setattr(cv2,'resize',lambda image,size:image)
    stop=threading.Event();calls=[];closed=[]
    def predict(image):
        calls.append(image.shape)
        if len(calls)>=4:stop.set()
        return []
    monkeypatch.setattr(mvp_runtime,'load_model',lambda *a:(SimpleNamespace(predict=predict,close=lambda:closed.append(True)),'fake'))
    class LocalQueue(Queue):
        def cancel_join_thread(self):pass
    messages=LocalQueue();packets=LocalQueue()
    profiler.replay_worker(dict(output=str(out),hz=30),stop,packets,messages)
    assert messages.get_nowait()['ready'] and len(calls)==4 and closed==[True]
    row=json.loads((out/'api.jsonl').read_text())
    assert row['api']=='inference_replay' and row['cpu_ms']>=0
    assert not (out/'error.txt').exists()
