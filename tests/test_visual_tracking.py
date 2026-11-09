import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
import cv2
import numpy as np
from mvp_visual_tracking import VisualTracking
from mvp_core import Detection

def test_single_person_large_camera_shift():
    rng=np.random.default_rng(42)
    image=rng.integers(0,256,(720,1280,3),dtype=np.uint8)
    d=Detection(0,.99,(600,300,660,460),(630,310),.99)
    tracker=VisualTracking()
    assert tracker.update(image,[d],1)==[]
    moved=cv2.warpAffine(image,np.float32([[1,0,-200],[0,1,40]]),(1280,720))
    tracks=tracker.update(moved,[],1.033)
    assert len(tracks)==1
    assert tracks[0]['shift']==(-200,40)
    assert tracks[0]['source']==1

def test_different_image_does_not_claim_same_identity():
    rng=np.random.default_rng(2)
    tracker=VisualTracking()
    d=Detection(0,.99,(600,300,660,460),(630,310),.99)
    tracker.update(rng.integers(0,256,(720,1280,3),dtype=np.uint8),[d],1)
    assert tracker.update(rng.integers(0,256,(720,1280,3),dtype=np.uint8),[],1.033)==[]

def test_long_gap_does_not_reuse_visual_identity():
    image=np.random.default_rng(1).integers(0,256,(720,1280,3),dtype=np.uint8)
    tracker=VisualTracking()
    d=Detection(0,.99,(600,300,660,460),(630,310),.99)
    tracker.update(image,[d],1)
    assert tracker.update(image,[d],2)==[]
