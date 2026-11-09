import sys
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from mvp_native_trt import decode_pose

def test_nms_is_class_aware_and_coordinates_undo_letterbox():
    # same-class duplicate is removed; overlapping opposing team is retained.
    rows=np.array([[320,200,80,180,.9,.1,320,130,.99],
                   [321,201,80,180,.8,.1,321,131,.98],
                   [320,200,80,180,.1,.85,320,130,.97]],np.float32)
    ds=decode_pose(rows.T[None],(1280,720),.5,(0,8))
    assert len(ds)==2 and {d.cls for d in ds}=={0,1}
    assert all(d.head==(640,244) for d in ds)
    assert all(d.box==(560,204,720,564) for d in ds)

def test_empty_and_nonfinite_network_outputs_are_ignored():
    assert decode_pose(np.zeros((1,9,2),np.float32),(1280,720),1,(0,0))==[]
    x=np.ones((1,9,1),np.float32);x[0,6,0]=np.nan
    assert decode_pose(x,(1280,720),1,(0,0))==[]
