import sys,math
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
import pytest
from mvp_one_euro import OneEuro
from report_latency import distribution

def test_constant_input_stays_constant():
    f=OneEuro()
    assert all(f(12,i/30)==12 for i in range(100))

def test_reduces_stationary_noise_and_adapts_to_motion():
    slow=OneEuro(beta=0);fast=OneEuro(beta=.1)
    raw=[(-1)**i for i in range(100)]
    smooth=[slow(x,i/30) for i,x in enumerate(raw)]
    assert max(smooth[20:])-min(smooth[20:])<1
    slow.reset()
    for i in range(30):
        a=slow(i*10,i/30);b=fast(i*10,i/30)
    assert abs(290-b)<abs(290-a)

def test_duplicate_source_is_rejected():
    f=OneEuro();f(1,1)
    with pytest.raises(ValueError):f(2,1)

def test_latency_report_excludes_missing_and_invalid_samples():
    r=distribution([1,2,3,None,float('nan'),-1])
    assert r['n']==3 and r['p50']==2 and r['max']==3
