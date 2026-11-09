import sys
from pathlib import Path
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from profile_stutter import summarize_csv

def test_main_swapchain_and_long_frames_are_counted(tmp_path):
    p=tmp_path/'trace.csv'
    p.write_text('SwapChainAddress,MsBetweenPresents\nA,10\nA,20\nA,60\nA,120\nB,1000\n')
    s=summarize_csv(p)
    assert s['n']==4 and s['median_ms']==40 and s['max_ms']==120
    assert s['above_50ms']==2 and s['above_100ms']==1

def test_empty_capture_is_an_error_not_zero_stutters(tmp_path):
    p=tmp_path/'trace.csv';p.write_text('SwapChainAddress,MsBetweenPresents\n')
    with pytest.raises(ValueError,match='No CS2 present'):summarize_csv(p)

def test_minimized_game_is_waited_for_instead_of_aborting():
    from profile_stutter import wait_for_game
    now=[0.];windows=iter([None,None,7])
    hwnd=wait_for_game(lambda _:next(windows),lambda hwnd:hwnd==7,clock=lambda:now[0],sleep=lambda dt:now.__setitem__(0,now[0]+dt))
    assert hwnd==7 and now[0]>=.2

def test_presentmon_actual_lowercase_ms_header_is_supported(tmp_path):
    p=tmp_path/'trace.csv'
    p.write_text('Application,SwapChainAddress,msBetweenPresents\ncs2.exe,A,8\ncs2.exe,A,80\n')
    s=summarize_csv(p)
    assert s['n']==2 and s['above_50ms']==1 and s['max_ms']==80


def test_system_trace_finalizes_same_named_session_after_stage_failure(tmp_path, monkeypatch):
    import profile_stutter

    commands = []
    monkeypatch.setattr(profile_stutter.subprocess, 'run', lambda args, **kwargs: commands.append(args))
    with pytest.raises(RuntimeError, match='lost focus'):
        with profile_stutter.system_capture_trace(tmp_path):
            raise RuntimeError('lost focus')

    start, stop = commands
    assert start[:4] == ['wpr', '-start', 'GPU.light', '-filemode']
    assert stop[:3] == ['wpr', '-stop', str(tmp_path / 'system.etl')]
    assert start[-2:] == stop[-2:] == ['-instancename', f'CS2ML_capture_{tmp_path.name}']
