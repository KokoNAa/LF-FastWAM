from pathlib import Path
import pytest
from scripts.run_robotwin_memory_guard_trial import source_process


def test_wait_binds_actual_command_root_and_kernel_start_time(tmp_path):
    proc = tmp_path / '123'
    proc.mkdir()
    fields = ['S'] + ['0'] * 18 + ['6789']
    (proc / 'stat').write_text('123 (python) ' + ' '.join(fields))
    (proc / 'cmdline').write_bytes(b'python\0/repo/run_robotwin_calibrated_joint_trial.py\0--output\0/runs/source\0')
    assert source_process(123, Path('/runs/source'), proc_root=tmp_path) == '6789'
    with pytest.raises(ValueError, match='another command'):
        source_process(123, Path('/runs/other'), proc_root=tmp_path)
    (proc / 'cmdline').write_bytes(b'python\0unrelated.py\0/runs/source\0')
    with pytest.raises(ValueError, match='another command'):
        source_process(123, Path('/runs/source'), proc_root=tmp_path)
    fields[0] = 'Z'
    (proc / 'stat').write_text('123 (python) ' + ' '.join(fields))
    assert source_process(123, Path('/runs/source'), proc_root=tmp_path) is None
    assert source_process(999, Path('/runs/source'), proc_root=tmp_path) is None
