from __future__ import annotations

from pathlib import Path

import pytest

import inscope_scheduler.backends as backends_module
import inscope_scheduler.manager as manager_module


def test_request_gpus_acquires_and_releases_lock(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    fake_gpus = [manager_module.GPUInfo(id=0, name="fake", total_memory_mb=1024, uuid="gpu0")]
    monkeypatch.setattr(backends_module, "detect_gpus", lambda: fake_gpus)

    backend = backends_module.LocalBackend(lock_dir=tmp_path)
    manager = manager_module.ResourceManager(backend=backend)
    lease = manager.request_gpus(count=1, timeout=0.0)
    assert lease.acquired
    assert lease.gpu_ids == [0]

    lease.release()
    assert not lease.acquired
    assert lease.locks[0].is_locked is False


def test_lock_dir_from_env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    env_dir = tmp_path / "locks"
    monkeypatch.setenv("INSCOPE_LOCK_DIR", str(env_dir))
    manager = manager_module.ResourceManager(lock_dir=None)
    assert manager.lock_dir == env_dir
