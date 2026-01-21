"""Resource manager for acquiring GPU leases."""

from __future__ import annotations

import os
from pathlib import Path
from typing import List

from filelock import FileLock, Timeout

from .lease import ResourceLease
from .probe import GPUInfo, detect_gpus


class ResourceManager:
    def __init__(
        self,
        lock_dir: Path | None = None,
        env_var: str = "INSCOPE_LOCK_DIR",
    ) -> None:
        env_lock_dir = os.getenv(env_var) if env_var else None
        if lock_dir is None and env_lock_dir:
            lock_dir = Path(env_lock_dir)
        self.lock_dir = lock_dir or Path("/tmp/inscope_scheduler/locks")
        self.lock_dir.mkdir(parents=True, exist_ok=True)

    def available_gpus(self) -> List[GPUInfo]:
        return detect_gpus()

    def request_gpus(self, count: int = 1, timeout: float = 0.0) -> ResourceLease:
        if count <= 0:
            raise ValueError("count must be >= 1")

        gpus = self.available_gpus()
        if not gpus:
            raise RuntimeError("no GPUs detected")

        lease = ResourceLease(gpu_ids=[], lock_dir=self.lock_dir)
        try:
            for gpu in gpus:
                lock_path = self.lock_dir / f"gpu-{gpu.id}.lock"
                lock = FileLock(lock_path)
                try:
                    lock.acquire(timeout=timeout)
                except Timeout:
                    continue
                lease.gpu_ids.append(gpu.id)
                lease.locks.append(lock)
                if len(lease.gpu_ids) >= count:
                    lease.acquired = True
                    return lease
        except Exception:
            lease.release()
            raise

        lease.release()
        raise RuntimeError("insufficient free GPUs to satisfy request")
