"""Resource manager for acquiring GPU leases."""

from __future__ import annotations

import os
from pathlib import Path
from typing import List

from .backends import Backend, LocalBackend, SlurmBackend
from .lease import ResourceLease
from .probe import GPUInfo


class ResourceManager:
    def __init__(
        self,
        lock_dir: Path | None = None,
        env_var: str = "INSCOPE_LOCK_DIR",
        backend: Backend | None = None,
    ) -> None:
        if backend is not None:
            self.backend = backend
            self.lock_dir = getattr(backend, "lock_dir", Path("/tmp/inscope_scheduler/locks"))
            return

        env_lock_dir = os.getenv(env_var) if env_var else None
        if lock_dir is None and env_lock_dir:
            lock_dir = Path(env_lock_dir)
        self.lock_dir = lock_dir or Path("/tmp/inscope_scheduler/locks")
        self.lock_dir.mkdir(parents=True, exist_ok=True)

        if os.getenv("SLURM_JOB_ID"):
            self.backend = SlurmBackend()
        else:
            self.backend = LocalBackend(lock_dir=self.lock_dir)

    def available_gpus(self) -> List[GPUInfo]:
        return self.backend.available_gpus()

    def request_gpus(
        self,
        count: int = 1,
        timeout: float = 0.0,
        ttl_seconds: float | None = None,
        heartbeat_interval: float | None = 30.0,
    ) -> ResourceLease:
        return self.backend.request_gpus(
            count=count,
            timeout=timeout,
            ttl_seconds=ttl_seconds,
            heartbeat_interval=heartbeat_interval,
        )

    def reap_stale_leases(self, ttl_seconds: float) -> List[str]:
        lease_dir = self.lock_dir / "leases"
        if not lease_dir.exists():
            return []

        now = time.time()
        reclaimed: List[str] = []
        for heartbeat in lease_dir.glob("*.heartbeat"):
            age = now - heartbeat.stat().st_mtime
            if age <= ttl_seconds:
                continue
            lease_id = heartbeat.stem
            metadata_path = lease_dir / f"{lease_id}.json"
            heartbeat.unlink(missing_ok=True)
            metadata_path.unlink(missing_ok=True)
            reclaimed.append(lease_id)

        return reclaimed
