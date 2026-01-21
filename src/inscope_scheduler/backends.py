"""Backend implementations for resource management."""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import List, Protocol

from filelock import FileLock, Timeout

from .lease import ResourceLease
from .probe import GPUInfo, detect_gpus


class Backend(Protocol):
    name: str

    def available_gpus(self) -> List[GPUInfo]:
        ...

    def request_gpus(
        self,
        count: int,
        timeout: float,
        ttl_seconds: float | None,
        heartbeat_interval: float | None,
    ) -> ResourceLease:
        ...


@dataclass
class LocalBackend:
    lock_dir: Path
    name: str = "local"

    def available_gpus(self) -> List[GPUInfo]:
        return detect_gpus()

    def request_gpus(
        self,
        count: int,
        timeout: float,
        ttl_seconds: float | None,
        heartbeat_interval: float | None,
    ) -> ResourceLease:
        if count <= 0:
            raise ValueError("count must be >= 1")

        gpus = self.available_gpus()
        if not gpus:
            raise RuntimeError("no GPUs detected")

        self.lock_dir.mkdir(parents=True, exist_ok=True)
        lease_dir = self.lock_dir / "leases"
        lease_dir.mkdir(parents=True, exist_ok=True)

        lease_id = os.urandom(8).hex()
        lease = ResourceLease(
            gpu_ids=[],
            lock_dir=self.lock_dir,
            lease_id=lease_id,
            ttl_seconds=ttl_seconds,
        )
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
                    if lease._metadata_path:
                        lease._metadata_path.write_text(
                            json.dumps(
                                {
                                    "lease_id": lease_id,
                                    "gpu_ids": lease.gpu_ids,
                                    "pid": os.getpid(),
                                    "created_at": time.time(),
                                    "ttl_seconds": ttl_seconds,
                                }
                            )
                        )
                    lease.heartbeat()
                    if heartbeat_interval is not None:
                        lease.start_heartbeat(interval_seconds=heartbeat_interval)
                    lease.register_atexit_cleanup()
                    return lease
        except Exception:
            lease.release()
            raise

        lease.release()
        raise RuntimeError("insufficient free GPUs to satisfy request")


@dataclass
class SlurmBackend:
    name: str = "slurm"

    def available_gpus(self) -> List[GPUInfo]:
        return detect_gpus()

    def request_gpus(
        self,
        count: int,
        timeout: float,
        ttl_seconds: float | None,
        heartbeat_interval: float | None,
    ) -> ResourceLease:
        slurm_job = os.getenv("SLURM_JOB_ID")
        if not slurm_job:
            raise RuntimeError("SLURM_JOB_ID not set; not running under Slurm")

        visible = os.getenv("CUDA_VISIBLE_DEVICES", "")
        gpu_ids = [int(x) for x in visible.split(",") if x.strip().isdigit()]
        if not gpu_ids:
            raise RuntimeError("CUDA_VISIBLE_DEVICES is empty; Slurm did not assign GPUs")
        if count > len(gpu_ids):
            raise RuntimeError("requested more GPUs than assigned by Slurm")

        lease = ResourceLease(
            gpu_ids=gpu_ids[:count],
            lock_dir=Path("/tmp/inscope_scheduler/locks"),
            lease_id=slurm_job,
            ttl_seconds=ttl_seconds,
        )
        lease.acquired = True
        if heartbeat_interval is not None:
            lease.start_heartbeat(interval_seconds=heartbeat_interval)
        lease.register_atexit_cleanup()
        return lease
