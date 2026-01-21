"""Resource lease acquisition and release."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import List

from filelock import FileLock


@dataclass
class ResourceLease:
    gpu_ids: List[int]
    lock_dir: Path
    locks: List[FileLock] = field(default_factory=list)
    acquired: bool = False

    def release(self) -> None:
        for lock in reversed(self.locks):
            if lock.is_locked:
                lock.release()
        self.acquired = False

    def __enter__(self) -> "ResourceLease":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.release()
