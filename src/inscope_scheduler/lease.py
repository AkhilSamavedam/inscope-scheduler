"""Resource lease acquisition and release."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import List
import threading
import time
import atexit

from filelock import FileLock


@dataclass
class ResourceLease:
    gpu_ids: List[int]
    lock_dir: Path
    locks: List[FileLock] = field(default_factory=list)
    acquired: bool = False
    lease_id: str | None = None
    ttl_seconds: float | None = None
    started_at: float = field(default_factory=time.monotonic)
    _heartbeat_thread: threading.Thread | None = field(default=None, init=False, repr=False)
    _heartbeat_stop: threading.Event = field(default_factory=threading.Event, init=False, repr=False)
    _atexit_registered: bool = field(default=False, init=False, repr=False)
    _heartbeat_path: Path | None = field(default=None, init=False, repr=False)
    _metadata_path: Path | None = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        if self.lease_id:
            self._prepare_paths()

    def _prepare_paths(self) -> None:
        if not self.lease_id:
            return
        lease_dir = self.lock_dir / "leases"
        self._heartbeat_path = lease_dir / f"{self.lease_id}.heartbeat"
        self._metadata_path = lease_dir / f"{self.lease_id}.json"

    def heartbeat(self) -> None:
        if self._heartbeat_path is None:
            return
        self._heartbeat_path.parent.mkdir(parents=True, exist_ok=True)
        self._heartbeat_path.write_text(str(time.time()))

    def start_heartbeat(self, interval_seconds: float = 30.0) -> None:
        if interval_seconds <= 0:
            raise ValueError("interval_seconds must be > 0")
        if self._heartbeat_path is None:
            return
        if self._heartbeat_thread and self._heartbeat_thread.is_alive():
            return

        self._heartbeat_stop.clear()

        def _worker() -> None:
            while not self._heartbeat_stop.is_set():
                self.heartbeat()
                self._heartbeat_stop.wait(interval_seconds)

        self._heartbeat_thread = threading.Thread(target=_worker, daemon=True)
        self._heartbeat_thread.start()

    def stop_heartbeat(self) -> None:
        self._heartbeat_stop.set()
        if self._heartbeat_thread:
            self._heartbeat_thread.join(timeout=1.0)

    def register_atexit_cleanup(self) -> None:
        if self._atexit_registered:
            return

        def _cleanup() -> None:
            self.release()

        atexit.register(_cleanup)
        self._atexit_registered = True

    def time_remaining(self) -> float | None:
        if self.ttl_seconds is None:
            return None
        return max(0.0, self.ttl_seconds - (time.monotonic() - self.started_at))

    def should_checkpoint(self, grace_seconds: float = 60.0) -> bool:
        remaining = self.time_remaining()
        if remaining is None:
            return False
        return remaining <= grace_seconds

    def release(self) -> None:
        self.stop_heartbeat()
        for lock in reversed(self.locks):
            if lock.is_locked:
                lock.release()
        if self._heartbeat_path and self._heartbeat_path.exists():
            self._heartbeat_path.unlink()
        if self._metadata_path and self._metadata_path.exists():
            self._metadata_path.unlink()
        self.acquired = False

    def __enter__(self) -> "ResourceLease":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.release()
