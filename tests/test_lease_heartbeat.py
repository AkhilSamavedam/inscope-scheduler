import time
from pathlib import Path

from inscope_scheduler.lease import ResourceLease


def test_heartbeat_writes_timestamp(tmp_path: Path) -> None:
    lease = ResourceLease(gpu_ids=[0], lock_dir=tmp_path, lease_id="abc123")

    lease.heartbeat()
    lease_dir = tmp_path / "leases"
    heartbeat_files = list(lease_dir.glob("*.heartbeat"))
    assert len(heartbeat_files) == 1
    assert float(heartbeat_files[0].read_text().strip()) > 0.0


def test_time_remaining_and_checkpoint(tmp_path: Path) -> None:
    lease = ResourceLease(gpu_ids=[0], lock_dir=tmp_path, ttl_seconds=0.1)
    assert lease.should_checkpoint(grace_seconds=1.0)
    time.sleep(0.2)
    assert lease.time_remaining() == 0.0
