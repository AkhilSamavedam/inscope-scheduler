"""Demonstrate Slurm-aware lease usage.

Note: This does not submit jobs to Slurm. It assumes the script is already
running inside a Slurm allocation (SLURM_JOB_ID is set).
"""

from __future__ import annotations

import os

from inscope_scheduler import ResourceManager, configure_jax_for_lease


def main() -> None:
    slurm_job = os.getenv("SLURM_JOB_ID")
    print(f"SLURM_JOB_ID: {slurm_job}")
    print(f"CUDA_VISIBLE_DEVICES: {os.getenv('CUDA_VISIBLE_DEVICES')}")

    manager = ResourceManager()
    lease = manager.request_gpus(count=1, timeout=0.0)
    try:
        configure_jax_for_lease(lease)
        print(f"Lease acquired for GPUs: {lease.gpu_ids}")
    finally:
        lease.release()


if __name__ == "__main__":
    main()
