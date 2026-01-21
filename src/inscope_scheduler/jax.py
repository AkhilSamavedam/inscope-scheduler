"""JAX helpers for applying a resource lease."""

from __future__ import annotations

import os
from typing import Iterable

from .lease import ResourceLease


def configure_jax_for_lease(lease: ResourceLease) -> None:
    if not lease.acquired:
        raise RuntimeError("lease is not acquired")
    gpu_ids = ",".join(str(gpu_id) for gpu_id in lease.gpu_ids)
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu_ids
