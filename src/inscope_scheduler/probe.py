"""Resource discovery for CPU, RAM, and GPU."""

from __future__ import annotations

from typing import List

import psutil
from pydantic import BaseModel

try:
    import pynvml as nvml
except Exception:  # pragma: no cover - optional GPU dependency
    nvml = None


def _nvml_str(value: object) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8")
    return str(value)


class GPUInfo(BaseModel):
    id: int
    name: str
    total_memory_mb: int
    uuid: str


class SystemInfo(BaseModel):
    cpu_logical_cores: int
    cpu_physical_cores: int
    memory_total_mb: int
    gpus: List[GPUInfo]


def detect_gpus() -> List[GPUInfo]:
    if nvml is None:
        return []

    gpus: List[GPUInfo] = []
    nvml.nvmlInit()
    try:
        count = nvml.nvmlDeviceGetCount()
        for idx in range(count):
            handle = nvml.nvmlDeviceGetHandleByIndex(idx)
            name = _nvml_str(nvml.nvmlDeviceGetName(handle))
            mem = nvml.nvmlDeviceGetMemoryInfo(handle)
            uuid = _nvml_str(nvml.nvmlDeviceGetUUID(handle))
            gpus.append(
                GPUInfo(
                    id=idx,
                    name=name,
                    total_memory_mb=int(mem.total // (1024 * 1024)),
                    uuid=uuid,
                )
            )
    finally:
        nvml.nvmlShutdown()

    return gpus


def detect_system() -> SystemInfo:
    memory_mb = int(psutil.virtual_memory().total // (1024 * 1024))
    cpu_physical = psutil.cpu_count(logical=False) or 0
    cpu_logical = psutil.cpu_count(logical=True) or 0
    return SystemInfo(
        cpu_logical_cores=cpu_logical,
        cpu_physical_cores=cpu_physical,
        memory_total_mb=memory_mb,
        gpus=detect_gpus(),
    )
