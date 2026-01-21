"""Slurm submission helpers."""

from __future__ import annotations

import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence


@dataclass
class SlurmResources:
    gpus: int = 1
    cpus_per_task: int | None = None
    mem: str | None = None
    time_limit: str | None = None
    partition: str | None = None
    job_name: str | None = None


def submit_slurm_job(
    script_path: Path,
    script_args: Sequence[str] | None = None,
    resources: SlurmResources | None = None,
    extra_sbatch_args: Iterable[str] | None = None,
    dry_run: bool = False,
) -> subprocess.CompletedProcess[str] | str:
    if not script_path.exists():
        raise FileNotFoundError(script_path)

    resources = resources or SlurmResources()
    cmd: list[str] = ["sbatch"]
    if resources.job_name:
        cmd += ["--job-name", resources.job_name]
    if resources.gpus:
        cmd += ["--gres", f"gpu:{resources.gpus}"]
    if resources.cpus_per_task:
        cmd += ["--cpus-per-task", str(resources.cpus_per_task)]
    if resources.mem:
        cmd += ["--mem", resources.mem]
    if resources.time_limit:
        cmd += ["--time", resources.time_limit]
    if resources.partition:
        cmd += ["--partition", resources.partition]

    if extra_sbatch_args:
        cmd += list(extra_sbatch_args)

    cmd.append(str(script_path))
    if script_args:
        cmd += list(script_args)

    if dry_run:
        return " ".join(cmd)

    return subprocess.run(cmd, check=True, text=True, capture_output=True)
