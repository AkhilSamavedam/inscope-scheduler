"""Submit a Slurm job from Python."""

from __future__ import annotations

from pathlib import Path

from inscope_scheduler import SlurmResources, submit_slurm_job


def main() -> None:
    resources = SlurmResources(
        gpus=1,
        cpus_per_task=4,
        mem="16G",
        time_limit="00:30:00",
        job_name="inscope-demo",
    )
    result = submit_slurm_job(
        script_path=Path("examples/slurm_demo.py"),
        resources=resources,
    )
    print(result.stdout.strip())


if __name__ == "__main__":
    main()
