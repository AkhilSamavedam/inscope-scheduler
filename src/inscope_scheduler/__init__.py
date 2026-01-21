"""In-scope resource manager for ML training runs."""

from .jax import configure_jax_for_lease
from .manager import ResourceManager
from .probe import GPUInfo, SystemInfo, detect_gpus, detect_system
from .slurm_submit import SlurmResources, submit_slurm_job

__all__ = [
    "GPUInfo",
    "ResourceManager",
    "SystemInfo",
    "configure_jax_for_lease",
    "detect_gpus",
    "detect_system",
    "SlurmResources",
    "submit_slurm_job",
    "__version__",
]

__version__ = "0.1.0"
