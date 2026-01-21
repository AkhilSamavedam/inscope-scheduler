"""Minimal demo for acquiring a GPU lease before importing JAX."""

from __future__ import annotations

from inscope_scheduler import ResourceManager, configure_jax_for_lease, detect_system


def main() -> None:
    system = detect_system()
    print(f"CPU cores (logical/physical): {system.cpu_logical_cores}/{system.cpu_physical_cores}")
    print(f"RAM total (MB): {system.memory_total_mb}")
    print(f"Detected GPUs: {[gpu.name for gpu in system.gpus]}")

    manager = ResourceManager()
    try:
        lease = manager.request_gpus(count=1, timeout=0.0)
    except RuntimeError as exc:
        print(f"GPU lease unavailable: {exc}")
        return
    try:
        configure_jax_for_lease(lease)
        # Import JAX only after the environment is configured.
        import jax  # noqa: F401

        print(f"Lease acquired for GPUs: {lease.gpu_ids}")
    finally:
        lease.release()


if __name__ == "__main__":
    main()
