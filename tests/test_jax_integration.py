from __future__ import annotations

import pytest

from inscope_scheduler import ResourceManager, configure_jax_for_lease, detect_system


def test_jax_training_step_with_lease() -> None:
    pytest.importorskip("jax")

    system = detect_system()
    if not system.gpus:
        pytest.skip("no GPUs available for lease")

    manager = ResourceManager()
    lease = manager.request_gpus(count=1, timeout=0.0)
    try:
        configure_jax_for_lease(lease)

        import jax
        import jax.numpy as jnp

        @jax.grad
        def loss_fn(w: jax.Array, x: jax.Array, y: jax.Array) -> jax.Array:
            pred = jnp.dot(x, w)
            return jnp.mean((pred - y) ** 2)

        x = jnp.ones((8, 4))
        y = jnp.ones((8,))
        w = jnp.zeros((4,))
        grad = loss_fn(w, x, y)
        updated = w - 0.1 * grad

        assert updated.shape == w.shape
    finally:
        lease.release()
