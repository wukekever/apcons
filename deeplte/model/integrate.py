"""Integration operator."""

from collections.abc import Callable
from typing import Optional

import jax.numpy as jnp

from deeplte.model.mapping import sharded_map


def quad(
    fun: Callable,
    quadratures: tuple[jnp.ndarray, jnp.ndarray],
    argnum: int = 0,
    shard_size: int | None = None,
    has_aux: Optional[bool] = False,
) -> Callable:
    """Compute the integral operator for a scalar function using
    quadratures.
    """
    points, weights = quadratures

    def integral_fn(*args):
        args = list(args)
        in_axes_ = [None] * len(args)
        args.insert(argnum, points)
        in_axes_.insert(argnum, int(0))
        out = sharded_map(fun, shard_size=shard_size, in_axes=in_axes_, out_axes=-1)(
            *args
        )
        if has_aux:
            values, aux = out
            result = jnp.dot(values, weights)
            return result, aux

        return jnp.dot(out, weights)

    return integral_fn
