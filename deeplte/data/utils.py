"""Utils for data."""

import jax
import jax.numpy as jnp
import numpy as np


def cartesian_product(*arrays):
    """Compute cartesian product of arrays
    with different shapes in an efficient manner.

    Args:
        arrays: each array shoud be rank 2 with shape (N_i, d_i).
        inds: indices for each array, should be rank 1.

    Returns:
        Cartesian product of arrays with shape (N_1, N_2, ..., N_n, sum(d_i)).
    """
    d = [*map(lambda x: x.shape[-1], arrays)]
    ls = [*map(len, arrays)]
    inds = [*map(np.arange, ls)]

    dtype = np.result_type(*arrays)
    arr = np.empty(ls + [sum(d)], dtype=dtype)

    for i, ind in enumerate(np.ix_(*inds)):
        arr[..., sum(d[:i]) : sum(d[: i + 1])] = arrays[i][ind]
    return arr


def jax_cartesian_product(arrays):
    """Compute cartesian product of arrays
    with different shapes in an efficient manner.

    Args:
        arrays: each array shoud be rank 2 with shape (N_i, d).
        inds: indices for each array, should be rank 1.

    Returns:
        Cartesian product of arrays with shape (N_1, N_2, ..., N_n, sum(d_i)).
    """
    num_arrs = len(arrays)

    def concat_fn(a):
        return jnp.concatenate(a, axis=-1)

    for i in range(num_arrs):
        in_axes = [None] * num_arrs
        in_axes[-i - 1] = int(0)
        concat_fn = jax.vmap(concat_fn, in_axes=(in_axes,))

    return concat_fn(arrays)


