"""Dataset Pipeline"""

from __future__ import annotations

import pathlib
from collections.abc import Generator, Mapping, Sequence
from typing import MutableMapping, Optional

import jax
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
import tree
from absl import logging

from deeplte.data.utils import cartesian_product

Batch = Mapping[str, np.ndarray]
AUTOTUNE = tf.data.AUTOTUNE

FeatureDict = MutableMapping[str, np.ndarray]


def log_shapes(d: dict, name: str):
    logs = f"{name} shapes"
    for k, v in get_nest_dict_shape(d).items():
        logs += f", {k:s}: {v}"

    logging.info(logs)


def get_nest_dict_shape(d):
    return tf.nest.map_structure(lambda x: x.shape, d)


def process(
    data_path: str | pathlib.Path,
    pre_shuffle: bool = False,
    pre_shuffle_seed: int = 0,
    num_samples: int = 1024,
    is_split_samples: bool = True,
    split_rate: float = 0.875,
    save_path: Optional[str] = None,
    sample_mesh: int = 1,
):
    # data_path = '/workspaces/DeepLTE/data'
    data_path = pathlib.Path(data_path)
    lte_data = np.load(data_path / "lte_data.npz", allow_pickle=False)

    grid_feature, data_feature = {}, {}
    # shape - ((101,), (32,), (64,))
    grid_feature["grid_t"] = lte_data["grid_t"]
    grid_feature["grid_x"] = lte_data["grid_x"]
    grid_feature["grid_v"] = lte_data["grid_v"]
    # shape - (1024, 32, 64)
    data_feature["lte_init"] = lte_data["lte_init"]
    # shape - (1024, 101, 32)
    data_feature["lte_sol"] = lte_data["lte_sol"]

    shape = [lte_data["grid_x"].shape[0], lte_data["grid_v"].shape[0]]
    indices_v = [i for i in range(0, shape[-1] // 2, sample_mesh)] + [
        -1 - i for i in range(0, shape[-1] // 2, sample_mesh)
    ][::-1]
    # shape - (16,)
    grid_feature["grid_v"] = np.take(grid_feature["grid_v"], indices_v)
    # shape - (1024, 32, 16)
    data_feature["lte_init"] = np.take(data_feature["lte_init"], indices_v, axis=-1)

    if pre_shuffle:
        rng = np.random.default_rng(seed=pre_shuffle_seed)
        indices = np.arange(lte_data["lte_init"].shape[0])

        _ = rng.shuffle(indices)

        data_feature = tree.map_structure(
            lambda x: np.take(x, indices, axis=0), data_feature
        )

    if is_split_samples:
        num_val_samples = int(num_samples * (1 - split_rate))
        val_ds = tree.map_structure(
            lambda x: x[:num_val_samples],
            data_feature,
        )
        train_ds = tree.map_structure(
            lambda x: x[num_val_samples:],
            data_feature,
        )

        if save_path:
            if not isinstance(save_path, pathlib.Path):
                save_path = pathlib.Path(save_path)
        else:
            save_path = data_path / "lte_data_test.npz"

        np.savez(save_path, **val_ds, **grid_feature)

        return {**train_ds, **grid_feature}

    else:
        return {**data_feature, **grid_feature}


def tf_data_to_generator(
    tf_data: FeatureDict,
    is_training: bool,
    # batch_sizes should be:
    # [device_count, per_device_outer_batch_size]
    # total_batch_size = device_count * per_device_outer_batch_size
    batch_sizes: Sequence[int] = [jax.device_count(), 2],
    # num of samples for interior, boundary
    collocation_sizes: Sequence[int] = [3, 4],
    # shuffle buffer size
    buffer_size: int = 5_000,
    # Dataset options
    threadpool_size: int = 48,
    max_intra_op_parallelism: int = 1,
) -> Generator[Batch, None, None]:
    lte_data = tf_data

    # shape - ((101,), (32,), (64,))
    grid_t, grid_x, grid_v = lte_data["grid_t"], lte_data["grid_x"], lte_data["grid_v"]
    num_t, num_x, num_v = grid_t.shape[0], grid_x.shape[0], grid_v.shape[0]

    # shape - (768, 32, 64), (768, 101, 32)
    lte_init, lte_label = lte_data["lte_init"], lte_data["lte_sol"]
    # (768, 32*64)
    # lte_init_vec = lte_init.reshape(-1, num_x * num_v)
    lte_init_vec = lte_init.reshape(-1, num_x, num_v)

    # (768, 101*32)
    lte_label_vec = lte_label.reshape(-1, num_t * num_x)
    # shape - (32*64, 2)
    combinations_xv = cartesian_product(grid_x[:, None], grid_v[:, None]).reshape(-1, 2)
    # shape - (32*64, 1), (32*64, 1)
    x_vec, v_vec = np.split(combinations_xv, indices_or_sections=[1], axis=-1)
    # shape - (101*32, 2)
    combinations_tx = cartesian_product(grid_t[:, None], grid_x[:, None]).reshape(-1, 2)
    # shape - (101*32, 1), (101*32, 1)
    _t_vec, _x_vec = np.split(combinations_tx, indices_or_sections=[1], axis=-1)

    ds = tf.data.Dataset.from_tensor_slices({"a": lte_init_vec, "label": lte_label_vec})

    if is_training:
        if jax.process_count() > 1:
            # Only cache if we are reading a subset of the dataset.
            ds = ds.cache()
        ds = ds.repeat()
        ds = ds.shuffle(buffer_size=buffer_size)

    # # batch per_device outer first
    # # since they share the same points
    # ds = ds.batch(batch_sizes[-1], drop_remainder=True)
    # # batch device dim
    # ds = ds.batch(batch_sizes[0], drop_remainder=True)
    # # or
    # Perform regular batching with reduced number of elements.
    # batch_sizes = [device_count, per_device_outer_batch_size]
    for batch_size in reversed(batch_sizes):
        ds = ds.batch(batch_size, drop_remainder=True)

    options = tf.data.Options()
    options.threading.max_intra_op_parallelism = max_intra_op_parallelism
    options.threading.private_threadpool_size = threadpool_size
    options.experimental_optimization.map_parallelization = True
    options.experimental_optimization.parallel_batch = True
    if is_training:
        options.deterministic = False

    def mesh_sampler(ds):
        interior_size, boundary_size = collocation_sizes
        # interior points
        t_interior = tf.random.uniform((interior_size, 1)) * grid_t[-1]
        x_interior = grid_x[0] + tf.random.uniform((interior_size, 1)) * (
            grid_x[-1] - grid_x[0]
        )
        v_interior = grid_v[0] + tf.random.uniform((interior_size, 1)) * (
            grid_v[-1] - grid_v[0]
        )
        # boundary points
        t_boundary = tf.random.uniform((boundary_size, 1)) * grid_t[-1]
        v_boundary = tf.random.uniform((boundary_size, 1)) * grid_v[-1]

        tf_dict = {
            "interior": {"t": t_interior, "x": x_interior, "v": v_interior},
            "boundary": {"t": t_boundary, "v": v_boundary},
            "grids": {
                "initial": {"x": x_vec, "v": v_vec},
                "density": {"t": _t_vec, "x": _x_vec},
            },
        }

        fn = lambda x: tf.tile(  # noqa: E731
            x[None, :], multiples=[batch_sizes[0]] + [1] * (len(x[None, :].shape) - 1)
        )
        # repeat_inner_batch
        tf_dict = tf.nest.map_structure(fn, tf_dict)

        tf_dict["func"] = ds

        return tf_dict

    ds = ds.map(mesh_sampler, num_parallel_calls=AUTOTUNE)
    ds = ds.prefetch(AUTOTUNE)
    ds = ds.with_options(options)

    # convert to a numpy generator
    yield from tfds.as_numpy(ds)


def make_device_batch(
    global_batch_size: int,
    num_devices: int,
):
    per_device_batch_size, ragged = divmod(global_batch_size, num_devices)
    # Raise error if not divisible
    if ragged:
        raise ValueError(
            f"Global batch size {global_batch_size} must be divisible by "
            f"number of devices {num_devices}"
        )
    return [num_devices, per_device_batch_size]
