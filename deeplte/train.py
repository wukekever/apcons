"""DeepLTE experiment."""
# tensorboard --logdir=./

import functools
import time
from collections.abc import Generator, Mapping

import haiku as hk
import jax
import jax.numpy as jnp
import matplotlib
import matplotlib.pyplot as plt
import optax
from absl import logging
from jaxline import experiment
from jaxline import utils as jl_utils
from matplotlib.colors import ListedColormap

from deeplte import optimizers
from deeplte.data.pipeline import (
    FeatureDict,
    make_device_batch,
    process,
    tf_data_to_generator,
)
from deeplte.model import modules_APDONv2 as modules

# from deeplte.model import modules_APDONv1 as modules


# from deeplte.model import modules_PIDON as modules


OptState = tuple[optax.TraceState, optax.ScaleByScheduleState, optax.ScaleState]
Scalars = Mapping[str, jax.Array]


def _format_logs(prefix, results):
    # f_list for less verbosity; e.g., "4." instead of
    # "array(4., dtype=float32)".
    logging_str = " - ".join(
        [
            f"{k}: {results[k]:.2%}" if k[-2:] == "pe" else f"{k}: {results[k]}"
            for k in sorted(results.keys())
        ]
    )
    logging.info("%s - %s", prefix, logging_str)


class Trainer(experiment.AbstractExperiment):
    """LTE solver."""

    # A map from object properties that will be checkpointed to their name
    # in a checkpoint. Currently we assume that these are all sharded
    # device arrays.
    CHECKPOINT_ATTRS = {
        "_params": "params",
        "_state": "state",
        "_opt_state": "opt_state",
    }

    def __init__(self, mode, init_rng, config):
        """Initializes solver."""
        super().__init__(mode=mode, init_rng=init_rng)

        if mode not in ("train", "eval", "train_eval_multithreaded"):
            raise ValueError(f"Invalid mode {mode}.")

        self.mode = mode
        self.init_rng = init_rng
        self.config = config  # config.experiment_kwargs.config

        # Checkpointed experiment state.
        self._params = None
        self._state = None
        self._opt_state = None

        # Initialize train and eval functions
        self._train_input = None
        self._eval_input = None
        self._lr_schedule = None

        # Track what has started
        self._training = False
        self._evaluating = False
        self._test = False
        self._density = {"predictions": [], "labels": []}

        # Initialize model functions
        def _forward_fn(*args, **kwargs):
            model = modules.DeepLTE(self.config.model)
            return model(*args, **kwargs)

        self.model = hk.transform_with_state(_forward_fn)

        self._process_data()  # self.tf_data = process()

    #  _             _
    # | |_ _ __ __ _(_)_ __
    # | __| '__/ _` | | '_ \
    # | |_| | | (_| | | | | |
    #  \__|_|  \__,_|_|_| |_|
    #

    def step(self, global_step, rng, *unused_args, **unused_kwargs):
        """See base class."""
        if not self._training:
            self._initialize_training()

        # Get next batch
        batch = next(self._train_input)

        # Update parameters
        outputs = self.update_fn(
            self._params, self._state, self._opt_state, global_step, rng, batch
        )
        self._params = outputs["params"]
        self._state = outputs["state"]
        self._opt_state = outputs["opt_state"]

        # We only return the loss scalars on the first devict for logging
        scalars = jl_utils.get_first(outputs["scalars"])

        # Evaluate
        if global_step[0] % 2240 == 0:
            self.evaluate(global_step, rng)

        return scalars

    def _update_fn(self, params, state, opt_state, global_step, rng, batch):
        # Logging dict.
        scalars = {}

        def loss(params):
            (loss, scalars), out_state = self.model.apply(
                params,
                state,
                rng,
                batch,
                compute_loss=True,
                compute_metrics=False,
            )
            scaled_loss = loss / jax.local_device_count()
            return scaled_loss, (scalars["loss"], out_state)

        # Gradient function w.r.t. params
        grad_fn = jax.grad(loss, has_aux=True)
        # Compute loss and gradients.
        scaled_grads, (loss_scalars, new_state) = grad_fn(params)
        grads = jax.lax.psum(scaled_grads, axis_name="i")

        # Grab the learning rate to log before performing the step.
        learning_rate = self._lr_schedule(global_step)
        scalars["learning_rate"] = learning_rate

        # Update params
        updates, opt_state = self.optimizer.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)

        # Update scalars dict
        loss_scalars = {f"train_{k}": v for k, v in loss_scalars.items()}
        scalars.update(loss_scalars)
        scalars = jax.lax.pmean(scalars, axis_name="i")
        return {
            "params": params,
            "state": new_state,
            "opt_state": opt_state,
            "scalars": scalars,
        }

    def _initialize_training(self):
        # Less verbose
        c = self.config

        # Performs prefetching of elements from an iterable
        # in a separate thread.
        train_input = jl_utils.py_prefetch(self._build_train_input)
        # This keeps two batches per-device in memory at all times, allowing
        # h2d transfers to overlap with execution.
        self._train_input = jl_utils.double_buffer_on_gpu(train_input)

        # Total batch size
        total_batch_size = c.training.batch_size

        # NOTE: Since we may have repeat number for the same batch
        # with different collocation points, stpes_per_epoch should be
        # multiplied by repeat.
        steps_per_epoch = (
            c.dataset.data_split.num_train_samples
            / c.training.batch_size
            * c.training.batch_repeat
        )
        total_steps = c.training.num_epochs * steps_per_epoch

        # Get learning rate schedule.
        self._lr_schedule = optimizers.get_learning_rate_schedule(
            total_batch_size,
            steps_per_epoch,
            total_steps,
            c.optimizer,
        )
        # Optimizer
        self.optimizer = optimizers.make_optimizer(c.optimizer, self._lr_schedule)

        # Initialize net if no params available.
        if self._params is None:
            logging.info("Initializing parameters.")

            # Pmap initial functions
            # init_net = jax.pmap(lambda *a: self.solution.init(*a))
            init_net = functools.partial(self.model.init, compute_loss=True)
            init_net = jax.pmap(init_net)
            init_opt = jax.pmap(self.optimizer.init)

            # Init uses the same RNG key on all hosts+devices to ensure
            # everyone computes the same initial state.
            init_rng = jl_utils.bcast_local_devices(
                self.init_rng
            )  # Array([42], dtype=int32, weak_type=True)

            # Load initial inputs
            dummy_inputs = self._build_dummy_input()
            self._params, self._state = init_net(init_rng, dummy_inputs)
            self._opt_state = init_opt(self._params)

            # Log total number of parameters
            num_params = hk.data_structures.tree_size(self._params)
            logging.info("Net parameters: %d", num_params // jax.local_device_count())

        # NOTE: We "donate" the `params, state, opt_state` arguments which
        # allows JAX (on some backends) to reuse the device memory associated
        # with these inputs to store the outputs of our function (which also
        # start with `params, state, opt_state`).
        self.update_fn = jax.pmap(self._update_fn, axis_name="i")

        # Set training state to True after initialization
        self._training = True

    def _build_train_input(self) -> Generator[FeatureDict, None, None]:
        """Build train input as generator/iterator."""
        c = self.config
        batch_sizes = make_device_batch(c.training.batch_size, jax.device_count())
        train_ds = tf_data_to_generator(
            tf_data=self.tf_data,
            is_training=True,
            batch_sizes=batch_sizes,
            collocation_sizes=c.training.collocation_sizes,
            buffer_size=c.dataset.buffer_size,
            threadpool_size=c.dataset.threadpool_size,
            max_intra_op_parallelism=c.dataset.max_intra_op_parallelism,
        )
        return train_ds

    #                  _
    #   _____   ____ _| |
    #  / _ \ \ / / _` | |
    # |  __/\ V / (_| | |
    #  \___| \_/ \__,_|_|
    #

    def evaluate(self, global_step, rng: jax.Array, **unused_args) -> Scalars:
        """See base class."""
        if not self._evaluating:
            self._initialize_evaluation()

        # Get global step value on the first device for logging.
        global_step_value = jl_utils.get_first(global_step)
        logging.info("Running evaluation at global_step %s...", global_step_value)

        t_0 = time.time()
        # Run evaluation for an epoch
        metrics = self._eval_epoch(self._params, self._state, rng)
        # Covert jnp.ndarry to list to have less verbose.
        metrics = jax.tree_util.tree_map(
            lambda x: x.tolist() if isinstance(x, jax.Array) else x, metrics
        )

        t_diff = time.time() - t_0

        _format_logs(
            f"Evaluation time {t_diff:.1f}s, " f"global_step {global_step_value}",
            metrics,
        )

        return metrics

    def _eval_epoch(self, params: hk.Params, state: hk.State, rng: jax.Array):
        """Evaluates an epoch."""
        num_examples = 0.0
        summed_metrics = None

        for batch in self._eval_input():
            # Account for pmaps
            num_examples += jnp.prod(jnp.array(batch["func"]["label"].shape[:2]))
            metrics = self.eval_fn(params, state, rng, batch)
            # Accumulate the sum of scalars for each step.
            metrics = jax.tree_util.tree_map(lambda x: jnp.sum(x[0], axis=0), metrics)
            if summed_metrics is None:
                summed_metrics = metrics
            else:
                summed_metrics = jax.tree_util.tree_map(
                    jnp.add, summed_metrics, metrics
                )

        # Compute mean_metrics
        mean_metrics = jax.tree_util.tree_map(
            lambda x: x / num_examples, summed_metrics
        )

        # Eval metrics dict
        metrics = {}
        # Take sqrt if it is squared
        for k, v in mean_metrics.items():
            metrics["eval_" + k] = jnp.sqrt(v) if k.split("_")[-1][0] == "r" else v

        return metrics

    def _eval_fn(self, params, state, rng, batch):
        """Evaluates a batch."""
        outputs, state = self.model.apply(
            params, state, rng, batch, compute_loss=False, compute_metrics=True
        )

        # NOTE: Returned values will be summed and finally divided
        # by num_samples.
        return jax.lax.psum(outputs["metrics"], axis_name="i")

    def _initialize_evaluation(self):
        def prefetch_and_double_buffer_input():
            # Performs prefetching of elements from an iterable
            # in a separate thread.
            eval_input = jl_utils.py_prefetch(self._build_eval_input)
            # This keeps two batches per-device in memory at all times,
            # allowing h2d transfers to overlap with execution.
            return jl_utils.double_buffer_on_gpu(eval_input)
            # return eval_input

        # Evaluation input as a Generator
        self._eval_input = prefetch_and_double_buffer_input

        # We pmap the evaluation function
        self.eval_fn = jax.pmap(self._eval_fn, axis_name="i")

        # Set evaluating state to True after initialization.
        self._evaluating = True

    def _build_eval_input(self) -> Generator[FeatureDict, None, None]:
        c = self.config
        batch_sizes = make_device_batch(c.training.batch_size, jax.device_count())
        val_ds = tf_data_to_generator(
            tf_data=self.tf_data,
            is_training=False,
            batch_sizes=batch_sizes,
            buffer_size=c.dataset.buffer_size,
            threadpool_size=c.dataset.threadpool_size,
            max_intra_op_parallelism=c.dataset.max_intra_op_parallelism,
        )
        return val_ds

    def test(self, rng: jax.Array, **unused_args) -> Scalars:
        if not self._test:
            self._initialize_test()

        logging.info("Running testing")

        # Plot the density
        results, (grid_t, grid_x) = self._get_density(rng)
        predictions = results["predictions"]
        labels = results["labels"]

        fig, _axs = plt.subplots(nrows=1, ncols=3, figsize=(20, 12))
        fig.subplots_adjust(hspace=0.3)
        axs = _axs.flatten()
        viridis = matplotlib.colormaps["viridis"](jnp.linspace(0, 1, 128))
        cs_1 = axs[0].contourf(
            grid_t,
            grid_x,
            predictions[0].reshape(-1, 32),
            cmap=ListedColormap(viridis),
        )
        axs[0].set_title(r"Predicted density $\rho(t, x)$", fontsize=20)
        axs[0].tick_params(axis="both", labelsize=15)
        cbar = fig.colorbar(cs_1)
        cbar.ax.tick_params(labelsize=16)
        cs_2 = axs[1].contourf(
            grid_t, grid_x, labels[0].reshape(-1, 32), cmap=ListedColormap(viridis)
        )
        axs[1].set_title(r"True density $\rho_{ex}(t, x)$", fontsize=20)
        axs[1].tick_params(axis="both", labelsize=15)
        cbar = fig.colorbar(cs_2)
        cbar.ax.tick_params(labelsize=16)
        cs_3 = axs[2].contourf(
            grid_t,
            grid_x,
            jnp.abs(predictions[0] - labels[0]).reshape(-1, 32),
            cmap=ListedColormap(viridis),
        )
        axs[2].set_title(
            r"Absolute error $|\rho(t, x) - \rho_{ex}(t, x)|$", fontsize=20
        )
        axs[2].tick_params(axis="both", labelsize=15)
        cbar = fig.colorbar(cs_3)
        cbar.ax.tick_params(labelsize=16)

        plt.tight_layout()
        plt.savefig("/workspaces/DeepLTE/figures/rho_test.png")
        plt.close()

    def _get_density(self, rng: jax.Array, **unused_args):
        if not self._test:
            self._initialize_test()

        # batch = self._test_batch
        batch = next(self._test_input)

        grid_t = batch["grids"]["density"]["t"][0].reshape(-1, 32)
        grid_x = batch["grids"]["density"]["x"][0].reshape(-1, 32)
        results = self.test_fn(self._params, self._state, rng, batch)
        self._density["predictions"].append(results["predictions"])
        self._density["labels"].append(results["labels"])
        return results, (grid_t, grid_x)

    def _initialize_test(self):
        test_input = jl_utils.py_prefetch(self._build_test_input)
        self._test_input = jl_utils.double_buffer_on_gpu(test_input)
        self.test_fn = jax.pmap(self._test_fn, axis_name="i")
        self._test = True
        # self._test_batch = next(self._test_input)  # fix test batch

    def _test_fn(self, params, state, rng, batch):
        ret = {}
        outputs, state = self.model.apply(
            params, state, rng, batch, compute_loss=False, compute_metrics=True
        )
        ret["predictions"], ret["labels"] = outputs["predictions"], outputs["labels"]
        return ret

    def _build_test_input(self) -> Generator[FeatureDict, None, None]:
        c = self.config
        test_data_path = c.dataset.data_path + "/lte_data_test.npz"
        test_data = jnp.load(test_data_path)
        batch_sizes = make_device_batch(c.test.batch_size, jax.device_count())
        test_ds = tf_data_to_generator(
            tf_data=test_data,
            is_training=False,
            batch_sizes=batch_sizes,
            buffer_size=c.dataset.buffer_size,
            threadpool_size=c.dataset.threadpool_size,
            max_intra_op_parallelism=c.dataset.max_intra_op_parallelism,
        )
        return test_ds

    def _build_dummy_input(self) -> tuple[jax.Array]:
        """Load dummy data for initializing network parameters."""

        ds = tf_data_to_generator(tf_data=self.tf_data, is_training=True)

        dummy_inputs = next(ds)

        return dummy_inputs

    # preprocess for Trainer
    def _process_data(self):
        """dataset loading."""
        c = self.config
        self.tf_data = process(
            data_path=c.dataset.data_path,
            pre_shuffle=c.dataset.pre_shuffle,
            pre_shuffle_seed=c.dataset.pre_shuffle_seed,
            num_samples=c.dataset.data_split.num_samples,
            is_split_samples=c.dataset.data_split.is_split_samples,
            split_rate=c.dataset.data_split.split_rate,
            save_path=c.dataset.data_split.save_path,
        )
