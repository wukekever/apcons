"""Checkpointing."""

import datetime
import os
import pathlib
import signal
import threading

import dill
import numpy as np
from absl import flags, logging
from jaxline import experiment
from jaxline import utils as jl_utils

from deeplte.train import Trainer
from deeplte.utils import to_flat_dict

FLAGS = flags.FLAGS


def _get_step_date_label(global_step):
    # Date removing microseconds.
    date_str = datetime.datetime.now().isoformat().split(".")[0]
    return f"step_{global_step}_{date_str}"


def restore_state_to_in_memory_checkpointer(restore_path):
    """Initializes experiment state from a checkpoint."""
    if not isinstance(restore_path, pathlib.Path):
        restore_path = pathlib.Path(restore_path)

    # Load pretrained experiment state.
    python_state_path = restore_path / "checkpoint.dill"
    with open(python_state_path, "rb") as f:
        pretrained_state = dill.load(f)
    logging.info("Restored checkpoint from %s", python_state_path)

    # Assign state to a dummy experiment instance for the in-memory checkpointer,
    # broadcasting to devices.
    # dummy_experiment = Trainer(
    #     mode="train_eval_multithreaded",
    #     init_rng=0,
    #     config=FLAGS.config.experiment_kwargs.config,
    # )
    dummy_experiment = Trainer(
        mode="train",
        init_rng=0,
        config=FLAGS.config.experiment_kwargs.config,
    )
    for attribute, key in Trainer.CHECKPOINT_ATTRS.items():
        setattr(
            dummy_experiment,
            attribute,
            jl_utils.bcast_local_devices(pretrained_state[key]),
        )

    jaxline_state = dict(
        global_step=pretrained_state["global_step"],
        experiment_module=dummy_experiment,
    )
    snapshot = jl_utils.SnapshotNT(0, jaxline_state)

    # Finally, seed the jaxline `utils.InMemoryCheckpointer` global dict.
    jl_utils.GLOBAL_CHECKPOINT_DICT["latest"] = jl_utils.CheckpointNT(
        threading.local(), [snapshot]
    )


def save_state_from_in_memory_checkpointer(
    save_path, experiment_class: experiment.AbstractExperiment
):
    """Saves experiment state to a checkpoint."""
    if not isinstance(save_path, pathlib.Path):
        save_path = pathlib.Path(save_path)

    # Serialize config as json
    logging.info("Saving config.")
    config_path = save_path.parent / "config.json"
    with open(config_path, "w", encoding="utf-8") as f:
        f.write(FLAGS.config.to_json_best_effort(indent=2))

    logging.info("Saving model.")
    for checkpoint_name, checkpoint in jl_utils.GLOBAL_CHECKPOINT_DICT.items():
        if not checkpoint.history:
            logging.info('Nothing to save in "%s"', checkpoint_name)
            continue

        pickle_nest = checkpoint.history[-1].pickle_nest
        global_step = pickle_nest["global_step"]

        state_dict = {"global_step": global_step}
        for attribute, key in experiment_class.CHECKPOINT_ATTRS.items():
            state_dict[key] = jl_utils.get_first(
                getattr(pickle_nest["experiment_module"], attribute)
            )

        # Saving directory
        save_dir = save_path / checkpoint_name / _get_step_date_label(global_step)

        # Save params and states in a dill file
        python_state_path = save_dir / "checkpoint.dill"
        save_dir.mkdir(parents=True, exist_ok=True)
        with open(python_state_path, "wb") as f:
            dill.dump(state_dict, f)

        # Save flat params separately
        numpy_params_path = save_dir / "params.npz"
        flat_np_params = to_flat_dict(state_dict["params"])
        np.savez(numpy_params_path, **flat_np_params)

        logging.info(
            'Saved "%s" checkpoint and flat numpy params under %s',
            checkpoint_name,
            save_dir,
        )


def setup_signals(save_model_fn):
    """Sets up a signal for model saving."""

    # Save a model on Ctrl+C.
    def sigint_handler(unused_sig, unused_frame):
        # Ideally, rather than saving immediately, we would then "wait" for a good
        # time to save. In practice this reads from an in-memory checkpoint that
        # only saves every 30 seconds or so, so chances of race conditions are very
        # small.
        save_model_fn()
        logging.info(r"Use `Ctrl+\` to save and exit.")

    # Exit on `Ctrl+\`, saving a model.
    prev_sigquit_handler = signal.getsignal(signal.SIGQUIT)

    def sigquit_handler(unused_sig, unused_frame):
        # Restore previous handler early, just in case something goes wrong in the
        # next lines, so it is possible to press again and exit.
        signal.signal(signal.SIGQUIT, prev_sigquit_handler)
        save_model_fn()
        logging.info(r"Exiting on `Ctrl+\`")

        # Re-raise for clean exit.
        os.kill(os.getpid(), signal.SIGQUIT)

    signal.signal(signal.SIGINT, sigint_handler)
    signal.signal(signal.SIGQUIT, sigquit_handler)
