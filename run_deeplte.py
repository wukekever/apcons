"""Train."""

import functools
import os
import pathlib

from absl import app, flags, logging
from jaxline import platform

from deeplte.checkpoint import (
    restore_state_to_in_memory_checkpointer,
    save_state_from_in_memory_checkpointer,
    setup_signals,
)
from deeplte.train import Trainer

FLAGS = flags.FLAGS

flags.DEFINE_string("data_path", None, "data path")


def main(experiment_class, argv):
    # make figures/ and ckpts/ directories
    mkdir(file_dirs=["./figure", "./ckpts"])

    write_data_path(FLAGS.config.experiment_kwargs.config.dataset)

    # Maybe restore a model.
    restore_dir = FLAGS.config.restore_dir

    if restore_dir:
        restore_state_to_in_memory_checkpointer(restore_dir)

    # Maybe save a model.
    save_dir = os.path.join(FLAGS.config.checkpoint_dir, "models")

    if FLAGS.config.one_off_evaluate:
        save_model_fn = (
            lambda: None
        )  # noqa: E731  # No need to save checkpoint in this case.
    else:
        save_model_fn = functools.partial(
            save_state_from_in_memory_checkpointer, save_dir, experiment_class
        )
    setup_signals(save_model_fn)  # Save on Ctrl+C (continue) or Ctrl+\ (exit).

    if FLAGS.jaxline_mode.startswith("train"):
        if not pathlib.Path(FLAGS.config.checkpoint_dir).exists():
            pathlib.Path(FLAGS.config.checkpoint_dir).mkdir()
        logging.get_absl_handler().use_absl_log_file(
            "train", FLAGS.config.checkpoint_dir
        )

    try:
        platform.main(experiment_class, argv)
    finally:
        save_model_fn()  # Save at the end of training or in case of exception.


def write_data_path(config):
    config.data_path = FLAGS.data_path


def mkdir(file_dirs):
    for file_dir in file_dirs:
        isExists = os.path.exists(file_dir)
        if not isExists:
            os.makedirs(file_dir)


if __name__ == "__main__":
    flags.mark_flag_as_required("config")
    app.run(functools.partial(main, Trainer))
