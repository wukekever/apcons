import jax
import ml_collections
from jaxline import base_config

from deeplte.model.config import model_config


def get_config() -> ml_collections.ConfigDict:
    config = base_config.get_base_config()

    num_epochs = 5000
    train_batch_size = 4
    batch_repeat = 1
    collocation_sizes = [1024, 256]
    test_batch_size = jax.device_count() * 1

    dataset = ml_collections.ConfigDict(
        dict(
            data_path="/workspaces/DeepLTE/data",
            data_split=dict(
                num_samples=1024,
                is_split_samples=True,
                split_rate=0.875,
                save_path="",
            ),
            pre_shuffle=True,
            pre_shuffle_seed=42,
            buffer_size=5000,
            threadpool_size=48,
            max_intra_op_parallelism=1,
        )
    )
    split_config = dataset.data_split
    num_samples = split_config.num_samples
    num_train_samples = int(num_samples * split_config.split_rate)
    split_config["num_train_samples"] = num_train_samples  # 1024 - 128 = 896
    split_config["num_test_samples"] = num_samples - num_train_samples  # 128

    model = model_config()

    def steps_from_epochs(num_epochs):
        return max(
            int(batch_repeat * num_epochs * num_train_samples // train_batch_size), 1
        )

    config.training_steps = steps_from_epochs(num_epochs)

    config.experiment_kwargs = ml_collections.ConfigDict(
        dict(
            config=dict(
                dataset=dataset,
                training=dict(
                    num_epochs=num_epochs,
                    batch_size=train_batch_size,
                    collocation_sizes=collocation_sizes,
                    batch_repeat=batch_repeat,
                ),
                optimizer=dict(
                    base_lr=1e-4,
                    scale_by_batch=False,
                    schedule_type="exponential",
                    exp_decay_kwargs=dict(
                        transition_steps=steps_from_epochs(100),
                        decay_rate=0.96,
                    ),
                    optimizer="adam",
                    adam_kwargs=dict(),
                ),
                test=dict(batch_size=test_batch_size),
                model=model,
            )
        )
    )

    config.interval_type = "steps"
    config.save_checkpoint_interval = steps_from_epochs(10)
    config.log_tensors_interval = steps_from_epochs(1)
    config.log_train_data_interval = steps_from_epochs(1)
    # When True, the eval job immediately loads a checkpoint
    # runs evaluate() once, then terminates.
    config.one_off_evaluate = False
    # Seed for the RNGs (default is 42).
    config.random_seed = 42
    config.checkpoint_dir = ""
    config.restore_dir = ""
    # config.restore_dir = "/workspaces/DeepLTE/ckpts/square_full_it_2023-06-26T08:17:36/models/latest/step_224000_2023-06-26T10:59:20"

    return config
