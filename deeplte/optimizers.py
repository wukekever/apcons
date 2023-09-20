"""Optimizer module."""

import optax

BASE_BATCH_SIZE = 8


def _get_batch_scaled_lr(total_batch_size, lr, scale_by_batch=True):
    # This is the linear scaling rule in Section 5.1 of
    # https://arxiv.org/pdf/1706.02677.pdf.

    if scale_by_batch:
        lr = (lr * total_batch_size) / BASE_BATCH_SIZE

    return lr


def get_learning_rate_schedule(
    total_batch_size, steps_per_epoch, total_steps, optimizer_config
):
    """Build the learning rate schedule function."""
    base_lr = _get_batch_scaled_lr(
        total_batch_size,
        optimizer_config.base_lr,
        optimizer_config.scale_by_batch,
    )

    schedule_type = optimizer_config.schedule_type
    if schedule_type == "steps":
        boundaries = optimizer_config.step_decay_kwargs.decay_boundaries
        boundaries.sort()

        decay_rate = optimizer_config.step_decay_kwargs.decay_rate
        boundaries_and_scales = {
            int(boundary * total_steps): decay_rate for boundary in boundaries
        }
        schedule_fn = optax.piecewise_constant_schedule(
            init_value=base_lr, boundaries_and_scales=boundaries_and_scales
        )
    elif schedule_type == "exponential":
        transition_steps = optimizer_config.exp_decay_kwargs.transition_steps
        decay_rate = optimizer_config.exp_decay_kwargs.decay_rate
        schedule_fn = optax.exponential_decay(
            init_value=base_lr,
            transition_steps=transition_steps,
            decay_rate=decay_rate,
        )
    elif schedule_type == "cosine":
        warmup_steps = (
            optimizer_config.cosine_decay_kwargs.warmup_epochs * steps_per_epoch
        )
        # Batch scale the other lr values as well:
        init_value = _get_batch_scaled_lr(
            total_batch_size,
            optimizer_config.cosine_decay_kwargs.init_value,
            optimizer_config.scale_by_batch,
        )
        end_value = _get_batch_scaled_lr(
            total_batch_size,
            optimizer_config.cosine_decay_kwargs.end_value,
            optimizer_config.scale_by_batch,
        )

        schedule_fn = optax.warmup_cosine_decay_schedule(
            init_value=init_value,
            peak_value=base_lr,
            warmup_steps=warmup_steps,
            decay_steps=total_steps,
            end_value=end_value,
        )
    elif schedule_type == "constant_cosine":
        # Convert end_value to alpha, used by cosine_decay_schedule.
        alpha = optimizer_config.constant_cosine_decay_kwargs.end_value / base_lr

        # Number of steps spent in constant phase.
        constant_steps = int(
            optimizer_config.constant_cosine_decay_kwargs.constant_fraction
            * total_steps
        )
        decay_steps = total_steps - constant_steps

        constant_phase = optax.constant_schedule(value=base_lr)
        decay_phase = optax.cosine_decay_schedule(
            init_value=base_lr, decay_steps=decay_steps, alpha=alpha
        )
        schedule_fn = optax.join_schedules(
            schedules=[constant_phase, decay_phase],
            boundaries=[constant_steps],
        )
    elif schedule_type == "constant":
        schedule_fn = optax.constant_schedule(value=base_lr)
    else:
        raise ValueError(f"Unknown learning rate schedule: {schedule_type}")

    return schedule_fn


def make_optimizer(optimizer_config, lr_schedule):
    """Construct the optax optimizer with given LR schedule."""
    optax_chain = []

    if optimizer_config.optimizer == "adam":
        # See: https://arxiv.org/abs/1412.6980
        optax_chain.extend([optax.scale_by_adam(**optimizer_config.adam_kwargs)])
    elif optimizer_config.optimizer == "lamb":
        # See: https://arxiv.org/abs/1904.00962
        optax_chain.extend(
            [
                optax.scale_by_adam(**optimizer_config.lamb_kwargs),
                optax.scale_by_trust_ratio(),
            ]
        )
    else:
        raise ValueError(f"Undefined optimizer {optimizer_config.optimizer}")

    # Scale by the (negative) learning rate.
    optax_chain.extend(
        [
            optax.scale_by_schedule(lr_schedule),
            optax.scale(-1),
        ]
    )

    return optax.chain(*optax_chain)
