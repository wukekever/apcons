"""Core modules including branch net and trunck net."""
import dataclasses
from typing import Optional

import haiku as hk
import jax.numpy as jnp
import numpy as np
from ml_collections import ConfigDict

from deeplte.model import integrate
from deeplte.model.basic_modules import BranchNet, TrunkNet, mean_squared_loss_fn


@dataclasses.dataclass
class DeepLTE(hk.Module):
    """Deep LTE model."""

    config: ConfigDict
    name: Optional[str] = "deeplte"

    def __init__(self, config, name="deeplte"):
        super().__init__(name)

        self.config = config
        self.kn = self.config.kn
        self.num_quads = self.config.num_quads
        self.quads, self.weights = np.polynomial.legendre.leggauss(self.num_quads)
        self.v_range = self.config.v_range
        self.quads = (
            0.5 * (1.0 + self.quads) * (self.v_range[-1] - self.v_range[0])
            + self.v_range[0]
        )

        self.xl, self.xr = self.config.x_range

        self.fn_left, self.fn_right = 1.0, 0.5
        self.fn_init = None

        self.regularizers = self.config.regularizers

        self._batch_axes = ((None, 0, 0), (0, None, None))

    # (a, t_interior, x_interior, v_interior)
    def residual_fn(self, *args):
        df_dt = hk.grad(self.lte_fn, argnums=1)
        df_dx = hk.grad(self.lte_fn, argnums=2)
        v = args[-1]
        transport = self.kn * df_dt(*args) + v @ df_dx(*args)
        collision = 1.0 / self.kn * (self.average_fn(*args[:-1]) - self.lte_fn(*args))
        residual = transport - collision
        return residual

    def average_fn(self, *args):
        integral_fn = integrate.quad(
            fun=self.lte_fn, quadratures=[self.quads[:, None], self.weights], argnum=3
        )
        return 0.5 * integral_fn(*args)

    def lte_fn(self, a, t, x, v):
        # a: (2048,), t: (1,), x: (1,), v: (1,)
        y = jnp.concatenate(arrays=(t, x, v), axis=-1)
        # Get nn output of branch and trunck net.
        cfg = self.config.lte_operator.model_f
        branch_outputs = BranchNet(cfg)(a)
        trunk_outputs = TrunkNet(cfg)(y)
        lte_sol = self.positive_fn(jnp.sum(branch_outputs * trunk_outputs))
        return lte_sol

    def positive_fn(self, inputs):
        # return jnp.exp(inputs)
        return jnp.log(1.0 + jnp.exp(inputs))

    def __call__(self, batch, compute_loss=False, compute_metrics=False):
        ret = {}

        """"batch shape:
                    function - a: (batch_function, num_x*num_v)
                    interior - (t, x, v): (batch_interior, 1) * 3
                    boundary - (t, v): (batch_boundary, 1) * 2
                    initial - (x, v): (batch_initial, 1) * 2
        """

        a = batch["func"]["a"]
        t_interior, x_interior, v_interior = (
            batch["interior"]["t"],
            batch["interior"]["x"],
            batch["interior"]["v"],
        )
        t_boundary, v_boundary = batch["boundary"]["t"], batch["boundary"]["v"]
        x_boundary_left, x_boundary_right = self.xl * jnp.ones(
            t_boundary.shape
        ), self.xr * jnp.ones(t_boundary.shape)
        x_initial, v_initial = (
            batch["grids"]["initial"]["x"],
            batch["grids"]["initial"]["v"],
        )
        t_initial = jnp.zeros(x_initial.shape)

        self.batched_average_fn = hk.vmap(
            hk.vmap(
                self.average_fn,
                in_axes=self._batch_axes[0],
                split_rng=(not hk.running_init()),
            ),
            in_axes=self._batch_axes[-1],
            split_rng=(not hk.running_init()),
        )

        if compute_loss:
            self.fn_init = a.reshape(
                -1, a.shape[-1] * a.shape[-2]
            )  # a.shape[-1] * a.shape[-2]
            batch_axes = ((None, 0, 0, 0), (0, None, None, None))

            batched_residual_fn = hk.vmap(
                hk.vmap(
                    self.residual_fn,
                    in_axes=batch_axes[0],
                    split_rng=(not hk.running_init()),
                ),
                in_axes=batch_axes[-1],
                split_rng=(not hk.running_init()),
            )

            batched_lte_fn = hk.vmap(
                hk.vmap(
                    self.lte_fn,
                    in_axes=batch_axes[0],
                    split_rng=(not hk.running_init()),
                ),
                in_axes=batch_axes[-1],
                split_rng=(not hk.running_init()),
            )

            residual_loss = jnp.squeeze(
                mean_squared_loss_fn(
                    batched_residual_fn(a, t_interior, x_interior, v_interior)
                )
            )
            boundary_loss = jnp.squeeze(
                mean_squared_loss_fn(
                    batched_lte_fn(a, t_boundary, x_boundary_left, v_boundary)
                    - self.fn_left
                )
            ) + jnp.squeeze(
                mean_squared_loss_fn(
                    batched_lte_fn(a, t_boundary, x_boundary_right, -v_boundary)
                    - self.fn_right
                )
            )
            initial_loss = jnp.squeeze(
                mean_squared_loss_fn(
                    batched_lte_fn(a, t_initial, x_initial, v_initial) - self.fn_init
                )
            )

            ret["loss"] = {
                "interior": residual_loss,
                "boundary": boundary_loss,
                "initial": initial_loss,
            }

            total_loss = (
                self.regularizers[0] * residual_loss
                + self.regularizers[1] * boundary_loss
                + self.regularizers[2] * initial_loss
            )

        if compute_metrics:
            density_labels = batch["func"]["label"]
            density_label_t, density_label_x = (
                batch["grids"]["density"]["t"],
                batch["grids"]["density"]["x"],
            )

            density_predictions = self.batched_average_fn(
                a, density_label_t, density_label_x
            )
            ret["labels"] = density_labels
            ret["predictions"] = density_predictions

            MSE = mean_squared_loss_fn(
                inputs=density_predictions - density_labels, axis=-1
            )
            RMSE = MSE / jnp.mean(density_labels**2)
            ret["metrics"] = {"mse": MSE, "rmse": RMSE}

        if compute_loss:
            return total_loss, ret

        return ret
