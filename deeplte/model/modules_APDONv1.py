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

        # # uniform grids
        # weights = 2.0 / self.num_quads
        # self.quads = np.linspace(
        #     -1.0 + weights / 2.0, 1.0 - weights / 2.0, self.num_quads
        # )
        # self.weights = weights * np.ones(self.num_quads)

        self.xl, self.xr = self.config.x_range

        self.fn_left, self.fn_right = 1.0, 0.5
        # self.fn_left, self.fn_right = 0.0, 0.0
        self.fn_init = None

        self.regularizers = self.config.regularizers

        self._batch_axes = ((None, 0, 0), (0, None, None))

    # (a, t_interior, x_interior, v_interior)
    def residual_fn(self, *args):
        drho_dt = hk.grad(self.rho_fn, argnums=1)
        drho_dx = hk.grad(self.rho_fn, argnums=2)
        dg_dt = hk.grad(self.g_fn, argnums=1)
        eqn_res = {}
        v = args[-1]
        aver_vg_x = self.aver_vg_x_fn(*args[:-1])
        micro_res = (
            self.kn**2 * dg_dt(*args)
            + self.kn * (self.vg_x_fn(*args) - aver_vg_x)
            + v * drho_dx(*args[:-1])
            - (0.0 - self.g_fn(*args))
        )
        macro_res = drho_dt(*args[:-1]) + aver_vg_x
        eqn_res.update({"micro": micro_res})
        eqn_res.update({"macro": macro_res})
        return eqn_res

    def average_fn(self, func, *args):
        integral_fn = integrate.quad(
            fun=func, quadratures=[self.quads[:, None], self.weights], argnum=3
        )
        return 0.5 * integral_fn(*args)

    def lte_fn(self, a, t, x, v):
        # a: (2048,), t: (1,), x: (1,), v: (1,)
        lte_sol = self.rho_fn(a, t, x) + self.kn * self.g_fn(a, t, x, v)
        return lte_sol

    def rho_fn(self, a, t, x):
        # a: (2048,), t: (1,), x: (1,)
        y = jnp.concatenate(arrays=(t, x), axis=-1)
        # Get nn output of branch and trunck net.
        cfg = self.config.lte_operator.model_rho
        branch_outputs = BranchNet(cfg)(a)
        trunk_outputs = TrunkNet(cfg)(y)
        rho = self.positive_fn(jnp.sum(branch_outputs * trunk_outputs))
        return rho

    def aver_vg_x_fn(self, a, t, x):
        return self.average_fn(self.vg_x_fn, a, t, x)

    def vg_x_fn(self, a, t, x, v):
        dg_dx = hk.grad(self.g_fn, argnums=2)
        vg_x = v * dg_dx(a, t, x, v)
        return vg_x

    def g_fn(self, a, t, x, v):
        # <g> = <g_hat - <g_hat>> = <g_hat> - <g_hat> = 0
        g = self.g_hat_fn(a, t, x, v) - self.average_fn(self.g_hat_fn, a, t, x)
        return g

    def g_hat_fn(self, a, t, x, v):
        # a: (2048,), t: (1,), x: (1,), v: (1,)
        y = jnp.concatenate(arrays=(t, x, v), axis=-1)
        # Get nn output of branch and trunck net.
        cfg = self.config.lte_operator.model_g
        branch_outputs = BranchNet(cfg)(a)
        trunk_outputs = TrunkNet(cfg)(y)
        g_hat = jnp.sum(branch_outputs * trunk_outputs)
        return g_hat

    def positive_fn(self, inputs):
        # return jnp.exp(inputs)
        # return jnp.log(1.0 + jnp.exp(inputs))
        return inputs

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

        self.batched_rho_fn = hk.vmap(
            hk.vmap(
                self.rho_fn,
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

            eqn_residual = batched_residual_fn(a, t_interior, x_interior, v_interior)
            macro_residual_loss = jnp.squeeze(
                mean_squared_loss_fn(eqn_residual["macro"])
            )
            micro_residual_loss = jnp.squeeze(
                mean_squared_loss_fn(eqn_residual["micro"])
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
                "macro_eqn": macro_residual_loss,
                "micro_eqn": micro_residual_loss,
                "boundary": boundary_loss,
                "initial": initial_loss,
            }

            total_loss = (
                self.regularizers[0] * macro_residual_loss
                + self.regularizers[1] * micro_residual_loss
                + self.regularizers[2] * boundary_loss
                + self.regularizers[3] * initial_loss
            )

        if compute_metrics:
            density_labels = batch["func"]["label"]
            density_label_t, density_label_x = (
                batch["grids"]["density"]["t"],
                batch["grids"]["density"]["x"],
            )

            density_predictions = self.batched_rho_fn(
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
