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
        drho_dt = hk.grad(self.rho_fn, argnums=1)
        dr_dt = hk.grad(self.r_fn, argnums=1)
        dr_dx = hk.grad(self.r_fn, argnums=2)
        dj_dt = hk.grad(self.j_fn, argnums=1)
        eqn_res = {}
        v = args[-1]
        aver_vj_x = self.aver_vj_x_fn(*args[:-1])
        rho = self.rho_fn(*args[:-1])
        parity1_res = self.kn**2 * (dr_dt(*args) + self.vj_x_fn(*args)) - (
            rho - self.r_fn(*args)
        )
        parity2_res = (
            self.kn**2 * dj_dt(*args) + v * dr_dx(*args) - (0.0 - self.j_fn(*args))
        )
        claw_res = drho_dt(*args[:-1]) + aver_vj_x
        contriant_res = self.rho_fn(*args[:-1]) - self.average_fn(self.r_fn, *args[:-1])

        eqn_res.update({"parity_1": parity1_res})
        eqn_res.update({"parity_2": parity2_res})
        eqn_res.update({"claw": claw_res})
        eqn_res.update({"constraint": contriant_res})
        return eqn_res

    def average_fn(self, func, *args):
        integral_fn = integrate.quad(
            fun=func, quadratures=[self.quads[:, None], self.weights], argnum=3
        )
        return 0.5 * integral_fn(*args)

    def lte_fn(self, a, t, x, v):
        # a: (2048,), t: (1,), x: (1,), v: (1,)
        lte_sol = self.r_fn(a, t, x, v) + self.kn * self.j_fn(a, t, x, v)
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

    def aver_vj_x_fn(self, a, t, x):
        return self.average_fn(self.vj_x_fn, a, t, x)

    def vj_x_fn(self, a, t, x, v):
        dj_dx = hk.grad(self.j_fn, argnums=2)
        vj_x = v * dj_dx(a, t, x, v)
        return vj_x

    def r_fn(self, a, t, x, v):
        r = self.positive_fn(
            0.5 * (self.r_hat_fn(a, t, x, v) + self.r_hat_fn(a, t, x, -v))
        )
        return r

    def r_hat_fn(self, a, t, x, v):
        # a: (2048,), t: (1,), x: (1,), v: (1,)
        y = jnp.concatenate(arrays=(t, x, v), axis=-1)
        # Get nn output of branch and trunck net.
        cfg = self.config.lte_operator.model_r
        branch_outputs = BranchNet(cfg)(a)
        trunk_outputs = TrunkNet(cfg)(y)
        r_hat = jnp.sum(branch_outputs * trunk_outputs)
        return r_hat

    def j_fn(self, a, t, x, v):
        j = self.j_hat_fn(a, t, x, v) - self.j_hat_fn(a, t, x, -v)
        return j

    def j_hat_fn(self, a, t, x, v):
        # a: (2048,), t: (1,), x: (1,), v: (1,)
        y = jnp.concatenate(arrays=(t, x, v), axis=-1)
        # Get nn output of branch and trunck net.
        cfg = self.config.lte_operator.model_j
        branch_outputs = BranchNet(cfg)(a)
        trunk_outputs = TrunkNet(cfg)(y)
        j_hat = jnp.sum(branch_outputs * trunk_outputs)
        return j_hat

    def positive_fn(self, inputs):
        # return jnp.exp(-inputs)
        # return jnp.log(1.0 + jnp.exp(inputs))
        return inputs

    def __call__(self, batch, compute_loss=False, compute_metrics=False):
        ret = {}

        """"batch shape:
                    function - a: (batch_function, num_x, num_v)
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
        _x_initial = x_initial[:: a.shape[-1]]
        _t_initial = jnp.zeros(_x_initial.shape)

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
            self.rho_init = jnp.mean(a, axis=-1, keepdims=False).reshape(
                -1, a.shape[-2]
            )
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
            parity1_residual_loss = jnp.squeeze(
                mean_squared_loss_fn(eqn_residual["parity_1"])
            )
            parity2_residual_loss = jnp.squeeze(
                mean_squared_loss_fn(eqn_residual["parity_2"])
            )
            claw_residual_loss = jnp.squeeze(mean_squared_loss_fn(eqn_residual["claw"]))
            constraint_residual_loss = jnp.squeeze(
                mean_squared_loss_fn(eqn_residual["constraint"])
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
            ) + jnp.squeeze(
                mean_squared_loss_fn(
                    self.batched_rho_fn(a, _t_initial, _x_initial) - self.rho_init
                )
            )

            ret["loss"] = {
                "parity1_eqn": parity1_residual_loss,
                "parity2_eqn": parity2_residual_loss,
                "claw_eqn": claw_residual_loss,
                "constriant_eqn": constraint_residual_loss,
                "boundary": boundary_loss,
                "initial": initial_loss,
            }

            total_loss = (
                self.regularizers[0] * parity1_residual_loss
                + self.regularizers[1] * parity2_residual_loss
                + self.regularizers[2] * claw_residual_loss
                + self.regularizers[3] * constraint_residual_loss
                + self.regularizers[4] * boundary_loss
                + self.regularizers[5] * initial_loss
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
