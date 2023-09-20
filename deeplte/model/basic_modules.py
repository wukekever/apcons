"""Core modules including branch net and trunck net."""
from typing import Optional

import haiku as hk
import jax.numpy as jnp
from ml_collections import ConfigDict

# from deeplte.model.networks import MLP as NN
from deeplte.model.networks import ConvNet
from deeplte.model.networks import modified_MLP_LN as NN

# from deeplte.model.networks import modified_MLP as NN


class BranchNet(hk.Module):
    def __init__(self, config: ConfigDict, name: Optional[str] = "branch_net"):
        super().__init__(name=name)

        self.config = config

    def __call__(self, a: jnp.ndarray) -> jnp.ndarray:
        c = self.config
        # CONV
        outputs = ConvNet(
            num_channels=c.num_channels, kernel_sizes=c.kernel_sizes, stride=c.stride
        )(a)
        # # MLP
        outputs = NN(
            output_sizes=c.branch_mlp.widths,
            squeeze_output=False,
            name="branch_net_mlp",
        )(outputs)
        return outputs


class TrunkNet(hk.Module):
    def __init__(self, config: ConfigDict, name: Optional[str] = "trunk_net"):
        super().__init__(name=name)

        self.config = config

    def __call__(self, y: jnp.ndarray) -> jnp.ndarray:
        c = self.config
        # MLP
        outputs = NN(
            output_sizes=c.trunk_mlp.widths, squeeze_output=False, name="trunk_net_mlp"
        )(y)
        return outputs


def mean_squared_loss_fn(inputs, axis=None):
    return jnp.mean(jnp.square(inputs), axis=axis)
