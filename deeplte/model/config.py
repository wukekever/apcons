"""Model config."""

import copy

import ml_collections


def model_config() -> ml_collections.ConfigDict:
    """Get the ConfigDict of DeepLTE model."""
    cfg = copy.deepcopy(CONFIG)
    return cfg


# TODO
# def model_config(name=None) -> ml_collections.ConfigDict:
#     """Get the ConfigDict of DeepLTE model."""
#     cfg = copy.deepcopy(CONFIG)
#     return cfg


# """config for kn = 1 and time range = [0, 0.5]"""
# CONFIG = ml_collections.ConfigDict(
#     {
#         "kn": 1.0,
#         "num_quads": 32,
#         "t_range": [0.0, 0.5],
#         "x_range": [0.0, 1.0],
#         "v_range": [-1.0, 1.0],
#         "lte_operator": {
#             "model_f": {
#                 "branch_mlp": {"widths": [64, 64, 64, 64, 64]},
#                 "trunk_mlp": {"widths": [64, 64, 64, 64]},
#                 "kernel_sizes": [2, 2],
#                 "num_channels": 4,
#                 "stride": 2,
#                 # "pooling": "AvgPool",
#                 # "activation": "gelu"
#                 # "num_blocks": 2,
#             },
#             "model_rho": {
#                 "branch_mlp": {"widths": [64, 64, 64, 64, 64]},
#                 "trunk_mlp": {"widths": [64, 64, 64, 64]},
#                 "kernel_sizes": [2, 2],
#                 "num_channels": 4,
#                 "stride": 2,
#                 # "pooling": "AvgPool",
#                 # "activation": "gelu"
#                 # "num_blocks": 2,
#             },
#             "model_g": {
#                 "branch_mlp": {"widths": [64, 64, 64, 64, 64]},
#                 "trunk_mlp": {"widths": [64, 64, 64, 64]},
#                 "kernel_sizes": [2, 2],
#                 "num_channels": 4,
#                 "stride": 2,
#                 # "pooling": "AvgPool",
#                 # "activation": "gelu"
#                 # "num_blocks": 2,
#             },
#             "model_r": {
#                 "branch_mlp": {"widths": [64, 64, 64, 64, 64]},
#                 "trunk_mlp": {"widths": [64, 64, 64, 64]},
#                 "kernel_sizes": [2, 2],
#                 "num_channels": 4,
#                 "stride": 2,
#                 # "pooling": "AvgPool",
#                 # "activation": "gelu"
#                 # "num_blocks": 2,
#             },
#             "model_j": {
#                 "branch_mlp": {"widths": [64, 64, 64, 64, 64]},
#                 "trunk_mlp": {"widths": [64, 64, 64, 64]},
#                 "kernel_sizes": [2, 2],
#                 "num_channels": 4,
#                 "stride": 2,
#                 # "pooling": "AvgPool",
#                 # "activation": "gelu"
#                 # "num_blocks": 2,
#             },
#         },
#         "regularizers": [1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
#     }
# )


"""config for kn = 1e-4 and time range = [0, 0.1]"""
CONFIG = ml_collections.ConfigDict(
    {
        "kn": 0.0001,
        "num_quads": 32,
        "t_range": [0.0, 0.1],
        "x_range": [0.0, 1.0],
        "v_range": [-1.0, 1.0],
        "lte_operator": {
            "model_f": {
                "branch_mlp": {"widths": [64, 64, 64, 64, 64]},
                "trunk_mlp": {"widths": [64, 64, 64, 64]},
                "kernel_sizes": [2, 2],
                "num_channels": 4,
                "stride": [2, 2],
                # "pooling": "AvgPool",
                # "activation": "gelu"
                # "num_blocks": 2,
            },
            "model_rho": {
                "branch_mlp": {"widths": [64, 64, 64, 64, 64]},
                "trunk_mlp": {"widths": [64, 64, 64, 64]},
                "kernel_sizes": [2, 2],
                "num_channels": 4,
                "stride": [2, 2],
                # "pooling": "AvgPool",
                # "activation": "gelu"
                # "num_blocks": 2,
            },
            "model_g": {
                "branch_mlp": {"widths": [64, 64, 64, 64, 64]},
                "trunk_mlp": {"widths": [64, 64, 64, 64]},
                "kernel_sizes": [2, 2],
                "num_channels": 4,
                "stride": [2, 2],
                # "pooling": "AvgPool",
                # "activation": "gelu"
                # "num_blocks": 2,
            },
            "model_r": {
                "branch_mlp": {"widths": [64, 64, 64, 64, 64]},
                "trunk_mlp": {"widths": [64, 64, 64, 64]},
                "kernel_sizes": [2, 2],
                "num_channels": 4,
                "stride": [2, 2],
                # "pooling": "AvgPool",
                # "activation": "gelu"
                # "num_blocks": 2,
            },
            "model_j": {
                "branch_mlp": {"widths": [64, 64, 64, 64, 64]},
                "trunk_mlp": {"widths": [64, 64, 64, 64]},
                "kernel_sizes": [2, 2],
                "num_channels": 4,
                "stride": [2, 2],
                # "pooling": "AvgPool",
                # "activation": "gelu"
                # "num_blocks": 2,
            },
        },
        "regularizers": [1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
    }
)
