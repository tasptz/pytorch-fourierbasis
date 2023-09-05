import math
from itertools import product

import torch
from torch import Tensor, nn


class FourierBasis(nn.Module):
    def __init__(self, features: int, order: int) -> None:
        """Construct a fourier basis. Number of output features
        is (order + 1)^features.

        Args:
            features (int): Input features.
            order (int): Order of fourier basis functions.
        """
        super().__init__()
        self.coeff = nn.Parameter(
            torch.tensor(
                list(product(range(order + 1), repeat=features)),
                dtype=torch.float32,
            ).T
            * math.pi,
            requires_grad=False,
        )
        assert self.coeff.shape == (features, (order + 1) ** features)

    def forward(self, x: Tensor) -> Tensor:
        """Encode x in fourier basis.

        Args:
            x (Tensor): (batch, features)

        Returns:
            Tensor: (batch, (order + 1)^features)
        """
        return torch.cos(x @ self.coeff)
