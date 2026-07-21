"""Energy-preserving routing across the four 2D Haar frequency bands."""

from __future__ import annotations

import torch
from torch import Tensor, nn


class EnergyPreservingHaarBandRouter(nn.Module):
    """Rotate Haar bands with a sample-adaptive orthogonal matrix.

    Four RMS energy descriptors (LL, HL, LH, and HH) predict the six
    independent entries of a 4x4 skew-symmetric matrix. Its Cayley transform
    is orthogonal, so routing redistributes information between bands without
    changing the L2 energy of the routed tensor.

    The final predictor layer is zero-initialized. Consequently, the skew
    matrix starts at zero, the Cayley transform starts at the identity, and a
    newly inserted router preserves an existing model's behavior exactly.
    """

    _SKEW_INDICES = ((0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3))

    def __init__(self, hidden_features: int = 8, max_skew: float = 1.0) -> None:
        super().__init__()
        if hidden_features < 1:
            raise ValueError("hidden_features must be positive")
        if max_skew <= 0.0:
            raise ValueError("max_skew must be positive")

        self.max_skew = float(max_skew)
        self.skew_predictor = nn.Sequential(
            nn.Linear(4, hidden_features),
            nn.SiLU(),
            nn.Linear(hidden_features, len(self._SKEW_INDICES)),
        )
        final_layer = self.skew_predictor[-1]
        nn.init.zeros_(final_layer.weight)
        nn.init.zeros_(final_layer.bias)

    @staticmethod
    def _as_bands(values: Tensor) -> Tensor:
        if values.ndim != 4:
            raise ValueError(f"expected a [B, C, H, W] tensor, got shape {tuple(values.shape)}")
        if values.shape[1] % 4:
            raise ValueError("the channel count must be divisible by four Haar bands")
        batch, channels, height, width = values.shape
        return values.reshape(batch, 4, channels // 4, height, width)

    def band_descriptors(self, values: Tensor) -> Tensor:
        """Return one RMS energy descriptor for each Haar band."""

        return self._band_rms(self._as_bands(values))

    @staticmethod
    def _band_rms(bands: Tensor) -> Tensor:
        """Compute stable RMS descriptors without a singular gradient at zero."""

        energy = bands.float().square().mean(dim=(2, 3, 4))
        return (energy + 1e-12).sqrt().to(dtype=bands.dtype)

    def routing_matrix_from_descriptors(self, descriptors: Tensor) -> Tensor:
        """Construct one orthogonal 4x4 routing matrix per sample."""

        if descriptors.ndim != 2 or descriptors.shape[1] != 4:
            raise ValueError(
                "band descriptors must have shape [B, 4], "
                f"got {tuple(descriptors.shape)}"
            )

        parameters = self.max_skew * torch.tanh(self.skew_predictor(descriptors))
        # Linear solves can be unsupported or inaccurate in reduced precision.
        # Keep this tiny 4x4 operation in float32 while preserving autograd.
        solve_dtype = (
            torch.float32
            if parameters.dtype in (torch.float16, torch.bfloat16)
            else parameters.dtype
        )
        parameters = parameters.to(dtype=solve_dtype)
        skew = parameters.new_zeros(parameters.shape[0], 4, 4)
        for index, (row, column) in enumerate(self._SKEW_INDICES):
            skew[:, row, column] = parameters[:, index]
            skew[:, column, row] = -parameters[:, index]

        identity = torch.eye(4, device=skew.device, dtype=skew.dtype).expand_as(skew)
        # For skew A, (I + A)^-1 (I - A) is the Cayley transform and is
        # orthogonal. Both factors are polynomials in A, so their order agrees
        # with the equivalent (I - A)(I + A)^-1 expression.
        routing = torch.linalg.solve(identity + skew, identity - skew)
        return routing.to(dtype=descriptors.dtype)

    def routing_matrix(self, values: Tensor) -> Tensor:
        """Return the sample-adaptive routing matrices for a Haar tensor."""

        return self.routing_matrix_from_descriptors(self.band_descriptors(values))

    def forward(self, values: Tensor) -> Tensor:
        bands = self._as_bands(values)
        routing = self.routing_matrix_from_descriptors(self._band_rms(bands))
        routed = torch.einsum("bij,bjchw->bichw", routing, bands)
        return routed.reshape_as(values)
