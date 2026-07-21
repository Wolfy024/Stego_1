"""Backward-compatible import for the refactored model."""

from stego.config import ModelConfig
from stego.models import StegoGAN


class AFASNet(StegoGAN):
    """Compatibility name retained for existing checkpoints and examples."""

    def __init__(self, config: ModelConfig | None = None) -> None:
        super().__init__(config)
