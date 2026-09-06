"""Large streaming phone Student for X-VC2."""

from typing import Any

__all__ = [
    "ExperimentConfig",
    "ModelConfig",
    "StreamingPhoneEncoder",
    "StreamingState",
    "load_config",
    "parameter_breakdown",
]


def __getattr__(name: str) -> Any:
    if name in {"ExperimentConfig", "ModelConfig", "load_config"}:
        from . import config

        return getattr(config, name)
    if name in {"StreamingPhoneEncoder", "StreamingState", "parameter_breakdown"}:
        from . import model

        return getattr(model, name)
    raise AttributeError(name)
