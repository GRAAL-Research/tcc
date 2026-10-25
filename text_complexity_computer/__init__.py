from importlib.metadata import PackageNotFoundError, version

from .text_complexity_computer import METRICS, SCALERS, TextComplexityComputer

try:
    __version__ = version("text_complexity_computer")
except PackageNotFoundError:  # pragma: no cover - source checkout without installation
    __version__ = "unknown"

__all__ = ["METRICS", "SCALERS", "TextComplexityComputer", "__version__"]
