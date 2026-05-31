"""Trading strategies + simulation harnesses."""

from .delta_hedge import (
    DeltaHedgeConfig,
    DeltaHedgeResult,
    simulate_short_straddle_hedged,
    summarise_runs,
)

__all__ = [
    "DeltaHedgeConfig",
    "DeltaHedgeResult",
    "simulate_short_straddle_hedged",
    "summarise_runs",
]
