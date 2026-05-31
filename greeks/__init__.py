"""Greek computation: analytical (BS), finite-difference, and pathwise (MC)."""

from .finite_difference import fd_greeks

__all__ = ["fd_greeks"]
