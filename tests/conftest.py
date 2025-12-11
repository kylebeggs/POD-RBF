"""Pytest configuration and shared fixtures."""

import jax

# Ensure float64 is enabled for all tests
jax.config.update("jax_enable_x64", True)
