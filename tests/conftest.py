"""Shared test setup: headless backends, and no figure left open."""

import os
import sys

import dascore as dc
import pytest

# Set before any backend is imported, so neither tries to open a window.
os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("PYVISTA_OFF_SCREEN", "true")


@pytest.fixture(autouse=True)
def _close_figures():
    """Close any matplotlib figure a test opened."""
    yield
    if "matplotlib.pyplot" in sys.modules:
        sys.modules["matplotlib.pyplot"].close("all")


@pytest.fixture()
def patch():
    """DASCore's example patch: dims (distance, time), absolute times."""
    return dc.get_example_patch()


@pytest.fixture()
def random_patch_with_xyz():
    """DASCore's example patch: dims (distance, time), absolute times."""
    return dc.get_example_patch("random_patch_with_xyz")


@pytest.fixture()
def spool():
    """DASCore's example spool."""
    return dc.get_example_spool()
