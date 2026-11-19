"""
The optional backends dasviz draws with, and how a method asks for one.

Every namespace is registered whatever extras are installed, since entry
points cannot depend on them. So loading a namespace imports no backend: a
method imports its backend when it is called, and if the backend is missing
it says which extra to install.
"""

from __future__ import annotations

import functools
import importlib.util
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from types import MappingProxyType, ModuleType
from typing import ParamSpec, TypeVar

from dascore.exceptions import MissingOptionalDependencyError
from dascore.utils.misc import optional_import

P = ParamSpec("P")
R = TypeVar("R")


@dataclass(frozen=True)
class Backend:
    """
    An optional package dasviz can draw with.

    Parameters
    ----------
    module
        The name to import.
    extra
        The dasviz extra which installs it, as in ``pip install dasviz[extra]``.
    aliases
        Other names of the same extra.
    """

    module: str
    extra: str
    aliases: tuple[str, ...] = ()


# The one list of backends. pyproject.toml declares an extra, and each
# alias, for every entry; tests/test_deps.py checks the two agree.
BACKENDS: Mapping[str, Backend] = MappingProxyType(
    {
        "matplotlib": Backend("matplotlib", "matplotlib", aliases=("mpl",)),
        "pyvista": Backend("pyvista", "pyvista", aliases=("pv",)),
    }
)


def _get_backend(key: str) -> Backend:
    """Return the backend named key, or raise naming the known ones."""
    try:
        return BACKENDS[key]
    except KeyError:
        msg = f"Unknown dasviz backend {key!r}; known backends: {sorted(BACKENDS)}."
        raise ValueError(msg) from None


def _is_missing(module: str) -> bool:
    """Return True if the package providing module is not installed at all."""
    top = module.partition(".")[0]
    try:
        return importlib.util.find_spec(top) is None
    except (ImportError, ValueError):
        return True


def import_backend(key: str, required_for: str = "this dasviz method") -> ModuleType:
    """
    Import a backend, or say which dasviz extra installs it.

    Parameters
    ----------
    key
        The backend's key in `BACKENDS`, e.g. ``"pyvista"``.
    required_for
        What needs the backend, for the error message.

    Raises
    ------
    MissingOptionalDependencyError
        If the backend is not installed. The message gives the extra to
        install, and so does the exception's ``install_name``. A backend
        which is installed but fails to import raises DASCore's error as is,
        since installing it again would not help.
    """
    backend = _get_backend(key)
    try:
        return optional_import(backend.module)
    except MissingOptionalDependencyError as ex:
        if not _is_missing(backend.module):
            raise
        install_name = f"dasviz[{backend.extra}]"
        msg = (
            f"{required_for} requires {backend.module}, which is not installed. "
            f'Install it with: pip install "{install_name}"'
        )
        # Set rather than passed: older DASCore releases' error takes no
        # install_name argument.
        error = MissingOptionalDependencyError(msg)
        error.install_name = install_name
        raise error from ex


def requires(*keys: str) -> Callable[[Callable[P, R]], Callable[P, R]]:
    """
    Mark a function as needing backends, and check for them when it is called.

    The function should still import the backend itself, inside its body;
    this checks first, so a missing one gives the extra to install. The keys
    are kept on the function as ``__dasviz_requires__``.

    Parameters
    ----------
    *keys
        Keys of `BACKENDS`.

    Examples
    --------
    >>> from dasviz import requires
    >>> @requires("matplotlib")
    ... def draw(patch):
    ...     import matplotlib.pyplot as plt
    ...     return plt.gca()
    >>> draw.__dasviz_requires__
    ('matplotlib',)
    """
    for key in keys:
        _get_backend(key)  # refuse an unknown key when the module is loaded

    def decorator(func: Callable[P, R]) -> Callable[P, R]:
        required_for = f"`{func.__name__}`"

        @functools.wraps(func)
        def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
            for key in keys:
                import_backend(key, required_for=required_for)
            return func(*args, **kwargs)

        wrapper.__dasviz_requires__ = keys  # type: ignore[attr-defined]
        return wrapper

    return decorator
