"""The bases every dasviz namespace builds on."""

from __future__ import annotations

from typing import ClassVar

from dascore.utils.namespace import PatchNameSpace, SpoolNameSpace


def _methods(cls) -> list[str]:
    """Return the names of a namespace class's public methods."""
    return sorted(
        key
        for key in dir(cls)
        if not key.startswith("_")
        and key not in ("name", "entry_point_group")
        and callable(getattr(cls, key))
    )


class _Described:
    """Give a namespace a repr which lists its methods."""

    def __repr__(self) -> str:
        host = type(self._obj).__name__
        methods = ", ".join(_methods(type(self))) or "no methods yet"
        return f"<dasviz {type(self).name!r} namespace on {host}: {methods}>"


class DasvizPatchNamespace(_Described, PatchNameSpace):
    """The base of dasviz's Patch namespaces; registers nothing itself."""

    name = None


class DasvizSpoolNamespace(_Described, SpoolNameSpace):
    """
    The base of dasviz's Spool namespaces; registers nothing itself.

    A patch-only method called on a spool is pointed to the patch it needs,
    as ``dascore.viz`` does. Set ``_patch_namespace`` to the namespace's
    Patch class. (It is private so that DASCore does not take the class
    for a method.)
    """

    name = None
    _patch_namespace: ClassVar[type | None] = None

    def __getattr__(self, item):
        """Point a patch method at the patch it needs."""
        patch_ns = type(self)._patch_namespace
        if patch_ns is not None and item in _methods(patch_ns):
            name = type(self).name
            msg = (
                f"{name}.{item} draws a patch, and a spool is many of them. "
                "Merge the ones you mean into one patch first, then draw it: "
                f"spool.chunk(time=None)[0].{name}.{item}()"
            )
            raise AttributeError(msg)
        msg = f"{type(self).__name__!r} object has no attribute {item!r}"
        raise AttributeError(msg)
