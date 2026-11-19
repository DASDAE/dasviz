"""The ``fig`` namespace: static figures, to save or publish."""

from dasviz.mpl import waterfall
from dasviz.namespaces._base import DasvizPatchNamespace, DasvizSpoolNamespace


class FigPatch(DasvizPatchNamespace):
    """Static figures, to save or publish."""

    name = "fig"
    waterfall = waterfall


class FigSpool(DasvizSpoolNamespace):
    """Static figures, to save or publish."""

    name = "fig"
    _patch_namespace = FigPatch
