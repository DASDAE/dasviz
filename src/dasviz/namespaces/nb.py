"""The ``nb`` namespace: interactive views in a notebook. Empty for now."""

from dasviz.namespaces._base import DasvizPatchNamespace, DasvizSpoolNamespace


class NbPatch(DasvizPatchNamespace):
    """Interactive views in a notebook. Empty for now."""

    name = "nb"


class NbSpool(DasvizSpoolNamespace):
    """Interactive views in a notebook. Empty for now."""

    name = "nb"
    _patch_namespace = NbPatch
