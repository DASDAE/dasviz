"""The ``anim`` namespace: animations and movies. Empty for now."""

from dasviz.namespaces._base import DasvizPatchNamespace, DasvizSpoolNamespace


class AnimPatch(DasvizPatchNamespace):
    """Animations and movies. Empty for now."""

    name = "anim"


class AnimSpool(DasvizSpoolNamespace):
    """Animations and movies. Empty for now."""

    name = "anim"
    _patch_namespace = AnimPatch
