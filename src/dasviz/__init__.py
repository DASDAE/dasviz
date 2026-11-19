"""
dasviz: advanced visualizations for DAS data, as DASCore namespaces.

Installing dasviz gives every DASCore patch and spool four namespaces, each
named by what it is for: ``fig`` (static figures), ``nb`` (interactive
notebook views), ``scene`` (3D scenes) and ``anim`` (animations). The code
that draws them is organized by backend (``dasviz.mpl``, ``dasviz.pv``), and
each backend is an optional extra, imported only when a method needs it.
"""

__version__ = "0.0.1"

from dasviz._deps import BACKENDS, Backend, import_backend, requires

__all__ = ["BACKENDS", "Backend", "__version__", "import_backend", "requires"]
