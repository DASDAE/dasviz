"""
The PyVista backend (extra: ``dasviz[pyvista]``, alias ``dasviz[pv]``).

Renderers drawn with PyVista go here, and are usually bound to the ``scene``
(3D) or ``anim`` namespace in `dasviz.namespaces`. Like every backend
package, importing this must not import PyVista or VTK: each renderer
imports PyVista in its body and is decorated ``@requires("pyvista")``.
AGENTS.md has the rules and ``tests/test_lightweight_import.py`` enforces
them.
"""

from dasviz.pv.patchviewer import PatchViewer, patchviewer

__all__ = ["PatchViewer", "patchviewer"]
