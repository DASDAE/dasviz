"""
Tests that loading dasviz imports no backend (AGENTS.md rule 1).

Entry points are registered whatever extras are installed, so DASCore loads
dasviz's namespaces even without them. This runs in a fresh interpreter where
importing any backend fails, imports every dasviz module and touches every
namespace. Only a method call may import a backend.
"""

import subprocess
import sys
import textwrap

SCRIPT = textwrap.dedent(
    """
    import importlib
    import pkgutil
    import sys
    from importlib.metadata import entry_points

    import dascore as dc

    from dasviz import BACKENDS

    # DASCore itself imports matplotlib, but not pyplot, which is the heavy
    # part. Block every backend (and pyplot) not already imported.
    guarded = {b.module for b in BACKENDS.values()} | {"matplotlib.pyplot"}
    for name in guarded - set(sys.modules):
        sys.modules[name] = None

    import dasviz

    for info in pkgutil.walk_packages(dasviz.__path__, "dasviz."):
        importlib.import_module(info.name)

    patch, spool = dc.get_example_patch(), dc.get_example_spool()
    for group, host in (
        ("dascore.patch_namespace", patch),
        ("dascore.spool_namespace", spool),
    ):
        for ep in entry_points(group=group):
            if ep.value.startswith("dasviz."):
                repr(getattr(host, ep.name))
    print("ok")
    """
)


def test_no_backend_imported_on_load():
    result = subprocess.run(
        [sys.executable, "-c", SCRIPT], capture_output=True, text=True, check=False
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "ok"
