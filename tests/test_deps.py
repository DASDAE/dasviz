"""Tests for the backend table, import_backend and requires."""

import inspect
import sys
from importlib.metadata import entry_points, metadata
from types import MappingProxyType

import pytest
from dascore.exceptions import MissingOptionalDependencyError
from packaging.requirements import Requirement

import dasviz._deps as deps
from dasviz import BACKENDS, Backend, import_backend, requires


@pytest.fixture()
def fake_backends(monkeypatch, tmp_path):
    """Add a backend which is not installed and one which fails to import."""
    broken = tmp_path / "dasviz_broken_pkg"
    broken.mkdir()
    (broken / "__init__.py").write_text("import dasviz_no_such_dependency\n")
    monkeypatch.syspath_prepend(str(tmp_path))
    table = dict(BACKENDS)
    table["absent"] = Backend("dasviz_no_such_module", "absent")
    table["broken"] = Backend("dasviz_broken_pkg", "broken")
    monkeypatch.setattr(deps, "BACKENDS", MappingProxyType(table))


def _requirements():
    """Return dasviz's requirements as (requirement, extra or None) pairs."""
    out = []
    for text in metadata("dasviz").get_all("Requires-Dist") or []:
        req = Requirement(text)
        extra = None
        for name in metadata("dasviz").get_all("Provides-Extra"):
            if req.marker is not None and req.marker.evaluate({"extra": name}):
                extra = name
                break
        out.append((req, extra))
    return out


def _installed_by(extra):
    """Return what an extra installs, following any dasviz[...] in it."""
    out = set()
    for req, name in _requirements():
        if name != extra:
            continue
        if req.name == "dasviz":
            for inner in req.extras:
                out |= _installed_by(inner)
        else:
            out.add(str(Requirement(f"{req.name}{req.specifier}")))
    return out


def _namespace_methods():
    """Yield (label, method) for every public method of every dasviz namespace."""
    for group in ("dascore.patch_namespace", "dascore.spool_namespace"):
        for ep in entry_points(group=group):
            if not ep.value.startswith("dasviz."):
                continue
            cls = ep.load()
            for key in dir(cls):
                value = getattr(cls, key)
                if key.startswith("_") or not callable(value):
                    continue
                yield f"{ep.name}.{key}", value


class TestPackageMetadata:
    """pyproject.toml's extras agree with BACKENDS."""

    @pytest.mark.parametrize("key", list(BACKENDS))
    def test_extra_installs_backend(self, key):
        backend = BACKENDS[key]
        installed = {
            req.name for req, extra in _requirements() if extra == backend.extra
        }
        assert backend.module.partition(".")[0] in installed

    @pytest.mark.parametrize(
        ("key", "alias"),
        [(key, alias) for key, b in BACKENDS.items() for alias in b.aliases],
    )
    def test_alias_is_the_same_extra(self, key, alias):
        # The build may list an alias's requirements directly rather than
        # as dasviz[extra]; either way they must be the extra's.
        assert _installed_by(alias) == _installed_by(BACKENDS[key].extra)

    def test_all_installs_every_backend(self):
        every = set().union(*(_installed_by(b.extra) for b in BACKENDS.values()))
        assert _installed_by("all") == every


class TestImportBackend:
    """import_backend returns a backend or names the extra to install."""

    def test_installed(self):
        mpl = pytest.importorskip("matplotlib")
        assert import_backend("matplotlib") is mpl

    def test_unknown_key(self):
        with pytest.raises(ValueError, match="Unknown dasviz backend"):
            import_backend("not_a_backend")

    @pytest.mark.usefixtures("fake_backends")
    def test_missing_names_extra(self):
        with pytest.raises(MissingOptionalDependencyError) as info:
            import_backend("absent", required_for="`thing`")
        message = str(info.value)
        assert message.startswith("`thing` requires dasviz_no_such_module")
        assert 'pip install "dasviz[absent]"' in message
        assert info.value.install_name == "dasviz[absent]"

    def test_blocked_real_backend(self, monkeypatch):
        """A backend which cannot be found (as when not installed) names its extra."""
        monkeypatch.setitem(sys.modules, "pyvista", None)
        with pytest.raises(MissingOptionalDependencyError, match=r"dasviz\[pyvista\]"):
            import_backend("pyvista")

    @pytest.mark.usefixtures("fake_backends")
    def test_broken_is_not_reported_missing(self):
        """Installing an extra again would not fix a backend which is broken."""
        with pytest.raises(MissingOptionalDependencyError) as info:
            import_backend("broken")
        assert "dasviz[" not in str(info.value)


class TestRequires:
    """requires checks backends on call, and records them."""

    def test_unknown_key_refused_at_definition(self):
        with pytest.raises(ValueError, match="Unknown dasviz backend"):
            requires("not_a_backend")

    def test_keeps_signature_and_records_keys(self):
        def draw(patch, ax=None, *, scale=1.0):
            """Draw it."""
            return patch

        wrapped = requires("matplotlib", "pyvista")(draw)
        assert wrapped.__name__ == "draw"
        assert wrapped.__doc__ == "Draw it."
        assert inspect.signature(wrapped) == inspect.signature(draw)
        assert wrapped.__dasviz_requires__ == ("matplotlib", "pyvista")

    @pytest.mark.usefixtures("fake_backends")
    def test_missing_backend_raises_before_call(self):
        called = []

        @requires("absent")
        def draw():
            called.append(True)

        with pytest.raises(MissingOptionalDependencyError, match="`draw` requires"):
            draw()
        assert not called

    def test_passes_through_when_present(self):
        pytest.importorskip("matplotlib")
        assert requires("matplotlib")(lambda x, y=2: x + y)(1, y=3) == 4


class TestNamespaceMethodsDeclareBackends:
    """Every namespace method says what it needs (AGENTS.md rule 2)."""

    @pytest.mark.parametrize(
        ("label", "method"), list(_namespace_methods()), ids=lambda x: str(x)
    )
    def test_declares_known_backends(self, label, method):
        keys = getattr(method, "__dasviz_requires__", None)
        assert keys, f"{label} is not decorated with dasviz.requires"
        assert set(keys) <= set(BACKENDS)
