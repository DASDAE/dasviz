# Agent instructions

dasviz adds visualizations to DASCore's `Patch` and `Spool` as method namespaces (DASCore's `make-namespace` skill describes the mechanism). It is also the incubator for `dascore.viz`, and its eventual replacement, so that DASCore can one day stop depending on matplotlib.

## Layout: code by backend, namespaces by meaning

- `src/dasviz/<backend>/` holds the renderers drawn with one backend: `mpl/` (matplotlib) and `pv/` (PyVista; empty for now, owned by a collaborator). Add a package for a new backend (anywidget, plotly, …) only when its first method arrives.
- `src/dasviz/namespaces/` holds what users type: `fig` (static figures), `nb` (interactive notebook views), `scene` (3D), `anim` (animations). Each module defines a `<Name>Patch` and a `<Name>Spool` class, both registered as entry points in `pyproject.toml`. A namespace may bind renderers from any backend.
- `src/dasviz/prep/` is backend-free: color limits, decimation, cell edges, labels. Renderers get their numbers from here and only draw.
- `src/dasviz/_deps.py` holds `BACKENDS`, the one list of backends, with the extra and aliases that install each.

## Rules

1. **A backend package never imports its backend at module level.** Entry points load whatever extras are installed, so loading a namespace must not import matplotlib's `pyplot`, PyVista, etc. Import the backend inside the renderer's body; put types used in annotations under `TYPE_CHECKING`. `tests/test_lightweight_import.py` enforces this.
2. **Every public renderer is decorated with `@requires("<backend>", ...)`.** It checks for the backend when the method is called, and if it is missing names the extra to install. `tests/test_deps.py` fails for a namespace method without it.
3. **`prep/` imports no plotting library.** Test it with plain arrays and patches.
4. **Bind renderers as class attributes**, as DASCore does: `waterfall = waterfall` in `FigPatch`. The renderer takes the patch (or spool) as its first argument.
5. A method for spools goes on the `<Name>Spool` class. A patch method called on a spool already raises an error pointing to `spool.chunk(time=None)[0]`.
6. Don't import DASCore's private helpers (`dascore.utils.plotting._…`); port what is needed into `prep/`. Public DASCore API (`dascore.units`, `dascore.exceptions`, `dascore.utils.misc.optional_import`, `dascore.constants`) is fine.

## Adding things

- **A method**: write the renderer in `src/dasviz/<backend>/<name>.py`, export it from the backend's `__init__`, bind it in the namespace class, test it in `tests/<backend>/` (use `pytest.importorskip` for the backend), and list it in `great-docs.yml`'s reference.
- **A backend**: add an entry to `BACKENDS`, an extra and its alias to `pyproject.toml` (and to `all`), and a package `src/dasviz/<backend>/`. `tests/test_deps.py` checks the extras agree with `BACKENDS`, and `tests/test_lightweight_import.py` blocks every backend listed there automatically.
- **A namespace**: needs a reason that fits none of the four. Add it to `namespaces/`, both entry-point groups, the `NAMES` in `tests/test_namespaces.py`, the README table, and DASCore's `plugin_registry/{patch,spool}.csv`.

## Checks

- `uv run pytest`, `uv run ruff check .`, `uv run ruff format --check .`.
- Without extras: `uv sync` then `uv run pytest` (backend tests skip; the import-guard tests still run). CI runs Python 3.12 and 3.14, each with no extras and with all of them, and once against DASCore's `dev` branch.
- Docs: a Great Docs site (`great-docs.yml`) from the README, the docstrings, and `docs/guide/` (pages ordered by number, grouped by Diataxis `guide-section`). Build with `uv run --group docs great-docs build` (needs Quarto).

## Releasing

Bump `__version__` in `src/dasviz/__init__.py` on `main`, then publish a GitHub release tagged `v<version>`. `.github/workflows/publish.yml` builds, refuses a tag which doesn't match the version, and publishes to PyPI by trusted publishing (environment `pypi`; no token is stored). The docs deploy from the same release.
