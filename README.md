**Experimental.** dasviz is an early-stage package; its API will change without notice.

dasviz adds advanced visualizations to [DASCore](https://github.com/DASDAE/dascore) for distributed acoustic sensing (DAS) data: 3D scenes, interactive notebook views, animations, and the next generation of DASCore's figures. It adds no classes of its own. Install it, and every patch and spool gains four namespaces:

| Namespace | For | Example |
|---|---|---|
| `fig` | static figures, to save or publish | `patch.fig.waterfall()` |
| `nb` | interactive views in a notebook | (coming) |
| `scene` | 3D scenes | (coming) |
| `anim` | animations and movies | (coming) |

## Install

Each backend dasviz draws with is an optional extra:

```bash
pip install dasviz               # the namespaces, and what DASCore already has
pip install "dasviz[pyvista]"    # or dasviz[pv]: 3D with PyVista
pip install "dasviz[all]"        # every backend
```

A method whose backend is missing says which extra to install.

## Quick start

```python
import dascore as dc

patch = dc.get_example_patch()
patch.fig.waterfall(scale=0.5)
```

There is no need to `import dasviz`: DASCore finds its namespaces when they are first used.

## Contributing

See [Add a method](https://dasdae.github.io/dasviz/user-guide/add-a-method.html) for how the package is laid out and the rules a new method follows.
