"""
The namespaces dasviz adds to DASCore's Patch and Spool.

Namespaces are named by what they are for; any of them may use any backend.

=========  =====================================  ===============================
Name       For                                    Example
=========  =====================================  ===============================
``fig``    static figures, to save or publish     ``patch.fig.waterfall()``
``nb``     interactive views in a notebook        (none yet)
``scene``  3D scenes                              ``patch.scene.patchviewer()``
``anim``   animations and movies                  (none yet)
=========  =====================================  ===============================

DASCore loads each from its entry point (see pyproject.toml) the first time
it is used, so ``import dasviz`` is never needed. Loading one imports no
backend.
"""
