"""
The matplotlib backend (extra: ``dasviz[matplotlib]``, alias ``dasviz[mpl]``).

This is where ``dascore.viz`` is incubated, and where its successors are
built, so that DASCore can one day stop depending on matplotlib. Like every
backend package, importing it does not import matplotlib: each renderer
imports matplotlib in its body, behind `dasviz.requires`.
"""

from dasviz.mpl.waterfall import waterfall

__all__ = ["waterfall"]
