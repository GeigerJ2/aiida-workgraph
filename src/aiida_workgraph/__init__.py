"""Deprecation shim: aiida-workgraph has moved into aiida-core as ``aiida.workgraph``.

Every module in this package now forwards to its ``aiida.workgraph`` counterpart. Kept for one release so
existing ``import aiida_workgraph`` code and the registered entry points keep working; then archived.
"""

import aiida.workgraph as _aiida_workgraph
import aiida_workgraph.task  # noqa: F401  force-load the ``task`` submodule now (binds attr=module)...
from aiida.workgraph import *  # noqa: F401,F403
from aiida.workgraph import __version__  # noqa: F401
from aiida.workgraph.tasks.shelljob_task import shelljob  # noqa: F401

# ...then rebind ``task`` to the decorator (last write wins; a later ``import aiida_workgraph.task``
# finds the submodule already loaded and does not re-bind the attribute).
task = _aiida_workgraph.task
