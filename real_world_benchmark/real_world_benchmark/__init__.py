"""Package bootstrap for flat repository layout.

The git repo root *is* the package content (``serve_policy_worldarena.py``,
``worldarena/``, ...), but Python only treats a directory as importable when it
appears on ``sys.path``. This stub package extends ``__path__`` to the
repository root so ``python -m real_world_benchmark.serve_policy_worldarena``
works from inside the repo.
"""

from __future__ import annotations

from pathlib import Path

_repo_root = Path(__file__).resolve().parent.parent
_repo_root_str = str(_repo_root)
if _repo_root_str not in __path__:
    __path__.insert(0, _repo_root_str)
