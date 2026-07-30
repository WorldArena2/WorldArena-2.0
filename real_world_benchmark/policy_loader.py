"""Load a user Policy module from a file path or dotted import path."""

from __future__ import annotations

import importlib
import importlib.util
from pathlib import Path


def load_policy(path_or_module: str):
    """Import a module that defines ``class Policy``.

    Accepts either:
    - a ``.py`` file path (e.g. ``examples/policy_template/policy.py``)
    - a dotted module name (e.g. ``real_world_benchmark.examples.policy_template.policy``)
    """
    path = Path(path_or_module)
    if path.suffix == '.py' and not path.exists():
        raise FileNotFoundError(
            f'Policy file not found: {path_or_module}\n'
            'Use a real path (e.g. examples/policy_template/policy.py) or a module '
            '(e.g. real_world_benchmark.examples.policy_template.policy).'
        )
    if path.exists() and path.suffix == '.py':
        spec = importlib.util.spec_from_file_location('user_policy', str(path))
        if spec is None or spec.loader is None:
            raise ImportError(f'Cannot import policy from file: {path_or_module}')
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)  # type: ignore[arg-type]
        return mod
    try:
        return importlib.import_module(path_or_module)
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            f'Cannot import policy module {path_or_module!r}. '
            'Pass a .py file path or a dotted module name '
            '(e.g. real_world_benchmark.examples.policy_template.policy).'
        ) from exc
