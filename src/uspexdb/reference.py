from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
from types import ModuleType
from typing import Optional


DEFAULT_REFERENCE_MODULE = Path(__file__).resolve().parent / "legacy" / "uspexdb_v2.py"


def load_reference_module(python_module: Optional[Path] = None) -> ModuleType:
    script_path = (
        Path(python_module).expanduser().resolve()
        if python_module is not None
        else DEFAULT_REFERENCE_MODULE.resolve()
    )
    if not script_path.exists():
        raise FileNotFoundError(f"reference module not found: {script_path}")

    module_name = "uspexdb_reference_" + str(abs(hash(str(script_path))))
    if module_name in sys.modules:
        return sys.modules[module_name]

    spec = importlib.util.spec_from_file_location(module_name, script_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot import reference module from {script_path}")

    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module
