from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional


PACKAGE_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = PACKAGE_ROOT.parents[1]


def load_json(path: Path) -> Dict[str, Any]:
    with Path(path).expanduser().resolve().open("r", encoding="utf-8") as fh:
        return json.load(fh)


def section(payload: Dict[str, Any], name: str) -> Dict[str, Any]:
    value = payload.get(name, {})
    return value if isinstance(value, dict) else {}


def resolve_path(value: Any, base_dir: Path) -> Path:
    path = Path(str(value)).expanduser()
    if not path.is_absolute():
        path = base_dir / path
    return path.resolve()


def optional_int(value: Any) -> Optional[int]:
    if value is None:
        return None
    text = str(value).strip().lower()
    if text in {"", "none", "null", "all", "full"}:
        return None
    return int(text)


def resolve_database_path(payload: Dict[str, Any], config_file: Path) -> Path:
    database_cfg = section(payload, "database")
    db_value = database_cfg.get("path")
    if not db_value:
        raise ValueError(f"database.path is required in {config_file}")
    return resolve_path(db_value, config_file.parent)


def resolve_reference_module(payload: Dict[str, Any], config_file: Path) -> Optional[Path]:
    reference_cfg = section(payload, "reference")
    module_path = reference_cfg.get("python_module")
    if not module_path:
        return None
    return resolve_path(module_path, config_file.parent)


def resolve_output_dir(output_cfg: Dict[str, Any], config_file: Path, default_dir: str) -> Path:
    output_dir = output_cfg.get("output_dir", default_dir)
    return resolve_path(output_dir, config_file.parent)


def resolve_fragment_path(query_cfg: Dict[str, Any], query_file: Path) -> Path:
    fragment = query_cfg.get("fragment_mol2")
    if not fragment:
        raise ValueError(f"query.fragment_mol2 is required in {query_file}")
    fragment_path = resolve_path(fragment, query_file.parent)
    if not fragment_path.exists():
        raise FileNotFoundError(
            "query.fragment_mol2 was resolved relative to query.json but the file was not found: "
            f"{fragment_path}"
        )
    return fragment_path


def resolve_graph_cache_path(payload: Dict[str, Any], db_path: Path) -> Path:
    graph_cache_cfg = section(payload, "graph_cache")
    graph_cache_value = graph_cache_cfg.get("path")
    if graph_cache_value:
        return resolve_path(graph_cache_value, db_path)

    for rel_path in ("indexes/graph_cache_updated2_fast_full_v2", "indexes/graph_cache"):
        candidate = (db_path / rel_path).resolve()
        if candidate.exists():
            return candidate
    return (db_path / "indexes/graph_cache_updated2_fast_full_v2").resolve()


def resolve_compact_cache_path(payload: Dict[str, Any], query_file: Path) -> Path:
    compact_cfg = section(payload, "compact_cache")
    compact_path = compact_cfg.get("path", "graph_cache_v3")
    return resolve_path(compact_path, query_file.parent)
