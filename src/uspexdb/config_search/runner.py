from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

from ..paths import load_json, resolve_database_path, resolve_output_dir, resolve_reference_module
from ..reference import load_reference_module


def _format_kwargs(fmt_name: str, fmt_config: Dict[str, Any]) -> Dict[str, Any]:
    if fmt_name == "cif":
        return {"include_symmetry": bool(fmt_config.get("include_symmetry", True))}
    if fmt_name == "json":
        return {
            "include_structure": bool(fmt_config.get("include_structure", True)),
            "indent": int(fmt_config.get("indent", 2)),
        }
    if fmt_name == "poscar":
        return {
            "direct": bool(fmt_config.get("direct", True)),
            "sort": bool(fmt_config.get("sort", False)),
        }
    return {}


def run_config_mode(config_json: Path) -> Dict[str, Any]:
    config_file = Path(config_json).expanduser().resolve()
    config_payload = load_json(config_file)
    reference_module = resolve_reference_module(config_payload, config_file)
    reference = load_reference_module(python_module=reference_module)
    config_manager = reference.ConfigManager(str(config_file))
    db_path = resolve_database_path(config_manager.config, config_file)
    filters = config_manager.get_search_filters()
    export_cfg = config_manager.get_export_settings()
    output_root = resolve_output_dir(export_cfg, config_file, "output/config_filtered")
    limit = int(export_cfg.get("limit", 100))
    formats = export_cfg.get("formats", {})

    print(f"Config JSON: {config_file}")
    print(f"Database:    {db_path}")
    print(f"Output:      {output_root}")
    print(f"Reference:   {reference.__file__}")
    print("Filters:")
    print(json.dumps(filters, indent=2, ensure_ascii=False, default=str))

    stats = {"success": 0, "failed": 0, "errors": [], "formats": {}}
    db = reference.DirectoryStructureDB(str(db_path))
    try:
        indices = db.search(**filters)
        print(f"Found structures: {len(indices)}")
        if not indices:
            return {"found": 0, **stats}

        for fmt_name, fmt_config in formats.items():
            fmt_name = str(fmt_name).lower()
            subdir = fmt_config.get("output_subdir", f"{fmt_name}_files")
            fmt_output_dir = output_root / str(subdir)
            kwargs = _format_kwargs(fmt_name, fmt_config)
            fmt_stats = db.export_structures(
                indices=indices,
                output_dir=str(fmt_output_dir),
                format=fmt_name,
                limit=limit,
                **kwargs,
            )
            stats["formats"][fmt_name] = fmt_stats
            stats["success"] += int(fmt_stats.get("success", 0))
            stats["failed"] += int(fmt_stats.get("failed", 0))
            stats["errors"].extend(fmt_stats.get("errors", []))
    finally:
        db.close()

    logging_cfg = config_manager.get_logging_settings()
    if logging_cfg.get("save_stats", True):
        stats_file = logging_cfg.get("stats_file", "stats.json")
        stats_path = (output_root / str(stats_file)).resolve()
        stats_path.parent.mkdir(parents=True, exist_ok=True)
        with stats_path.open("w", encoding="utf-8") as fh:
            json.dump(stats, fh, indent=2, ensure_ascii=False, default=str)
        stats["stats_file"] = str(stats_path)

    print("Export stats:")
    print(json.dumps(stats, indent=2, ensure_ascii=False, default=str))
    return {"found": len(indices), **stats}
