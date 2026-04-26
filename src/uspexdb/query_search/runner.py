from __future__ import annotations

from contextlib import contextmanager
import json
import os
from pathlib import Path
from typing import Any, Dict, Optional

from ..paths import (
    load_json,
    optional_int,
    resolve_compact_cache_path,
    resolve_database_path,
    resolve_fragment_path,
    resolve_graph_cache_path,
    resolve_output_dir,
    resolve_reference_module,
    section,
)
from .compact_cache import CompactGraphCache, build_compact_cache
from .search import CFastMol2ContactSearch, write_payload
from .validate import compare_with_reference_file


@contextmanager
def suppress_native_stderr():
    saved_fd = os.dup(2)
    try:
        with open(os.devnull, "w", encoding="utf-8") as devnull:
            os.dup2(devnull.fileno(), 2)
            yield
    finally:
        os.dup2(saved_fd, 2)
        os.close(saved_fd)


def compact_cache_build_settings(query_payload: Dict[str, Any]) -> Dict[str, int]:
    compact_cfg = section(query_payload, "compact_cache")
    return {
        "batch_structures": int(compact_cfg.get("batch_structures", 256)),
        "workers": int(compact_cfg.get("workers", 1)),
        "progress_every": int(compact_cfg.get("progress_every", 1000)),
        "parquet_batch_rows": int(compact_cfg.get("parquet_batch_rows", 200000)),
    }


def run_query_mode(
    query_json: Path,
    rebuild_compact: bool = False,
    validate: Optional[bool] = None,
) -> Dict[str, Any]:
    query_file = Path(query_json).expanduser().resolve()
    payload = load_json(query_file)
    db_path = resolve_database_path(payload, query_file)
    reference_module = resolve_reference_module(payload, query_file)
    query_cfg = section(payload, "query")
    compact_cfg = section(payload, "compact_cache")
    output_cfg = section(payload, "output")

    if query_cfg.get("mode", "mol2_contact") != "mol2_contact":
        raise ValueError("Only query.mode='mol2_contact' is currently supported")
    if query_cfg.get("search_backend", "c_anchor_v3") != "c_anchor_v3":
        raise ValueError("Only query.search_backend='c_anchor_v3' is currently supported")

    fragment_path = resolve_fragment_path(query_cfg, query_file)
    source_cache = resolve_graph_cache_path(payload, db_path)
    compact_cache_dir = resolve_compact_cache_path(payload, query_file)
    should_rebuild = bool(rebuild_compact or compact_cfg.get("rebuild", False))
    compact_build = compact_cache_build_settings(payload)

    print("MOL2 contact search")
    print(f"Query JSON:   {query_file}")
    print(f"Database:     {db_path}")
    print(f"Fragment:     {fragment_path}")
    print(f"Graph cache:  {source_cache}")
    print(f"Compact:      {compact_cache_dir}")

    with suppress_native_stderr():
        build_compact_cache(
            source_cache,
            compact_cache_dir,
            overwrite=should_rebuild,
            batch_structures=compact_build["batch_structures"],
            workers=compact_build["workers"],
            progress_every=compact_build["progress_every"],
            parquet_batch_rows=compact_build["parquet_batch_rows"],
        )

    compact_cache = CompactGraphCache(compact_cache_dir)
    searcher = CFastMol2ContactSearch(
        db_path=db_path,
        compact_cache=compact_cache,
        reference_python_module=reference_module,
    )
    try:
        with suppress_native_stderr():
            result_payload = searcher.search(
                fragment_mol2=fragment_path,
                radius_max=float(query_cfg.get("radius_max", 4.0)),
                contact_elements=query_cfg.get("contact_elements", []),
                contact_scope=query_cfg.get("contact_scope", "intermolecular"),
                strict_bonds=bool(query_cfg.get("strict_bonds", False)),
                strict_atom_types=bool(query_cfg.get("strict_atom_types", True)),
                allow_hydrogen_wildcards=bool(query_cfg.get("allow_hydrogen_wildcards", True)),
                structure_ids=query_cfg.get("structure_ids", []),
                refcodes=query_cfg.get("refcodes", []),
                max_structures=optional_int(query_cfg.get("max_structures")),
                progress_every=int(query_cfg.get("progress_every", 0)),
            )
        output_dir = resolve_output_dir(output_cfg, query_file, "output/query")
        summary = write_payload(
            result_payload,
            output_dir,
            basename=str(output_cfg.get("basename", "mol2_contact_results")),
            output_settings=output_cfg,
            db=searcher.db,
        )
    finally:
        searcher.close()

    result_summary = result_payload.get("summary", {})
    print(
        f"  Done: {int(result_summary.get('structures_with_hits', 0))} structures, "
        f"{int(result_summary.get('contacts_found', 0))} contacts",
        flush=True,
    )
    print(f"  Output: {output_dir}", flush=True)

    validation_cfg = section(payload, "validation")
    validate_enabled = bool(validation_cfg.get("enabled", False)) if validate is None else bool(validate)
    reference_json = validation_cfg.get("reference_json")
    if validate_enabled and reference_json:
        reference_path = Path(reference_json).expanduser()
        if not reference_path.is_absolute():
            reference_path = (query_file.parent / reference_path).resolve()
        if reference_path.exists():
            validation_payload = compare_with_reference_file(result_payload, reference_path)
            validation_path = output_dir / f"{output_cfg.get('basename', 'mol2_contact_results')}_validation.json"
            with validation_path.open("w", encoding="utf-8") as fh:
                json.dump(validation_payload, fh, indent=2, ensure_ascii=False, default=str)
            if not validation_payload.get("exact_key_match", False):
                print(f"  Validation: mismatch, details: {validation_path}", flush=True)
        else:
            print(f"  Validation skipped: reference JSON not found: {reference_path}")

    return summary
