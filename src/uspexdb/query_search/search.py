from __future__ import annotations

from collections import Counter
import csv
import html
import json
import os
from pathlib import Path
import shutil
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple

from ase.data import chemical_symbols
from ase.geometry import find_mic
import numpy as np
import pandas as pd

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False

from . import _c_anchor
from .codes import atomic_number_from_symbol, generic_edge_code
from .compact_cache import CompactGraphCache
from .mol2 import QueryGraph, mol2_to_query_graph
from ..reference import load_reference_module


def _normalize_contact_scope(value: Optional[str]) -> str:
    text = str(value or "intermolecular").strip().lower()
    if text in {"intermolecular", "inter", "intercomponent", "different_component"}:
        return "intermolecular"
    if text in {"all", "any", "both", "inter_and_intra", "intermolecular_and_intramolecular"}:
        return "all"
    raise ValueError(f"unsupported contact_scope: {value!r}")


def _symbol(number: int) -> str:
    number = int(number)
    if 0 < number < len(chemical_symbols):
        return str(chemical_symbols[number])
    return str(number)


def _contact_base_label(query: QueryGraph) -> str:
    anchor_element = _symbol(int(query.atomic_numbers[query.anchor_index]))
    if query.prev_index is None:
        return anchor_element
    prev_element = _symbol(int(query.atomic_numbers[query.prev_index]))
    return f"{prev_element}-{anchor_element}"


def _contact_element_numbers(values: Sequence[Any], query: QueryGraph) -> Tuple[Set[int], bool]:
    result: Set[int] = set()
    wildcard = False
    for value in values or []:
        text = str(value).strip()
        if not text:
            continue
        if text == "*":
            wildcard = True
            continue
        number = atomic_number_from_symbol(text)
        if number > 0:
            result.add(int(number))
    if not result and not wildcard:
        result.update(int(value) for value in query.contact_elements_from_dummy if int(value) > 0)
    return result, wildcard or not result


def _query_required_edge_codes(query: QueryGraph, strict_bonds: bool) -> Set[int]:
    if strict_bonds:
        # The current compact prefilter supports strict edge codes, but the first
        # experimental backend keeps the public path focused on strict_bonds=false.
        # Returning generic codes here preserves candidate safety if strict_bonds is
        # accidentally enabled: no valid candidate is lost before exact matching.
        return query.generic_edge_codes
    return query.generic_edge_codes


class CFastMol2ContactSearch:
    def __init__(
        self,
        db_path: Path,
        compact_cache: CompactGraphCache,
        reference_python_module: Optional[Path] = None,
    ):
        self.db_path = Path(db_path).expanduser().resolve()
        self.compact_cache = compact_cache
        self.reference = load_reference_module(python_module=reference_python_module)
        self.db = self.reference.DirectoryStructureDB(str(self.db_path))
        self._angle_degrees = self.reference._angle_degrees
        self._dihedral_degrees = self.reference._dihedral_degrees

    def close(self) -> None:
        self.db.close()

    def _requested_structure_ids(
        self,
        structure_ids: Optional[Sequence[int]],
        refcodes: Optional[Sequence[str]],
        max_structures: Optional[int],
    ) -> Optional[Set[int]]:
        structure_ids = structure_ids or []
        refcodes = refcodes or []
        if not structure_ids and not refcodes and max_structures is None:
            return None
        result: List[int] = []
        if structure_ids:
            result = [int(value) for value in structure_ids]
        elif refcodes:
            seen: Set[int] = set()
            for refcode in refcodes:
                record = self.db._load_refcode_record(str(refcode).strip())
                if record is None:
                    continue
                structure_id = int(record["structure_id"])
                if structure_id not in seen:
                    seen.add(structure_id)
                    result.append(structure_id)
        else:
            result = [int(value) for value in self.compact_cache.structure_ids.tolist()]
        if max_structures is not None:
            result = result[: int(max_structures)]
        return set(int(value) for value in result)

    def _match_fragment(self, query: QueryGraph, node_start: int, node_stop: int, strict_bonds: bool, strict_atom_types: bool, allow_hydrogen_wildcards: bool):
        return _c_anchor.match_fragment(
            query.atomic_numbers,
            query.hybridization,
            query.adj_offsets,
            query.adj_neighbors,
            query.adj_orders,
            query.match_order,
            self.compact_cache.node_atomic_numbers[node_start:node_stop],
            self.compact_cache.node_hybridization[node_start:node_stop],
            self.compact_cache.adjacency_offsets[node_start : node_stop + 1],
            self.compact_cache.adjacency_neighbors,
            self.compact_cache.adjacency_orders,
            int(bool(strict_bonds)),
            int(bool(strict_atom_types)),
            int(bool(allow_hydrogen_wildcards)),
        )

    def search(
        self,
        fragment_mol2: Path,
        radius_max: float,
        contact_elements: Optional[Sequence[str]] = None,
        contact_scope: str = "intermolecular",
        strict_bonds: bool = False,
        strict_atom_types: bool = True,
        allow_hydrogen_wildcards: bool = True,
        structure_ids: Optional[Sequence[int]] = None,
        refcodes: Optional[Sequence[str]] = None,
        max_structures: Optional[int] = None,
        progress_every: int = 0,
    ) -> Dict[str, Any]:
        query = mol2_to_query_graph(Path(fragment_mol2))
        contact_scope = _normalize_contact_scope(contact_scope)
        contact_numbers, wildcard_contact = _contact_element_numbers(contact_elements or [], query)
        required_counts = dict(query.required_element_counts)
        required_edges = _query_required_edge_codes(query, strict_bonds)
        candidate_indices = self.compact_cache.candidate_indices(
            required_counts=required_counts,
            required_edge_codes=required_edges,
            allow_hydrogen_wildcards=allow_hydrogen_wildcards,
            strict_bonds=False,
        )
        requested_ids = self._requested_structure_ids(structure_ids, refcodes, max_structures)
        if requested_ids is not None:
            candidate_indices = [
                cache_index
                for cache_index in candidate_indices
                if self.compact_cache.structure_id_for_cache_index(cache_index) in requested_ids
            ]

        print(f"  Candidates: {len(candidate_indices)}", flush=True)

        results: List[Dict[str, Any]] = []
        structures_with_hits = 0
        contact_base_label = _contact_base_label(query)

        for candidate_pos, cache_index in enumerate(candidate_indices, start=1):
            structure_id = self.compact_cache.structure_id_for_cache_index(cache_index)
            node_start, node_stop = self.compact_cache.node_range(cache_index)
            if node_stop <= node_start:
                continue
            node_numbers = self.compact_cache.node_atomic_numbers[node_start:node_stop]
            component_ids = self.compact_cache.node_component_ids[node_start:node_stop]
            if wildcard_contact:
                contact_candidate_indices = np.arange(node_stop - node_start, dtype=np.int32)
            else:
                contact_mask = np.isin(node_numbers, np.asarray(sorted(contact_numbers), dtype=np.uint8))
                contact_candidate_indices = np.flatnonzero(contact_mask).astype(np.int32)
            if contact_candidate_indices.size == 0:
                continue

            raw_matches = self._match_fragment(
                query=query,
                node_start=node_start,
                node_stop=node_stop,
                strict_bonds=strict_bonds,
                strict_atom_types=strict_atom_types,
                allow_hydrogen_wildcards=allow_hydrogen_wildcards,
            )
            if not raw_matches:
                continue

            atoms = None
            metadata = None
            positions = None
            structure_hits = 0
            seen_fragment_keys: Set[Tuple[Tuple[int, int], ...]] = set()
            seen_contact_keys: Set[Tuple[int, int, Tuple[int, ...]]] = set()

            for mapping_tuple in raw_matches:
                targets = tuple(int(value) for value in mapping_tuple)
                fragment_key = tuple(sorted((int(query_idx), int(target)) for query_idx, target in enumerate(targets)))
                if fragment_key in seen_fragment_keys:
                    continue
                seen_fragment_keys.add(fragment_key)

                matched_atoms = tuple(sorted(targets))
                anchor_global = int(targets[query.anchor_index])
                prev_global = int(targets[query.prev_index]) if query.prev_index is not None else None
                prev2_global = int(targets[query.prev2_index]) if query.prev2_index is not None else None
                anchor_component = int(component_ids[anchor_global])

                eligible = []
                matched_set = set(matched_atoms)
                for contact_node in contact_candidate_indices.tolist():
                    contact_node = int(contact_node)
                    if contact_node in matched_set:
                        continue
                    contact_component = int(component_ids[contact_node])
                    if contact_scope == "intermolecular" and contact_component == anchor_component:
                        continue
                    eligible.append((contact_node, contact_component, int(node_numbers[contact_node])))
                if not eligible:
                    continue

                if atoms is None:
                    atoms, metadata = self.db.get_structure(int(structure_id))
                    if atoms is None or metadata is None:
                        break
                    positions = atoms.get_positions()

                contact_nodes = np.asarray([item[0] for item in eligible], dtype=int)
                raw_vectors = positions[contact_nodes] - positions[anchor_global]
                mic_vectors, _ = find_mic(raw_vectors, atoms.cell, atoms.pbc)
                distances = np.linalg.norm(mic_vectors, axis=1)
                close_indices = np.flatnonzero(distances <= float(radius_max))

                for close_idx in close_indices.tolist():
                    contact_node, contact_component, contact_number = eligible[int(close_idx)]
                    contact_key = (anchor_global, int(contact_node), matched_atoms)
                    if contact_key in seen_contact_keys:
                        continue
                    seen_contact_keys.add(contact_key)

                    mic_vector = np.asarray(mic_vectors[int(close_idx)], dtype=float)
                    raw_vector = np.asarray(raw_vectors[int(close_idx)], dtype=float)
                    distance = float(distances[int(close_idx)])
                    image_delta = mic_vector - raw_vector
                    try:
                        contact_offset = np.rint(
                            np.linalg.solve(np.asarray(atoms.cell, dtype=float).T, image_delta)
                        ).astype(int)
                    except Exception:
                        contact_offset = np.zeros(3, dtype=int)

                    anchor_position = positions[anchor_global]
                    contact_position = anchor_position + mic_vector
                    angle = None
                    torsion = None
                    prev_position = None
                    if prev_global is not None:
                        prev_vector, _ = find_mic(
                            positions[int(prev_global)] - anchor_position,
                            atoms.cell,
                            atoms.pbc,
                        )
                        prev_position = anchor_position + prev_vector
                        angle = self._angle_degrees(prev_position, anchor_position, contact_position)
                    if prev_position is not None and prev2_global is not None:
                        prev2_vector, _ = find_mic(
                            positions[int(prev2_global)] - prev_position,
                            atoms.cell,
                            atoms.pbc,
                        )
                        prev2_position = prev_position + prev2_vector
                        torsion = self._dihedral_degrees(
                            prev2_position,
                            prev_position,
                            anchor_position,
                            contact_position,
                        )

                    contact_element = _symbol(contact_number)
                    same_component = int(contact_component) == int(anchor_component)
                    result = {
                        "structure_id": int(structure_id),
                        "refcode": metadata.get("refcode") if metadata else None,
                        "fragment_mol2": str(Path(fragment_mol2).expanduser().resolve()),
                        "matched_atoms": [int(value) + 1 for value in matched_atoms],
                        "matched_atoms_zero_based": [int(value) for value in matched_atoms],
                        "matched_elements": [_symbol(int(node_numbers[int(value)])) for value in matched_atoms],
                        "prev_atom": int(prev_global + 1) if prev_global is not None else None,
                        "prev_atom_zero_based": int(prev_global) if prev_global is not None else None,
                        "prev_element": _symbol(int(node_numbers[int(prev_global)])) if prev_global is not None else None,
                        "anchor_atom": int(anchor_global + 1),
                        "anchor_atom_zero_based": int(anchor_global),
                        "anchor_element": _symbol(int(node_numbers[int(anchor_global)])),
                        "anchor_component": int(anchor_component),
                        "contact_atom": int(contact_node + 1),
                        "contact_atom_zero_based": int(contact_node),
                        "contact_element": contact_element,
                        "contact_component": int(contact_component),
                        "contact_scope": contact_scope,
                        "same_component": bool(same_component),
                        "contact_relation": "intracomponent" if same_component else "intercomponent",
                        "contact_label": f"{contact_base_label}...{contact_element}",
                        "distance": distance,
                        "anchor_contact_distance": distance,
                        "contact_offset_x": int(contact_offset[0]),
                        "contact_offset_y": int(contact_offset[1]),
                        "contact_offset_z": int(contact_offset[2]),
                        "prev_x": float(positions[int(prev_global)][0]) if prev_global is not None else None,
                        "prev_y": float(positions[int(prev_global)][1]) if prev_global is not None else None,
                        "prev_z": float(positions[int(prev_global)][2]) if prev_global is not None else None,
                        "prev_image_x": float(prev_position[0]) if prev_position is not None else None,
                        "prev_image_y": float(prev_position[1]) if prev_position is not None else None,
                        "prev_image_z": float(prev_position[2]) if prev_position is not None else None,
                        "anchor_x": float(anchor_position[0]),
                        "anchor_y": float(anchor_position[1]),
                        "anchor_z": float(anchor_position[2]),
                        "contact_x": float(positions[int(contact_node)][0]),
                        "contact_y": float(positions[int(contact_node)][1]),
                        "contact_z": float(positions[int(contact_node)][2]),
                        "contact_image_x": float(contact_position[0]),
                        "contact_image_y": float(contact_position[1]),
                        "contact_image_z": float(contact_position[2]),
                        "angle_a": angle,
                        "angle_b": None,
                        "torsion_a": torsion,
                        "torsion_b": None,
                    }
                    structure_hits += 1
                    results.append(result)

            if structure_hits > 0:
                structures_with_hits += 1
            if progress_every and candidate_pos % int(progress_every) == 0:
                print(
                    f"  Progress: {candidate_pos}/{len(candidate_indices)}, contacts: {len(results)}",
                    flush=True,
                )

        return {
            "summary": {
                "mode": "mol2_contact",
                "search_backend": "c_anchor_v3",
                "graph_cache": str(self.compact_cache.cache_dir),
                "fragment_mol2": str(Path(fragment_mol2).expanduser().resolve()),
                "scanned_structures": len(candidate_indices),
                "structures_with_hits": structures_with_hits,
                "contacts_found": len(results),
                "radius_max": float(radius_max),
                "contact_scope": contact_scope,
                "contact_elements": sorted(_symbol(value) for value in contact_numbers) if not wildcard_contact else ["*"],
                "strict_bonds": bool(strict_bonds),
                "strict_atom_types": bool(strict_atom_types),
                "allow_hydrogen_wildcards": bool(allow_hydrogen_wildcards),
            },
            "results": results,
        }


def _formats_list(output_settings: Optional[Dict[str, Any]]) -> List[str]:
    output_settings = output_settings or {}
    formats = output_settings.get("formats", ["json"])
    if isinstance(formats, str):
        formats = [formats]
    if isinstance(formats, dict):
        formats = list(formats)
    return [str(value).strip().lower() for value in formats if str(value).strip()]


def _clean_generated_output(output_dir: Path, basename: str) -> None:
    targets = [
        output_dir / f"{basename}.json",
        output_dir / f"{basename}.csv",
        output_dir / f"{basename}_summary.json",
        output_dir / f"{basename}_report.html",
        output_dir / f"{basename}_plots",
        output_dir / "cif",
        output_dir / "poscar",
    ]
    for path in targets:
        if path.is_dir():
            shutil.rmtree(path)
        elif path.exists():
            path.unlink()


def _safe_name(value: Any, fallback: str) -> str:
    text = str(value or fallback)
    safe = "".join(ch if ch.isalnum() else "_" for ch in text)[:80]
    return safe or fallback


def _export_hit_structures(
    db: Any,
    payload: Dict[str, Any],
    output_dir: Path,
    export_cif: bool = True,
    export_poscar: bool = True,
) -> Dict[int, Dict[str, str]]:
    results = payload.get("results", [])
    structure_ids = sorted({int(item["structure_id"]) for item in results if "structure_id" in item})
    exported: Dict[int, Dict[str, str]] = {}
    if not structure_ids:
        return exported
    cif_dir = output_dir / "cif"
    poscar_dir = output_dir / "poscar"
    if export_cif:
        cif_dir.mkdir(parents=True, exist_ok=True)
    if export_poscar:
        poscar_dir.mkdir(parents=True, exist_ok=True)

    for structure_id in structure_ids:
        atoms, metadata = db.get_structure(int(structure_id))
        if atoms is None or metadata is None:
            continue
        refcode = str(metadata.get("refcode") or f"id_{structure_id}")
        safe = _safe_name(refcode, f"id_{structure_id}")
        paths: Dict[str, str] = {}
        if export_cif:
            cif_path = cif_dir / f"{safe}.cif"
            db.export_to_cif(atoms, metadata, str(cif_path), include_symmetry=True)
            paths["cif_path"] = str(cif_path)
        if export_poscar:
            poscar_path = poscar_dir / f"{safe}_POSCAR"
            db.export_to_poscar(atoms, metadata, str(poscar_path))
            paths["poscar_path"] = str(poscar_path)
        exported[int(structure_id)] = paths

    for item in results:
        paths = exported.get(int(item.get("structure_id", -1)))
        if paths:
            item.update(paths)
    return exported


def _numeric_summary(values: Sequence[float]) -> Dict[str, Optional[float]]:
    series = pd.Series(values, dtype="float64").dropna()
    if series.empty:
        return {
            "count": 0,
            "min": None,
            "max": None,
            "mean": None,
            "median": None,
            "std": None,
            "q25": None,
            "q75": None,
        }
    return {
        "count": int(series.count()),
        "min": float(series.min()),
        "max": float(series.max()),
        "mean": float(series.mean()),
        "median": float(series.median()),
        "std": float(series.std(ddof=0)),
        "q25": float(series.quantile(0.25)),
        "q75": float(series.quantile(0.75)),
    }


def _result_statistics(payload: Dict[str, Any]) -> Dict[str, Any]:
    df = pd.DataFrame(payload.get("results", []))
    if df.empty:
        return {
            "distance": _numeric_summary([]),
            "angle": _numeric_summary([]),
            "torsion": _numeric_summary([]),
            "by_contact_element": {},
        }
    distance_values = df["distance"].dropna().astype(float).tolist() if "distance" in df else []
    angle_values: List[float] = []
    torsion_values: List[float] = []
    for field in ("angle_a", "angle_b"):
        if field in df:
            angle_values.extend(df[field].dropna().astype(float).tolist())
    for field in ("torsion_a", "torsion_b"):
        if field in df:
            torsion_values.extend(df[field].dropna().astype(float).tolist())

    by_contact_element: Dict[str, Any] = {}
    if "contact_element" in df:
        for contact_element, group in df.groupby("contact_element", dropna=False):
            group_angles: List[float] = []
            group_torsions: List[float] = []
            for field in ("angle_a", "angle_b"):
                if field in group:
                    group_angles.extend(group[field].dropna().astype(float).tolist())
            for field in ("torsion_a", "torsion_b"):
                if field in group:
                    group_torsions.extend(group[field].dropna().astype(float).tolist())
            by_contact_element[str(contact_element)] = {
                "distance": _numeric_summary(
                    group["distance"].dropna().astype(float).tolist()
                    if "distance" in group
                    else []
                ),
                "angle": _numeric_summary(group_angles),
                "torsion": _numeric_summary(group_torsions),
            }
    return {
        "distance": _numeric_summary(distance_values),
        "angle": _numeric_summary(angle_values),
        "torsion": _numeric_summary(torsion_values),
        "by_contact_element": by_contact_element,
    }


def _safe_label(label: str) -> str:
    safe = "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in str(label).strip())
    return safe.strip("_") or "unknown"


def _plot_density(values: Sequence[float], title: str, xlabel: str, output_path: Path, bins: int, dpi: int) -> None:
    if not MATPLOTLIB_AVAILABLE:
        return
    series = np.asarray(values, dtype=float)
    series = series[~np.isnan(series)]
    fig, ax = plt.subplots(figsize=(7, 4.5))
    if series.size == 0:
        ax.text(0.5, 0.5, "No data", ha="center", va="center", fontsize=14)
        ax.set_yticks([])
        ax.set_xticks([])
    else:
        hist_bins = max(5, min(int(bins), max(10, min(100, int(series.size)))))
        hist_range = None
        if series.size == 1 or np.allclose(series, series[0]):
            center = float(series[0])
            spread = max(abs(center) * 0.05, 0.5)
            hist_range = (center - spread, center + spread)
            hist_bins = min(hist_bins, 10)
        hist, edges, _ = ax.hist(
            series,
            bins=hist_bins,
            range=hist_range,
            density=True,
            alpha=0.45,
            color="#4c78a8",
            edgecolor="white",
            linewidth=0.8,
        )
        centers = 0.5 * (edges[:-1] + edges[1:])
        ax.plot(centers, hist, color="#1f3b63", linewidth=1.8)
        ax.set_ylabel("Probability density")
        ax.grid(alpha=0.25, linewidth=0.5)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    fig.tight_layout()
    fig.savefig(output_path, dpi=dpi)
    plt.close(fig)


def _build_plots(payload: Dict[str, Any], output_dir: Path, basename: str, bins: int, dpi: int) -> List[str]:
    if not MATPLOTLIB_AVAILABLE:
        return []
    df = pd.DataFrame(payload.get("results", []))
    if df.empty or "contact_element" not in df:
        return []
    plots_dir = output_dir / f"{basename}_plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    generated: List[str] = []

    by_element_dir = plots_dir / "by_contact_element"
    by_element_dir.mkdir(parents=True, exist_ok=True)
    for contact_element, group in df.groupby("contact_element", dropna=False):
        contact_element = str(contact_element)
        safe = _safe_label(contact_element)

        angle_values: List[float] = []
        torsion_values: List[float] = []
        for field in ("angle_a", "angle_b"):
            if field in group:
                angle_values.extend(group[field].dropna().astype(float).tolist())
        for field in ("torsion_a", "torsion_b"):
            if field in group:
                torsion_values.extend(group[field].dropna().astype(float).tolist())

        plot_specs = [
            (
                by_element_dir / f"distance_contact_{safe}.png",
                group["distance"].dropna().astype(float).tolist() if "distance" in group else [],
                f"Anchor-Contact Distance Density: {contact_element}",
                "Anchor...contact distance, Angstrom",
            ),
            (
                by_element_dir / f"angle_contact_{safe}.png",
                angle_values,
                f"Angle Density: {contact_element}",
                "Angle, degrees",
            ),
            (
                by_element_dir / f"torsion_contact_{safe}.png",
                torsion_values,
                f"Torsion Density: {contact_element}",
                "Torsion angle, degrees",
            ),
        ]
        for path, series, title, xlabel in plot_specs:
            if not series:
                continue
            _plot_density(series, title, xlabel, path, bins=bins, dpi=dpi)
            if path.exists():
                generated.append(str(path))
    return generated


def _write_html_report(output_path: Path, payload: Dict[str, Any], summary: Dict[str, Any], plot_files: Sequence[str]) -> None:
    df = pd.DataFrame(payload.get("results", []))

    def path_link(value: Any) -> str:
        if value is None or (isinstance(value, float) and np.isnan(value)):
            return ""
        text = str(value)
        if not text:
            return ""
        relpath = os.path.relpath(text, output_path.parent)
        return f'<a href="{html.escape(relpath)}">{html.escape(Path(text).name)}</a>'

    formatters = {
        column: path_link
        for column in ("cif_path", "poscar_path")
        if column in df.columns
    }
    table_html = (
        "<p>No results.</p>"
        if df.empty
        else df.head(500).to_html(index=False, escape=False, formatters=formatters)
    )
    plots_html = "\n".join(
        f'<figure><img src="{html.escape(os.path.relpath(path, output_path.parent))}" '
        f'alt="{html.escape(Path(path).name)}"><figcaption>{html.escape(Path(path).name)}</figcaption></figure>'
        for path in plot_files
    )
    summary_items = "\n".join(
        f"<tr><th>{html.escape(str(key))}</th><td>{html.escape(str(value))}</td></tr>"
        for key, value in summary.items()
        if key not in {"written_files", "plot_files", "statistics"}
    )
    document = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>USPEX compact graph query report</title>
  <style>
    body {{ font-family: Arial, sans-serif; margin: 28px; line-height: 1.45; }}
    table {{ border-collapse: collapse; width: 100%; margin: 16px 0; font-size: 13px; }}
    th, td {{ border: 1px solid #ddd; padding: 6px 8px; text-align: left; }}
    th {{ background: #f2f2f2; }}
    img {{ max-width: 720px; width: 100%; height: auto; border: 1px solid #ddd; }}
    figure {{ margin: 18px 0; }}
  </style>
</head>
<body>
  <h1>USPEX compact graph query report</h1>
  <h2>Summary</h2>
  <table>{summary_items}</table>
  <h2>Distributions</h2>
  {plots_html or "<p>No plots generated.</p>"}
  <h2>Results</h2>
  {table_html}
</body>
</html>
"""
    with output_path.open("w", encoding="utf-8") as fh:
        fh.write(document)


def write_payload(
    payload: Dict[str, Any],
    output_dir: Path,
    basename: str = "mol2_contact_results_cfast",
    output_settings: Optional[Dict[str, Any]] = None,
    db: Any = None,
) -> Dict[str, Any]:
    output_settings = output_settings or {}
    output_dir = Path(output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    if output_settings.get("clean_output", False):
        _clean_generated_output(output_dir, basename)

    formats = _formats_list(output_settings)
    if not formats:
        formats = ["json"]

    written_files: List[str] = []
    exported_structures: Dict[int, Dict[str, str]] = {}
    export_structures = bool(output_settings.get("export_structures", False)) or any(
        fmt in formats for fmt in ("cif", "poscar")
    )
    if export_structures and db is not None:
        explicit_structure_formats = {"cif", "poscar"}.intersection(formats)
        export_cif = "cif" in explicit_structure_formats if explicit_structure_formats else True
        export_poscar = "poscar" in explicit_structure_formats if explicit_structure_formats else True
        exported_structures = _export_hit_structures(
            db,
            payload,
            output_dir=output_dir,
            export_cif=export_cif,
            export_poscar=export_poscar,
        )

    json_path = output_dir / f"{basename}.json"
    csv_path = output_dir / f"{basename}.csv"
    summary_path = output_dir / f"{basename}_summary.json"

    if "json" in formats:
        with json_path.open("w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=2, ensure_ascii=False, default=str)
        written_files.append(str(json_path))
    if "csv" in formats:
        df = pd.DataFrame(payload.get("results", []))
        if df.empty:
            with csv_path.open("w", encoding="utf-8", newline="") as fh:
                writer = csv.writer(fh)
                writer.writerow(["no_results"])
        else:
            df.to_csv(csv_path, index=False)
        written_files.append(str(csv_path))

    plots_cfg = output_settings.get("plots", {})
    if not isinstance(plots_cfg, dict):
        plots_cfg = {}
    plot_files: List[str] = []
    if plots_cfg.get("enabled", False):
        plot_files = _build_plots(
            payload,
            output_dir=output_dir,
            basename=basename,
            bins=int(plots_cfg.get("bins", 50)),
            dpi=int(plots_cfg.get("dpi", 160)),
        )
        written_files.extend(plot_files)

    summary = dict(payload.get("summary", {}))
    summary["statistics"] = _result_statistics(payload)
    summary["exported_structures"] = len(exported_structures)
    summary["summary_file"] = str(summary_path)
    summary["written_files"] = written_files + [str(summary_path)]
    summary["plot_files"] = plot_files

    if output_settings.get("html_report", False):
        html_path = output_dir / f"{basename}_report.html"
        _write_html_report(html_path, payload, summary, plot_files)
        summary["written_files"].append(str(html_path))

    with summary_path.open("w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2, ensure_ascii=False, default=str)
    return summary
