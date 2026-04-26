from __future__ import annotations

import json
import shutil
import tempfile
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Optional, Sequence, Set, Tuple

import numpy as np
import pandas as pd

try:
    import pyarrow.parquet as pq
except ImportError as exc:  # pragma: no cover - import guard
    raise ImportError("pyarrow is required for compact cache builder") from exc

from . import _c_anchor
from .codes import (
    atomic_number_from_symbol,
    bond_order_code,
    generic_edge_code,
    hybridization_code,
    parse_edge_key,
    strict_edge_code,
)


STATUS_OK = 1

SUMMARY_COLUMNS = (
    "structure_id",
    "graph_status",
    "n_atoms",
    "n_edges",
    "element_counts_json",
    "generic_edge_keys_json",
    "edge_keys_json",
)
NODE_COLUMNS = (
    "structure_id",
    "atom_index",
    "atomic_number",
    "component_id",
    "geometry_hybridization",
)
EDGE_COLUMNS = (
    "structure_id",
    "atom_i",
    "atom_j",
    "bond_order",
)


def _json_loads(value, default):
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return default
    if isinstance(value, (list, dict)):
        return value
    text = str(value).strip()
    if not text:
        return default
    try:
        return json.loads(text)
    except Exception:
        return default


def _open_memmap(path: Path, dtype: np.dtype, shape: Tuple[int, ...]) -> np.memmap:
    path.parent.mkdir(parents=True, exist_ok=True)
    return np.lib.format.open_memmap(path, mode="w+", dtype=dtype, shape=shape)


def _flush_array(value: np.ndarray) -> None:
    flush = getattr(value, "flush", None)
    if callable(flush):
        flush()


def _remove_path(path: Path) -> None:
    if path.is_dir():
        shutil.rmtree(path)
    elif path.exists():
        path.unlink()


def _replace_output_dir(build_dir: Path, output_dir: Path) -> None:
    backup_dir: Optional[Path] = None
    if output_dir.exists():
        backup_dir = Path(
            tempfile.mkdtemp(
                prefix=f"{output_dir.name}.old-",
                dir=str(output_dir.parent),
            )
        )
        backup_dir.rmdir()
        output_dir.rename(backup_dir)
    try:
        build_dir.rename(output_dir)
    except Exception:
        if backup_dir is not None and backup_dir.exists() and not output_dir.exists():
            backup_dir.rename(output_dir)
        raise
    else:
        if backup_dir is not None and backup_dir.exists():
            try:
                _remove_path(backup_dir)
            except Exception:
                pass


def _edge_codes_from_keys(keys: Sequence[str], strict: bool) -> np.ndarray:
    codes: Set[int] = set()
    for key in keys:
        left, right, order = parse_edge_key(str(key))
        if left <= 0 or right <= 0:
            continue
        if strict:
            codes.add(strict_edge_code(left, right, order))
        else:
            codes.add(generic_edge_code(left, right))
    return np.asarray(sorted(codes), dtype=np.uint32)


def _normalized_element_counts(raw_counts) -> List[Tuple[int, int]]:
    counts = _json_loads(raw_counts, {})
    normalized: List[Tuple[int, int]] = []
    for symbol, count in counts.items():
        number = atomic_number_from_symbol(symbol)
        if number > 0 and int(count) > 0:
            normalized.append((int(number), int(count)))
    normalized.sort()
    return normalized


def _summary_sparse_payload(
    element_counts_json,
    generic_edge_keys_json,
    edge_keys_json,
) -> Tuple[List[Tuple[int, int]], np.ndarray, np.ndarray]:
    normalized_counts = _normalized_element_counts(element_counts_json)
    generic_codes = _edge_codes_from_keys(
        _json_loads(generic_edge_keys_json, []),
        strict=False,
    )
    strict_codes = _edge_codes_from_keys(
        _json_loads(edge_keys_json, []),
        strict=True,
    )
    return normalized_counts, generic_codes, strict_codes


def _iter_parquet_batches(
    path: Path,
    columns: Sequence[str],
    batch_rows: int,
) -> Iterator[pd.DataFrame]:
    parquet_file = pq.ParquetFile(path)
    for batch in parquet_file.iter_batches(columns=list(columns), batch_size=int(batch_rows)):
        yield batch.to_pandas()


def _iter_structure_frames(
    path: Path,
    columns: Sequence[str],
    batch_rows: int,
) -> Iterator[Tuple[int, pd.DataFrame]]:
    carry: Optional[pd.DataFrame] = None
    for batch_df in _iter_parquet_batches(path, columns, batch_rows):
        if carry is not None and not carry.empty:
            batch_df = pd.concat([carry, batch_df], ignore_index=True)
            carry = None
        if batch_df.empty:
            continue

        last_structure_id = int(batch_df["structure_id"].iloc[-1])
        complete_mask = batch_df["structure_id"].astype("int64") != last_structure_id
        complete_df = batch_df.loc[complete_mask]
        carry = batch_df.loc[~complete_mask].reset_index(drop=True)

        if not complete_df.empty:
            for structure_id, group in complete_df.groupby("structure_id", sort=False):
                yield int(structure_id), group.reset_index(drop=True)

    if carry is not None and not carry.empty:
        structure_id = int(carry["structure_id"].iloc[0])
        if not (carry["structure_id"].astype("int64") == structure_id).all():
            raise ValueError("structure_id rows are not contiguous in parquet source")
        yield structure_id, carry.reset_index(drop=True)


def _next_or_none(iterator: Iterator[Tuple[int, pd.DataFrame]]) -> Optional[Tuple[int, pd.DataFrame]]:
    try:
        return next(iterator)
    except StopIteration:
        return None


def _analyze_summary(
    summary_path: Path,
    batch_rows: int,
    progress_every: int,
) -> Dict[str, int]:
    parquet_file = pq.ParquetFile(summary_path)
    structure_count = int(parquet_file.metadata.num_rows)
    seen = 0
    total_nodes = 0
    total_edges = 0
    total_adjacency = 0
    total_element_entries = 0
    total_generic_codes = 0
    total_strict_codes = 0

    for batch_df in _iter_parquet_batches(summary_path, SUMMARY_COLUMNS, batch_rows):
        for row in batch_df.itertuples(index=False):
            seen += 1
            is_ok = str(row.graph_status) == "ok"
            n_nodes = int(row.n_atoms) if is_ok else 0
            n_edges = int(row.n_edges) if is_ok else 0
            total_nodes += n_nodes
            total_edges += n_edges
            total_adjacency += 2 * n_edges
            normalized_counts, generic_codes, strict_codes = _summary_sparse_payload(
                row.element_counts_json,
                row.generic_edge_keys_json,
                row.edge_keys_json,
            )
            total_element_entries += len(normalized_counts)
            total_generic_codes += int(len(generic_codes))
            total_strict_codes += int(len(strict_codes))
        if progress_every > 0 and seen % progress_every == 0:
            print(f"Summary analysis: {seen}/{structure_count} structures", flush=True)

    if seen != structure_count:
        raise ValueError(f"Summary row count mismatch: {seen} != {structure_count}")

    return {
        "structure_count": structure_count,
        "node_count": int(total_nodes),
        "source_edge_count": int(total_edges),
        "adjacency_entry_count": int(total_adjacency),
        "element_entry_count": int(total_element_entries),
        "generic_edge_code_count": int(total_generic_codes),
        "strict_edge_code_count": int(total_strict_codes),
    }


def _allocate_output_arrays(build_dir: Path, summary_stats: Dict[str, int]) -> Dict[str, np.ndarray]:
    structure_count = int(summary_stats["structure_count"])
    node_count = int(summary_stats["node_count"])
    adjacency_entry_count = int(summary_stats["adjacency_entry_count"])
    element_entry_count = int(summary_stats["element_entry_count"])
    generic_edge_code_count = int(summary_stats["generic_edge_code_count"])
    strict_edge_code_count = int(summary_stats["strict_edge_code_count"])

    arrays = {
        "structure_ids": _open_memmap(build_dir / "structure_ids.npy", np.int64, (structure_count,)),
        "status": _open_memmap(build_dir / "status.npy", np.uint8, (structure_count,)),
        "node_offsets": _open_memmap(build_dir / "node_offsets.npy", np.int64, (structure_count + 1,)),
        "node_atomic_numbers": _open_memmap(build_dir / "node_atomic_numbers.npy", np.uint8, (node_count,)),
        "node_component_ids": _open_memmap(build_dir / "node_component_ids.npy", np.int32, (node_count,)),
        "node_hybridization": _open_memmap(build_dir / "node_hybridization.npy", np.uint8, (node_count,)),
        "adjacency_offsets": _open_memmap(build_dir / "adjacency_offsets.npy", np.int64, (node_count + 1,)),
        "adjacency_neighbors": _open_memmap(build_dir / "adjacency_neighbors.npy", np.int32, (adjacency_entry_count,)),
        "adjacency_orders": _open_memmap(build_dir / "adjacency_orders.npy", np.uint8, (adjacency_entry_count,)),
        "element_offsets": _open_memmap(build_dir / "element_offsets.npy", np.int64, (structure_count + 1,)),
        "element_numbers": _open_memmap(build_dir / "element_numbers.npy", np.uint8, (element_entry_count,)),
        "element_counts": _open_memmap(build_dir / "element_counts.npy", np.uint16, (element_entry_count,)),
        "generic_edge_offsets": _open_memmap(build_dir / "generic_edge_offsets.npy", np.int64, (structure_count + 1,)),
        "generic_edge_codes": _open_memmap(build_dir / "generic_edge_codes.npy", np.uint32, (generic_edge_code_count,)),
        "strict_edge_offsets": _open_memmap(build_dir / "strict_edge_offsets.npy", np.int64, (structure_count + 1,)),
        "strict_edge_codes": _open_memmap(build_dir / "strict_edge_codes.npy", np.uint32, (strict_edge_code_count,)),
    }
    arrays["node_offsets"][0] = 0
    arrays["adjacency_offsets"][0] = 0
    arrays["element_offsets"][0] = 0
    arrays["generic_edge_offsets"][0] = 0
    arrays["strict_edge_offsets"][0] = 0
    return arrays


def _write_summary_arrays(
    summary_path: Path,
    arrays: Dict[str, np.ndarray],
    structure_count: int,
    batch_rows: int,
    progress_every: int,
) -> np.ndarray:
    edge_counts = np.zeros(structure_count, dtype=np.int64)
    structure_idx = 0
    node_cursor = 0
    element_cursor = 0
    generic_cursor = 0
    strict_cursor = 0

    for batch_df in _iter_parquet_batches(summary_path, SUMMARY_COLUMNS, batch_rows):
        for row in batch_df.itertuples(index=False):
            structure_id = int(row.structure_id)
            is_ok = str(row.graph_status) == "ok"
            n_nodes = int(row.n_atoms) if is_ok else 0
            n_edges = int(row.n_edges) if is_ok else 0

            arrays["structure_ids"][structure_idx] = structure_id
            arrays["status"][structure_idx] = STATUS_OK if is_ok else 0

            node_cursor += n_nodes
            arrays["node_offsets"][structure_idx + 1] = node_cursor
            edge_counts[structure_idx] = n_edges

            normalized_counts, generic_codes, strict_codes = _summary_sparse_payload(
                row.element_counts_json,
                row.generic_edge_keys_json,
                row.edge_keys_json,
            )

            for number, count in normalized_counts:
                arrays["element_numbers"][element_cursor] = int(number)
                arrays["element_counts"][element_cursor] = min(int(count), np.iinfo(np.uint16).max)
                element_cursor += 1
            arrays["element_offsets"][structure_idx + 1] = element_cursor

            if len(generic_codes):
                arrays["generic_edge_codes"][generic_cursor : generic_cursor + len(generic_codes)] = generic_codes
                generic_cursor += len(generic_codes)
            arrays["generic_edge_offsets"][structure_idx + 1] = generic_cursor

            if len(strict_codes):
                arrays["strict_edge_codes"][strict_cursor : strict_cursor + len(strict_codes)] = strict_codes
                strict_cursor += len(strict_codes)
            arrays["strict_edge_offsets"][structure_idx + 1] = strict_cursor

            structure_idx += 1

        if progress_every > 0 and structure_idx % progress_every == 0:
            print(f"Summary write: {structure_idx}/{structure_count} structures", flush=True)

    if structure_idx != structure_count:
        raise ValueError(f"Summary row count mismatch while writing arrays: {structure_idx} != {structure_count}")
    return edge_counts


def _prepare_structure_payload(
    structure_id: int,
    expected_node_count: int,
    expected_edge_count: int,
    node_group: Optional[pd.DataFrame],
    edge_group: Optional[pd.DataFrame],
) -> Dict[str, Any]:
    if expected_node_count < 0 or expected_edge_count < 0:
        raise ValueError("expected node/edge counts must be non-negative")

    if expected_node_count == 0:
        atomic_numbers = np.empty(0, dtype=np.uint8)
        component_ids = np.empty(0, dtype=np.int32)
        geometry_hybridization: List[Any] = []
    else:
        if node_group is None or node_group.empty:
            raise ValueError(f"Missing graph_nodes rows for structure_id={structure_id}")
        group = node_group.sort_values("atom_index").reset_index(drop=True)
        if len(group) != expected_node_count:
            raise ValueError(
                f"graph_nodes row count mismatch for structure_id={structure_id}: "
                f"{len(group)} != {expected_node_count}"
            )
        atom_indices = group["atom_index"].astype("int64").to_numpy()
        expected = np.arange(expected_node_count, dtype=np.int64)
        if not np.array_equal(atom_indices, expected):
            raise ValueError(
                f"graph_nodes for structure_id={structure_id} are not dense atom indexes 0..n-1"
            )
        atomic_numbers = group["atomic_number"].astype("uint8").to_numpy(copy=False)
        component_ids = group["component_id"].fillna(-1).astype("int32").to_numpy(copy=False)
        geometry_hybridization = group["geometry_hybridization"].tolist()

    if expected_edge_count == 0:
        src = np.empty(0, dtype=np.int32)
        dst = np.empty(0, dtype=np.int32)
        bond_orders: List[Any] = []
    else:
        if edge_group is None or edge_group.empty:
            raise ValueError(f"Missing graph_edges rows for structure_id={structure_id}")
        if len(edge_group) != expected_edge_count:
            raise ValueError(
                f"graph_edges row count mismatch for structure_id={structure_id}: "
                f"{len(edge_group)} != {expected_edge_count}"
            )
        src = edge_group["atom_i"].astype("int32").to_numpy(copy=False)
        dst = edge_group["atom_j"].astype("int32").to_numpy(copy=False)
        bond_orders = edge_group["bond_order"].tolist()

    return {
        "structure_id": int(structure_id),
        "atomic_numbers": np.asarray(atomic_numbers, dtype=np.uint8),
        "component_ids": np.asarray(component_ids, dtype=np.int32),
        "geometry_hybridization": list(geometry_hybridization),
        "src": np.asarray(src, dtype=np.int32),
        "dst": np.asarray(dst, dtype=np.int32),
        "bond_orders": list(bond_orders),
    }


def _build_compact_chunk(structures: Sequence[Dict[str, Any]]) -> Dict[str, np.ndarray]:
    node_atomic_numbers_parts: List[np.ndarray] = []
    node_component_ids_parts: List[np.ndarray] = []
    node_hybridization_parts: List[np.ndarray] = []
    degree_parts: List[np.ndarray] = []
    adjacency_neighbors_parts: List[np.ndarray] = []
    adjacency_orders_parts: List[np.ndarray] = []
    total_edges = 0

    for structure in structures:
        atomic_numbers = np.asarray(structure["atomic_numbers"], dtype=np.uint8)
        component_ids = np.asarray(structure["component_ids"], dtype=np.int32)
        hybridization = np.asarray(
            [hybridization_code(value) for value in structure["geometry_hybridization"]],
            dtype=np.uint8,
        )
        src = np.asarray(structure["src"], dtype=np.int32)
        dst = np.asarray(structure["dst"], dtype=np.int32)
        orders = np.asarray(
            [bond_order_code(value) for value in structure["bond_orders"]],
            dtype=np.uint8,
        )

        n_nodes = int(len(atomic_numbers))
        if len(component_ids) != n_nodes or len(hybridization) != n_nodes:
            raise ValueError(f"Node payload length mismatch for structure_id={structure['structure_id']}")
        if len(src) != len(dst) or len(src) != len(orders):
            raise ValueError(f"Edge payload length mismatch for structure_id={structure['structure_id']}")

        valid = (src >= 0) & (src < n_nodes) & (dst >= 0) & (dst < n_nodes)
        src = src[valid]
        dst = dst[valid]
        orders = orders[valid]

        degree = np.zeros(n_nodes, dtype=np.int64)
        if len(src):
            np.add.at(degree, src, 1)
            np.add.at(degree, dst, 1)

        local_offsets = np.concatenate(([0], np.cumsum(degree, dtype=np.int64)))
        local_neighbors = np.empty(int(local_offsets[-1]), dtype=np.int32)
        local_orders = np.empty(int(local_offsets[-1]), dtype=np.uint8)
        fill = local_offsets[:-1].copy()
        for left, right, order in zip(src.tolist(), dst.tolist(), orders.tolist()):
            pos = int(fill[left])
            local_neighbors[pos] = int(right)
            local_orders[pos] = int(order)
            fill[left] += 1
            pos = int(fill[right])
            local_neighbors[pos] = int(left)
            local_orders[pos] = int(order)
            fill[right] += 1

        node_atomic_numbers_parts.append(atomic_numbers)
        node_component_ids_parts.append(component_ids)
        node_hybridization_parts.append(hybridization)
        degree_parts.append(degree)
        adjacency_neighbors_parts.append(local_neighbors)
        adjacency_orders_parts.append(local_orders)
        total_edges += int(len(src))

    return {
        "node_atomic_numbers": (
            np.concatenate(node_atomic_numbers_parts).astype(np.uint8, copy=False)
            if node_atomic_numbers_parts
            else np.empty(0, dtype=np.uint8)
        ),
        "node_component_ids": (
            np.concatenate(node_component_ids_parts).astype(np.int32, copy=False)
            if node_component_ids_parts
            else np.empty(0, dtype=np.int32)
        ),
        "node_hybridization": (
            np.concatenate(node_hybridization_parts).astype(np.uint8, copy=False)
            if node_hybridization_parts
            else np.empty(0, dtype=np.uint8)
        ),
        "degrees": (
            np.concatenate(degree_parts).astype(np.int64, copy=False)
            if degree_parts
            else np.empty(0, dtype=np.int64)
        ),
        "adjacency_neighbors": (
            np.concatenate(adjacency_neighbors_parts).astype(np.int32, copy=False)
            if adjacency_neighbors_parts
            else np.empty(0, dtype=np.int32)
        ),
        "adjacency_orders": (
            np.concatenate(adjacency_orders_parts).astype(np.uint8, copy=False)
            if adjacency_orders_parts
            else np.empty(0, dtype=np.uint8)
        ),
        "source_edge_count": np.asarray(total_edges, dtype=np.int64),
    }


def _write_compact_chunk(
    arrays: Dict[str, np.ndarray],
    chunk_meta: Sequence[Dict[str, int]],
    chunk_result: Dict[str, np.ndarray],
    adjacency_cursor: int,
) -> Tuple[int, int]:
    node_cursor = 0
    local_adjacency_cursor = 0
    total_edges = int(chunk_result["source_edge_count"])

    for meta in chunk_meta:
        node_start = int(meta["node_start"])
        node_stop = int(meta["node_stop"])
        n_nodes = node_stop - node_start
        next_node_cursor = node_cursor + n_nodes

        arrays["node_atomic_numbers"][node_start:node_stop] = chunk_result["node_atomic_numbers"][node_cursor:next_node_cursor]
        arrays["node_component_ids"][node_start:node_stop] = chunk_result["node_component_ids"][node_cursor:next_node_cursor]
        arrays["node_hybridization"][node_start:node_stop] = chunk_result["node_hybridization"][node_cursor:next_node_cursor]

        degrees = chunk_result["degrees"][node_cursor:next_node_cursor]
        local_offsets = adjacency_cursor + np.concatenate(([0], np.cumsum(degrees, dtype=np.int64)))
        arrays["adjacency_offsets"][node_start : node_stop + 1] = local_offsets
        adjacency_count = int(local_offsets[-1] - adjacency_cursor)
        next_adjacency_cursor = adjacency_cursor + adjacency_count
        next_local_adjacency_cursor = local_adjacency_cursor + adjacency_count
        arrays["adjacency_neighbors"][adjacency_cursor:next_adjacency_cursor] = (
            chunk_result["adjacency_neighbors"][local_adjacency_cursor:next_local_adjacency_cursor]
        )
        arrays["adjacency_orders"][adjacency_cursor:next_adjacency_cursor] = (
            chunk_result["adjacency_orders"][local_adjacency_cursor:next_local_adjacency_cursor]
        )

        adjacency_cursor = next_adjacency_cursor
        local_adjacency_cursor = next_local_adjacency_cursor
        node_cursor = next_node_cursor

    if node_cursor != len(chunk_result["node_atomic_numbers"]):
        raise ValueError("Chunk node write mismatch while building compact cache")
    if local_adjacency_cursor != len(chunk_result["adjacency_neighbors"]):
        raise ValueError("Chunk adjacency write mismatch while building compact cache")
    return adjacency_cursor, total_edges


def build_compact_cache(
    source_cache_dir: Path,
    output_dir: Path,
    overwrite: bool = False,
    batch_structures: int = 256,
    workers: int = 1,
    progress_every: int = 1000,
    parquet_batch_rows: int = 200_000,
) -> Path:
    source_cache_dir = Path(source_cache_dir).expanduser().resolve()
    output_dir = Path(output_dir).expanduser().resolve()
    manifest_path = output_dir / "manifest.json"
    if manifest_path.exists() and not overwrite:
        return output_dir

    summary_path = source_cache_dir / "graph_summary.parquet"
    nodes_path = source_cache_dir / "graph_nodes.parquet"
    edges_path = source_cache_dir / "graph_edges.parquet"
    missing = [path for path in (summary_path, nodes_path, edges_path) if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing source graph cache files: " + ", ".join(str(path) for path in missing))

    batch_structures_int = int(batch_structures or 0)
    if batch_structures_int < 1:
        batch_structures_int = 1
    workers_int = int(workers or 0)
    if workers_int < 1:
        workers_int = 1
    progress_every_int = int(progress_every or 0)
    if progress_every_int < 0:
        progress_every_int = 0
    parquet_batch_rows_int = int(parquet_batch_rows or 0)
    if parquet_batch_rows_int < 1:
        parquet_batch_rows_int = 1

    output_dir.parent.mkdir(parents=True, exist_ok=True)
    build_dir = Path(
        tempfile.mkdtemp(
            prefix=f"{output_dir.name}.tmp-build-",
            dir=str(output_dir.parent),
        )
    )
    arrays: Dict[str, np.ndarray] = {}
    build_completed = False

    try:
        print(f"Compact cache source: {source_cache_dir}", flush=True)
        print("Analyzing summary parquet...", flush=True)
        summary_stats = _analyze_summary(
            summary_path=summary_path,
            batch_rows=parquet_batch_rows_int,
            progress_every=progress_every_int,
        )
        print(
            "Allocating compact cache arrays: "
            f"structures={summary_stats['structure_count']}, "
            f"nodes={summary_stats['node_count']}, "
            f"adjacency={summary_stats['adjacency_entry_count']}",
            flush=True,
        )
        arrays = _allocate_output_arrays(build_dir, summary_stats)
        edge_counts = _write_summary_arrays(
            summary_path=summary_path,
            arrays=arrays,
            structure_count=int(summary_stats["structure_count"]),
            batch_rows=parquet_batch_rows_int,
            progress_every=progress_every_int,
        )

        structure_ids = arrays["structure_ids"]
        status = arrays["status"]
        node_offsets = arrays["node_offsets"]
        n_structures = int(summary_stats["structure_count"])
        ok_structure_count = int(np.count_nonzero(status))
        total_nodes = int(summary_stats["node_count"])
        processed_structures = 0
        adjacency_cursor = 0
        total_edges = 0

        node_iter = _iter_structure_frames(nodes_path, NODE_COLUMNS, parquet_batch_rows_int)
        edge_iter = _iter_structure_frames(edges_path, EDGE_COLUMNS, parquet_batch_rows_int)
        next_node_group = _next_or_none(node_iter)
        next_edge_group = _next_or_none(edge_iter)

        def iter_structure_chunks() -> Iterator[Tuple[List[Dict[str, int]], List[Dict[str, Any]]]]:
            nonlocal next_edge_group, next_node_group
            chunk_meta: List[Dict[str, int]] = []
            chunk_payload: List[Dict[str, Any]] = []
            for idx in range(n_structures):
                if int(status[idx]) != STATUS_OK:
                    continue
                structure_id = int(structure_ids[idx])
                node_start = int(node_offsets[idx])
                node_stop = int(node_offsets[idx + 1])
                expected_node_count = node_stop - node_start
                expected_edge_count = int(edge_counts[idx])

                node_group = None
                if expected_node_count > 0:
                    if next_node_group is None or int(next_node_group[0]) != structure_id:
                        raise ValueError(f"Missing graph_nodes group for structure_id={structure_id}")
                    node_group = next_node_group[1]
                    next_node_group = _next_or_none(node_iter)

                edge_group = None
                if expected_edge_count > 0:
                    if next_edge_group is None or int(next_edge_group[0]) != structure_id:
                        raise ValueError(f"Missing graph_edges group for structure_id={structure_id}")
                    edge_group = next_edge_group[1]
                    next_edge_group = _next_or_none(edge_iter)

                chunk_meta.append(
                    {
                        "idx": int(idx),
                        "node_start": node_start,
                        "node_stop": node_stop,
                    }
                )
                chunk_payload.append(
                    _prepare_structure_payload(
                        structure_id=structure_id,
                        expected_node_count=expected_node_count,
                        expected_edge_count=expected_edge_count,
                        node_group=node_group,
                        edge_group=edge_group,
                    )
                )
                if len(chunk_payload) >= batch_structures_int:
                    yield chunk_meta, chunk_payload
                    chunk_meta = []
                    chunk_payload = []

            if chunk_payload:
                yield chunk_meta, chunk_payload

            if next_node_group is not None:
                raise ValueError(f"Unused graph_nodes rows remain starting at structure_id={next_node_group[0]}")
            if next_edge_group is not None:
                raise ValueError(f"Unused graph_edges rows remain starting at structure_id={next_edge_group[0]}")

        print(
            f"Streaming nodes/edges into compact cache with workers={workers_int}, batch_structures={batch_structures_int}",
            flush=True,
        )
        if total_nodes == 0:
            arrays["adjacency_offsets"][0] = 0
        if workers_int <= 1:
            for chunk_meta, chunk_payload in iter_structure_chunks():
                chunk_result = _build_compact_chunk(chunk_payload)
                adjacency_cursor, edges_added = _write_compact_chunk(
                    arrays=arrays,
                    chunk_meta=chunk_meta,
                    chunk_result=chunk_result,
                    adjacency_cursor=adjacency_cursor,
                )
                total_edges += int(edges_added)
                processed_structures += len(chunk_meta)
                if progress_every_int > 0 and processed_structures % progress_every_int == 0:
                    print(
                        f"Compact cache build: {processed_structures}/{ok_structure_count} ok structures",
                        flush=True,
                    )
        else:
            pending: List[Tuple[List[Dict[str, int]], Any]] = []
            with ProcessPoolExecutor(max_workers=workers_int) as executor:
                for chunk_meta, chunk_payload in iter_structure_chunks():
                    pending.append((chunk_meta, executor.submit(_build_compact_chunk, chunk_payload)))
                    if len(pending) >= workers_int * 2:
                        ready_meta, future = pending.pop(0)
                        chunk_result = future.result()
                        adjacency_cursor, edges_added = _write_compact_chunk(
                            arrays=arrays,
                            chunk_meta=ready_meta,
                            chunk_result=chunk_result,
                            adjacency_cursor=adjacency_cursor,
                        )
                        total_edges += int(edges_added)
                        processed_structures += len(ready_meta)
                        if progress_every_int > 0 and processed_structures % progress_every_int == 0:
                            print(
                                f"Compact cache build: {processed_structures}/{ok_structure_count} ok structures",
                                flush=True,
                            )
                for ready_meta, future in pending:
                    chunk_result = future.result()
                    adjacency_cursor, edges_added = _write_compact_chunk(
                        arrays=arrays,
                        chunk_meta=ready_meta,
                        chunk_result=chunk_result,
                        adjacency_cursor=adjacency_cursor,
                    )
                    total_edges += int(edges_added)
                    processed_structures += len(ready_meta)
                    if progress_every_int > 0 and processed_structures % progress_every_int == 0:
                        print(
                            f"Compact cache build: {processed_structures}/{ok_structure_count} ok structures",
                            flush=True,
                        )

        if adjacency_cursor != int(summary_stats["adjacency_entry_count"]):
            raise ValueError(
                "Adjacency entry count mismatch while building compact cache: "
                f"{adjacency_cursor} != {summary_stats['adjacency_entry_count']}"
            )
        if total_edges != int(summary_stats["source_edge_count"]):
            raise ValueError(
                "Edge count mismatch while building compact cache: "
                f"{total_edges} != {summary_stats['source_edge_count']}"
            )

        for name, value in arrays.items():
            _flush_array(value)
        for stale_name in ("edge_offsets", "edge_src", "edge_dst", "edge_order"):
            stale_path = build_dir / f"{stale_name}.npy"
            if stale_path.exists():
                stale_path.unlink()

        manifest = {
            "format": "uspex_compact_graph_cache_v3",
            "source_cache_dir": str(source_cache_dir),
            "structure_count": int(summary_stats["structure_count"]),
            "node_count": int(summary_stats["node_count"]),
            "source_edge_count": int(total_edges),
            "adjacency_entry_count": int(adjacency_cursor),
            "settings": {
                "batch_structures": int(batch_structures_int),
                "workers": int(workers_int),
                "progress_every": int(progress_every_int),
                "parquet_batch_rows": int(parquet_batch_rows_int),
            },
            "files": {name: f"{name}.npy" for name in arrays},
        }
        with (build_dir / "manifest.json").open("w", encoding="utf-8") as fh:
            json.dump(manifest, fh, indent=2, ensure_ascii=False)
        _replace_output_dir(build_dir, output_dir)
        build_completed = True
        print(f"Compact cache ready: {output_dir}", flush=True)
        return output_dir
    finally:
        for value in arrays.values():
            _flush_array(value)
        if not build_completed and build_dir.exists():
            shutil.rmtree(build_dir, ignore_errors=True)


class CompactGraphCache:
    def __init__(self, cache_dir: Path):
        self.cache_dir = Path(cache_dir).expanduser().resolve()
        manifest_path = self.cache_dir / "manifest.json"
        if not manifest_path.exists():
            raise FileNotFoundError(f"compact cache manifest not found: {manifest_path}")
        self.manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        for name in self.manifest["files"]:
            setattr(self, name, np.load(self.cache_dir / f"{name}.npy", mmap_mode="r", allow_pickle=False))

    def candidate_indices(
        self,
        required_counts: Dict[int, int],
        required_edge_codes: Iterable[int],
        allow_hydrogen_wildcards: bool,
        strict_bonds: bool,
    ) -> List[int]:
        counts = dict(required_counts)
        if allow_hydrogen_wildcards:
            counts.pop(1, None)
        req_numbers = np.asarray(sorted(counts), dtype=np.uint8)
        req_counts = np.asarray([counts[int(number)] for number in req_numbers], dtype=np.uint16)
        req_edges = np.asarray(sorted(set(int(value) for value in required_edge_codes)), dtype=np.uint32)
        edge_offsets = self.strict_edge_offsets if strict_bonds else self.generic_edge_offsets
        edge_codes = self.strict_edge_codes if strict_bonds else self.generic_edge_codes
        return [
            int(value)
            for value in _c_anchor.prefilter_candidates(
                self.status,
                self.element_offsets,
                self.element_numbers,
                self.element_counts,
                edge_offsets,
                edge_codes,
                req_numbers,
                req_counts,
                req_edges,
            )
        ]

    def structure_id_for_cache_index(self, cache_index: int) -> int:
        return int(self.structure_ids[int(cache_index)])

    def node_range(self, cache_index: int) -> Tuple[int, int]:
        idx = int(cache_index)
        return int(self.node_offsets[idx]), int(self.node_offsets[idx + 1])
