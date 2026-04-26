from __future__ import annotations

from collections import Counter, defaultdict, deque
from dataclasses import dataclass
import re
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

import numpy as np

from .codes import atomic_number_from_symbol, bond_order_code, generic_edge_code, hybridization_code


MOL2_DUMMY_TOKENS = {"*", "X", "DU", "DUM", "LP"}


@dataclass(frozen=True)
class Mol2Atom:
    atom_id: int
    name: str
    atom_type: str
    atomic_number: int
    hybridization: str
    is_dummy: bool


@dataclass(frozen=True)
class Mol2Bond:
    atom_a: int
    atom_b: int
    bond_order: str


@dataclass(frozen=True)
class QueryGraph:
    original_node_ids: Tuple[int, ...]
    atomic_numbers: np.ndarray
    hybridization: np.ndarray
    adj_offsets: np.ndarray
    adj_neighbors: np.ndarray
    adj_orders: np.ndarray
    match_order: np.ndarray
    anchor_index: int
    prev_index: Optional[int]
    prev2_index: Optional[int]
    contact_elements_from_dummy: Tuple[int, ...]

    @property
    def required_element_counts(self) -> Counter:
        return Counter(int(value) for value in self.atomic_numbers.tolist() if int(value) > 0)

    @property
    def generic_edge_codes(self) -> Set[int]:
        result: Set[int] = set()
        for node in range(len(self.atomic_numbers)):
            start = int(self.adj_offsets[node])
            stop = int(self.adj_offsets[node + 1])
            for pos in range(start, stop):
                neighbor = int(self.adj_neighbors[pos])
                if neighbor <= node:
                    continue
                left = int(self.atomic_numbers[node])
                right = int(self.atomic_numbers[neighbor])
                if left and right:
                    result.add(generic_edge_code(left, right))
        return result


def _mol2_atom_element(name: str, atom_type: str) -> Tuple[str, bool]:
    atom_type_text = str(atom_type).strip()
    name_text = str(name).strip()
    head = atom_type_text.split(".", 1)[0].strip()
    token = head or re.sub(r"[^A-Za-z*]", "", name_text)
    token_upper = token.upper()
    if token_upper in MOL2_DUMMY_TOKENS or name_text.upper() in MOL2_DUMMY_TOKENS:
        return "*", True
    if token.startswith("#") and token[1:].isdigit():
        return token, False
    if len(token) >= 2 and atomic_number_from_symbol(token[:2]) > 0:
        return token[:2], False
    if token[:1] and atomic_number_from_symbol(token[:1]) > 0:
        return token[:1], False
    return token or "*", token_upper in MOL2_DUMMY_TOKENS


def _mol2_atom_hybridization(atom_type: str) -> str:
    text = str(atom_type).strip().lower()
    if "." not in text:
        return ""
    suffix = text.split(".", 1)[1]
    if suffix == "1":
        return "sp"
    if suffix in {"2", "cat", "co2", "am", "pl3"}:
        return "sp2"
    if suffix in {"3", "4"}:
        return "sp3"
    if suffix == "ar":
        return "aromatic"
    return ""


def _normalize_mol2_bond_order(raw: str) -> str:
    text = str(raw).strip().lower()
    return {
        "1": "single",
        "2": "double",
        "3": "triple",
        "ar": "aromatic",
        "am": "amide",
        "du": "dummy",
        "un": "unknown",
        "nc": "not_connected",
    }.get(text, text or "unknown")


def parse_mol2(path: Path) -> Tuple[List[Mol2Atom], List[Mol2Bond]]:
    mol2_path = Path(path).expanduser().resolve()
    section = None
    atoms: List[Mol2Atom] = []
    bonds: List[Mol2Bond] = []
    with mol2_path.open("r", encoding="utf-8", errors="ignore") as fh:
        for raw_line in fh:
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            if line.startswith("@<TRIPOS>"):
                section = line[len("@<TRIPOS>") :].strip().upper()
                continue
            parts = line.split()
            if section == "ATOM" and len(parts) >= 6:
                atom_id = int(parts[0])
                name = parts[1]
                atom_type = parts[5]
                element, is_dummy = _mol2_atom_element(name, atom_type)
                atoms.append(
                    Mol2Atom(
                        atom_id=atom_id,
                        name=name,
                        atom_type=atom_type,
                        atomic_number=atomic_number_from_symbol(element),
                        hybridization=_mol2_atom_hybridization(atom_type),
                        is_dummy=is_dummy,
                    )
                )
            elif section == "BOND" and len(parts) >= 4:
                bonds.append(
                    Mol2Bond(
                        atom_a=int(parts[1]),
                        atom_b=int(parts[2]),
                        bond_order=_normalize_mol2_bond_order(parts[3]),
                    )
                )
    if not atoms:
        raise ValueError(f"MOL2 query does not contain atoms: {mol2_path}")
    return atoms, bonds


def _query_match_order(adjacency: Dict[int, Set[int]], atomic_numbers: Dict[int, int], anchor: int) -> List[int]:
    order = [int(anchor)]
    in_order = {int(anchor)}
    while len(order) < len(adjacency):
        candidates = [
            int(node)
            for node in adjacency
            if int(node) not in in_order and any(int(neighbor) in in_order for neighbor in adjacency[node])
        ]
        if not candidates:
            return []
        candidates.sort(
            key=lambda node: (
                -sum(1 for neighbor in adjacency[node] if int(neighbor) in in_order),
                -len(adjacency[node]),
                str(atomic_numbers[node]),
                int(node),
            )
        )
        chosen = int(candidates[0])
        order.append(chosen)
        in_order.add(chosen)
    return order


def mol2_to_query_graph(path: Path) -> QueryGraph:
    atoms, bonds = parse_mol2(Path(path))
    atom_by_id = {atom.atom_id: atom for atom in atoms}
    full_adjacency: Dict[int, Set[int]] = {atom.atom_id: set() for atom in atoms}
    bond_orders: Dict[Tuple[int, int], str] = {}
    for bond in bonds:
        if bond.atom_a not in atom_by_id or bond.atom_b not in atom_by_id:
            continue
        full_adjacency[bond.atom_a].add(bond.atom_b)
        full_adjacency[bond.atom_b].add(bond.atom_a)
        bond_orders[tuple(sorted((bond.atom_a, bond.atom_b)))] = bond.bond_order

    dummy_ids = [atom.atom_id for atom in atoms if atom.is_dummy]
    anchor_original: Optional[int] = None
    contact_elements: Set[int] = set()
    if dummy_ids:
        first_dummy = dummy_ids[0]
        neighbors = sorted(full_adjacency[first_dummy])
        if neighbors:
            anchor_original = int(neighbors[0])
        for dummy_id in dummy_ids:
            dummy_atom = atom_by_id[dummy_id]
            if dummy_atom.atomic_number > 0:
                contact_elements.add(int(dummy_atom.atomic_number))

    kept_ids = tuple(atom.atom_id for atom in atoms if not atom.is_dummy)
    if not kept_ids:
        raise ValueError("MOL2 contact query has no searchable fragment atoms")
    if anchor_original is None:
        anchor_original = int(kept_ids[0])
    if anchor_original not in kept_ids:
        raise ValueError("MOL2 contact query anchor was removed with dummy atoms")

    local_index = {atom_id: idx for idx, atom_id in enumerate(kept_ids)}
    adjacency: Dict[int, Set[int]] = {atom_id: set() for atom_id in kept_ids}
    local_edges: List[Tuple[int, int, int]] = []
    for (atom_a, atom_b), order in bond_orders.items():
        if atom_a not in local_index or atom_b not in local_index:
            continue
        adjacency[atom_a].add(atom_b)
        adjacency[atom_b].add(atom_a)
        local_edges.append((local_index[atom_a], local_index[atom_b], bond_order_code(order)))

    anchor_index = local_index[int(anchor_original)]
    original_neighbors = sorted(int(node) for node in adjacency[int(anchor_original)])
    prev_original = original_neighbors[0] if original_neighbors else None
    prev2_original = None
    if prev_original is not None:
        second_neighbors = sorted(
            int(node)
            for node in adjacency[int(prev_original)]
            if int(node) != int(anchor_original)
        )
        prev2_original = second_neighbors[0] if second_neighbors else None

    atomic_by_original = {
        atom.atom_id: int(atom.atomic_number)
        for atom in atoms
        if atom.atom_id in local_index
    }
    order_original = _query_match_order(adjacency, atomic_by_original, int(anchor_original))
    if not order_original:
        raise ValueError("MOL2 query graph is disconnected after dummy removal")
    match_order = np.asarray([local_index[node] for node in order_original], dtype=np.int32)

    neighbor_lists: Dict[int, List[Tuple[int, int]]] = defaultdict(list)
    for left, right, order in local_edges:
        neighbor_lists[left].append((right, order))
        neighbor_lists[right].append((left, order))
    offsets = np.zeros(len(kept_ids) + 1, dtype=np.int64)
    neighbors: List[int] = []
    orders: List[int] = []
    for node in range(len(kept_ids)):
        entries = sorted(neighbor_lists[node], key=lambda item: item[0])
        offsets[node + 1] = offsets[node] + len(entries)
        for neighbor, order in entries:
            neighbors.append(int(neighbor))
            orders.append(int(order))

    return QueryGraph(
        original_node_ids=kept_ids,
        atomic_numbers=np.asarray([atom_by_id[node].atomic_number for node in kept_ids], dtype=np.uint8),
        hybridization=np.asarray(
            [hybridization_code(atom_by_id[node].hybridization) for node in kept_ids],
            dtype=np.uint8,
        ),
        adj_offsets=offsets,
        adj_neighbors=np.asarray(neighbors, dtype=np.int32),
        adj_orders=np.asarray(orders, dtype=np.uint8),
        match_order=match_order,
        anchor_index=int(anchor_index),
        prev_index=local_index[prev_original] if prev_original is not None else None,
        prev2_index=local_index[prev2_original] if prev2_original is not None else None,
        contact_elements_from_dummy=tuple(sorted(contact_elements)),
    )
