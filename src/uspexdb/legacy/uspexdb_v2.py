#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
USPEX db-v2 export and search tool.

Scalable reader for the directory layout produced by convert_uspex_db.py:
  manifest.json
  metadata/*.parquet or metadata/metadata.parquet
  indexes/structure_lookup/*.parquet or indexes/structure_lookup.parquet
  indexes/refcode_lookup/*.parquet (optional)
  structures/shard-XXXXXX/*.npy
"""

from __future__ import annotations

import argparse
from bisect import bisect_right
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
import csv
from dataclasses import dataclass
import html
import json
import os
import re
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Set, Tuple, Union

import numpy as np
import pandas as pd

try:
    import pyarrow as pa
    import pyarrow.dataset as ds
    import pyarrow.parquet as pq
except ImportError as exc:  # pragma: no cover - import guard
    raise ImportError("pyarrow is required for db-v2 reader") from exc

try:
    from ase import Atoms
    from ase.data import atomic_numbers as ASE_ATOMIC_NUMBERS
    from ase.data import chemical_symbols as ASE_CHEMICAL_SYMBOLS
    from ase.data import covalent_radii as ASE_COVALENT_RADII
    from ase.geometry import find_mic
    from ase.io import write
    from ase.neighborlist import NeighborList, natural_cutoffs

    ASE_AVAILABLE = True
except ImportError:
    ASE_AVAILABLE = False
    ASE_ATOMIC_NUMBERS = {}
    ASE_CHEMICAL_SYMBOLS = []
    ASE_COVALENT_RADII = []
    print("⚠️ ASE not found: pip install ase")

try:
    from pymatgen.core import Structure
    from pymatgen.io.cif import CifWriter

    PYMATGEN_AVAILABLE = True
except ImportError:
    PYMATGEN_AVAILABLE = False

try:
    from pymatgen.core.local_env import CrystalNN

    PYMATGEN_GRAPH_AVAILABLE = True
except ImportError:
    try:
        from pymatgen.analysis.local_env import CrystalNN

        PYMATGEN_GRAPH_AVAILABLE = True
    except ImportError:
        PYMATGEN_GRAPH_AVAILABLE = False

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False

try:
    from rdkit import Chem
    from rdkit import RDLogger
    from rdkit.Chem import rdDetermineBonds

    RDKIT_AVAILABLE = True
    RDLogger.DisableLog("rdApp.*")
except ImportError:
    RDKIT_AVAILABLE = False

try:
    from openbabel import openbabel as OB

    OPENBABEL_AVAILABLE = True
except ImportError:
    OPENBABEL_AVAILABLE = False

try:
    import networkx as nx

    NETWORKX_AVAILABLE = True
except ImportError:
    NETWORKX_AVAILABLE = False


NULL_STRINGS = {"", "-1", "None"}
METADATA_ALIASES = {
    "global_idx": "legacy_global_idx",
    "hdf5_path": "source_hdf5_path",
    "db_path": "source_db_path",
}
USER_METADATA_EXCLUDED_FIELDS = {
    "hdf5_path",
    "db_path",
    "source_hdf5_path",
    "source_db_path",
}
UNBRACKETED_SMARTS_TOKENS = (
    "Cl",
    "Br",
    "Si",
    "Na",
    "Li",
    "Ca",
    "Al",
    "Mg",
    "Zn",
    "Cu",
    "Fe",
    "Ni",
    "Co",
    "Mn",
    "Sn",
    "Ag",
    "Au",
    "Hg",
    "Pb",
    "Pt",
    "Pd",
    "Ir",
    "Ru",
    "Rh",
    "Os",
    "Ti",
    "Cr",
    "Mo",
    "Se",
    "As",
    "cl",
    "br",
)
SINGLE_CHAR_SMARTS_ATOMS = set("*BCNOFPSIbcnops")
AROMATIC_SMARTS_ATOM_SYMBOLS = {
    "b": "B",
    "c": "C",
    "n": "N",
    "o": "O",
    "p": "P",
    "s": "S",
}
MOL2_DUMMY_TOKENS = {"*", "X", "DU", "DUM", "LP"}
MOL2_BOND_ORDER_MAP = {
    "1": "single",
    "2": "double",
    "3": "triple",
    "ar": "aromatic",
    "am": "amide",
    "du": "dummy",
    "un": "unknown",
    "nc": "not_connected",
}
RDKIT_BOND_ORDER_MAP = {}
if RDKIT_AVAILABLE:
    RDKIT_BOND_ORDER_MAP = {
        Chem.BondType.SINGLE: "single",
        Chem.BondType.DOUBLE: "double",
        Chem.BondType.TRIPLE: "triple",
        Chem.BondType.AROMATIC: "aromatic",
    }
METAL_SYMBOLS = {
    "Li", "Be", "Na", "Mg", "Al", "K", "Ca", "Sc", "Ti", "V", "Cr", "Mn",
    "Fe", "Co", "Ni", "Cu", "Zn", "Ga", "Rb", "Sr", "Y", "Zr", "Nb", "Mo",
    "Tc", "Ru", "Rh", "Pd", "Ag", "Cd", "In", "Sn", "Cs", "Ba", "La", "Ce",
    "Pr", "Nd", "Pm", "Sm", "Eu", "Gd", "Tb", "Dy", "Ho", "Er", "Tm", "Yb",
    "Lu", "Hf", "Ta", "W", "Re", "Os", "Ir", "Pt", "Au", "Hg", "Tl", "Pb",
    "Bi", "Fr", "Ra", "Ac", "Th", "Pa", "U", "Np", "Pu",
}
METAL_ATOMIC_NUMBERS = {
    int(ASE_ATOMIC_NUMBERS[symbol])
    for symbol in METAL_SYMBOLS
    if ASE_AVAILABLE and symbol in ASE_ATOMIC_NUMBERS
}


def _normalize_scalar(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    return value


def build_default_config(database_path: str = "uspex_db_v2") -> Dict[str, Any]:
    return {
        "database": {"path": database_path},
        "search": {
            "name_keyword": None,
            "elements": None,
            "year": None,
            "spacegroup": None,
            "temperature": None,
            "r_factor": None,
            "n_atoms": None,
            "refcode": None,
            "smarts_fragment": None,
        },
        "export": {
            "output_dir": "export_output",
            "limit": 100,
            "formats": {
                "cif": {"output_subdir": "cif_files", "include_symmetry": True}
            },
        },
        "logging": {
            "verbose": True,
            "save_stats": True,
            "stats_file": "export_stats.json",
        },
    }


def build_default_query(database_path: str = "uspex_db_v2") -> Dict[str, Any]:
    return {
        "database": {"path": database_path},
        "query": {
            "mode": "mol2_contact",
            "search_backend": "fast_anchor",
            "fragment_mol2": "fragment_contact.mol2",
            "fragment_a": "[O:1]-[H:2]",
            "fragment_b": "[O:1]",
            "radius_max": 3.0,
            "contact_elements": ["O", "N", "S", "Cl", "Br", "F"],
            "contact_scope": "intermolecular",
            "structure_ids": [],
            "refcodes": [],
            "max_structures": 100,
            "covalent_scale": 1.15,
            "strict_bonds": False,
            "strict_atom_types": True,
            "allow_hydrogen_wildcards": True,
            "progress_every": 100,
        },
        "graph_cache": {
            "enabled": True,
            "path": "indexes/graph_cache",
            "build_if_missing": True,
            "rebuild": False,
            "max_structures": 100,
            "max_atoms": 1000,
            "covalent_scale": 1.15,
            "min_nonbonded_distance": 0.6,
            "flush_every": 250,
            "workers": 1,
            "worker_chunk_size": 10,
            "skip_extended_networks": True,
            "component_filter_backend": "pymatgen",
            "bond_order_backend": "rdkit",
        },
        "output": {
            "output_dir": "query_output",
            "basename": "mol2_contact_results",
            "formats": ["json", "csv"],
            "export_structures": True,
            "clean_output": True,
            "html_report": True,
            "plots": {
                "enabled": True,
                "bins": 50,
                "dpi": 160,
            },
        },
        "logging": {
            "verbose": True,
            "save_summary": True,
        },
    }


def create_config_template(config_path: Union[str, Path], database_path: str = "uspex_db_v2") -> Path:
    config_file = Path(config_path).expanduser().resolve()
    config_file.parent.mkdir(parents=True, exist_ok=True)
    default_config = build_default_config(database_path)
    with config_file.open("w", encoding="utf-8") as fh:
        json.dump(default_config, fh, indent=4, ensure_ascii=False)
    return config_file


def create_query_template(query_path: Union[str, Path], database_path: str = "uspex_db_v2") -> Path:
    query_file = Path(query_path).expanduser().resolve()
    query_file.parent.mkdir(parents=True, exist_ok=True)
    default_query = build_default_query(database_path)
    with query_file.open("w", encoding="utf-8") as fh:
        json.dump(default_query, fh, indent=4, ensure_ascii=False)
    return query_file


@dataclass(frozen=True)
class FragmentSpec:
    label: str
    smarts: str
    pattern: Any
    atom_count: int
    exact_atomic_number_requirements: Tuple[Tuple[int, int], ...]
    mapped_query_indices: Tuple[int, ...]
    mapped_numbers: Tuple[int, ...]
    anchor_query_index: int
    angle_query_index: Optional[int]
    torsion_query_index: Optional[int]


@dataclass
class FragmentMatch:
    query_atoms: Tuple[int, ...]
    global_atoms: Tuple[int, ...]
    anchor_global: int
    anchor_local: int
    prev_global: Optional[int]
    prev_local: Optional[int]
    prev2_global: Optional[int]
    prev2_local: Optional[int]


@dataclass
class MoleculeComponent:
    search_component_id: int
    molecule_id: int
    global_indices: np.ndarray
    positions_unwrapped: np.ndarray
    atomic_number_counts: Dict[int, int]
    rdkit_mol: Any


@dataclass(frozen=True)
class Mol2AtomRecord:
    atom_id: int
    name: str
    x: float
    y: float
    z: float
    atom_type: str
    hybridization: Optional[str]
    element: str
    is_dummy: bool


@dataclass(frozen=True)
class Mol2BondRecord:
    bond_id: int
    atom_a: int
    atom_b: int
    bond_type: str
    bond_order: str


@dataclass(frozen=True)
class Mol2Query:
    atoms: Tuple[Mol2AtomRecord, ...]
    bonds: Tuple[Mol2BondRecord, ...]


def _is_metal_atomic_number(atomic_number: int) -> bool:
    return int(atomic_number) in METAL_ATOMIC_NUMBERS


def _is_metal_symbol(symbol: str) -> bool:
    return str(symbol).strip().capitalize() in METAL_SYMBOLS


def _symbol_from_atomic_number(atomic_number: int) -> str:
    atomic_number = int(atomic_number)
    if ASE_AVAILABLE and 0 < atomic_number < len(ASE_CHEMICAL_SYMBOLS):
        return str(ASE_CHEMICAL_SYMBOLS[atomic_number])
    return str(atomic_number)


def _json_dumps_sorted(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True)


def _json_loads_or_empty(value: Any, default: Any) -> Any:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return default
    if isinstance(value, (list, dict, set, tuple)):
        return value
    text = str(value).strip()
    if not text:
        return default
    try:
        return json.loads(text)
    except Exception:
        return default


def _optional_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    if isinstance(value, float) and np.isnan(value):
        return None
    text = str(value).strip()
    if not text or text.lower() in {"none", "null", "nan"}:
        return None
    return float(text)


def _optional_int_arg(value: Any) -> Optional[int]:
    if value is None:
        return None
    text = str(value).strip().lower()
    if not text or text in {"none", "null", "all", "full"}:
        return None
    return int(text)


def _normalize_bond_order(raw_bond_type: str) -> str:
    text = str(raw_bond_type).strip().lower()
    return MOL2_BOND_ORDER_MAP.get(text, text or "unknown")


def _mol2_atom_element(name: str, atom_type: str) -> Tuple[str, bool]:
    atom_type_text = str(atom_type).strip()
    name_text = str(name).strip()
    head = atom_type_text.split(".", 1)[0].strip()
    token = head or re.sub(r"[^A-Za-z*]", "", name_text)
    token_upper = token.upper()
    if token_upper in MOL2_DUMMY_TOKENS or name_text.upper() in MOL2_DUMMY_TOKENS:
        return "*", True
    if token.startswith("#") and token[1:].isdigit():
        symbol = _symbol_from_atomic_number(int(token[1:]))
        return symbol, False
    if len(token) >= 2 and token[:2].capitalize() in ASE_ATOMIC_NUMBERS:
        return token[:2].capitalize(), False
    if token[:1].upper() in ASE_ATOMIC_NUMBERS:
        return token[:1].upper(), False
    return token.capitalize() if token else "*", token_upper in MOL2_DUMMY_TOKENS


def _mol2_atom_hybridization(atom_type: str) -> Optional[str]:
    text = str(atom_type).strip().lower()
    if "." not in text:
        return None
    suffix = text.split(".", 1)[1]
    if suffix in {"1"}:
        return "sp"
    if suffix in {"2", "cat", "co2", "am", "pl3"}:
        return "sp2"
    if suffix in {"3", "4"}:
        return "sp3"
    if suffix in {"ar"}:
        return "aromatic"
    return None


def parse_mol2_file(mol2_path: Union[str, Path]) -> Mol2Query:
    path = Path(mol2_path).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"MOL2 query file not found: {path}")

    section = None
    atoms: List[Mol2AtomRecord] = []
    bonds: List[Mol2BondRecord] = []
    with path.open("r", encoding="utf-8", errors="ignore") as fh:
        for raw_line in fh:
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            if line.startswith("@<TRIPOS>"):
                section = line[len("@<TRIPOS>"):].strip().upper()
                continue
            parts = line.split()
            if section == "ATOM":
                if len(parts) < 6:
                    continue
                atom_id = int(parts[0])
                name = parts[1]
                x, y, z = float(parts[2]), float(parts[3]), float(parts[4])
                atom_type = parts[5]
                element, is_dummy = _mol2_atom_element(name, atom_type)
                atoms.append(
                    Mol2AtomRecord(
                        atom_id=atom_id,
                        name=name,
                        x=x,
                        y=y,
                        z=z,
                        atom_type=atom_type,
                        hybridization=_mol2_atom_hybridization(atom_type),
                        element=element,
                        is_dummy=is_dummy,
                    )
                )
            elif section == "BOND":
                if len(parts) < 4:
                    continue
                bond_type = parts[3]
                bonds.append(
                    Mol2BondRecord(
                        bond_id=int(parts[0]),
                        atom_a=int(parts[1]),
                        atom_b=int(parts[2]),
                        bond_type=bond_type,
                        bond_order=_normalize_bond_order(bond_type),
                    )
                )
    if not atoms:
        raise ValueError(f"MOL2 query does not contain atoms: {path}")
    return Mol2Query(atoms=tuple(atoms), bonds=tuple(bonds))


def mol2_query_to_graph(query: Mol2Query, drop_dummy: bool = False) -> Any:
    if not NETWORKX_AVAILABLE:
        raise ImportError("networkx is required for MOL2 graph search")
    graph = nx.Graph()
    atom_ids = {atom.atom_id for atom in query.atoms}
    for atom in query.atoms:
        if drop_dummy and atom.is_dummy:
            continue
        graph.add_node(
            int(atom.atom_id),
            atom_id=int(atom.atom_id),
            element=atom.element,
            atom_type=atom.atom_type,
            hybridization=atom.hybridization,
            is_dummy=bool(atom.is_dummy),
            is_metal=_is_metal_symbol(atom.element),
            label=atom.name,
            x=float(atom.x),
            y=float(atom.y),
            z=float(atom.z),
        )
    for bond in query.bonds:
        if bond.atom_a not in atom_ids or bond.atom_b not in atom_ids:
            continue
        if drop_dummy and (
            bond.atom_a not in graph.nodes or bond.atom_b not in graph.nodes
        ):
            continue
        graph.add_edge(
            int(bond.atom_a),
            int(bond.atom_b),
            bond_type=bond.bond_type,
            bond_order=bond.bond_order,
        )
    return graph


def _edge_key(element_a: str, element_b: str, bond_order: str) -> str:
    left, right = sorted((str(element_a), str(element_b)))
    return f"{left}-{right}:{bond_order}"


def graph_fingerprint(graph: Any, ignore_dummy: bool = True) -> Dict[str, Any]:
    element_counts: Counter[str] = Counter()
    node_keys: Set[str] = set()
    edge_keys: Set[str] = set()
    generic_edge_keys: Set[str] = set()
    has_metal = False
    has_aromatic = False

    for _, attrs in graph.nodes(data=True):
        if ignore_dummy and attrs.get("is_dummy"):
            continue
        element = str(attrs.get("element", ""))
        if not element or element == "*":
            continue
        element_counts[element] += 1
        node_keys.add(element)
        if attrs.get("is_metal") or _is_metal_symbol(element):
            has_metal = True

    for a, b, attrs in graph.edges(data=True):
        node_a = graph.nodes[a]
        node_b = graph.nodes[b]
        if ignore_dummy and (node_a.get("is_dummy") or node_b.get("is_dummy")):
            continue
        element_a = str(node_a.get("element", ""))
        element_b = str(node_b.get("element", ""))
        if "*" in (element_a, element_b):
            continue
        bond_order = str(attrs.get("bond_order", "unknown"))
        if bond_order == "aromatic":
            has_aromatic = True
        edge_keys.add(_edge_key(element_a, element_b, bond_order))
        generic_edge_keys.add(_edge_key(element_a, element_b, "any"))

    return {
        "element_counts": dict(sorted(element_counts.items())),
        "node_keys": sorted(node_keys),
        "edge_keys": sorted(edge_keys),
        "generic_edge_keys": sorted(generic_edge_keys),
        "has_metal": has_metal,
        "has_aromatic": has_aromatic,
    }


def _parse_smarts_atom_order(smarts: str) -> List[Tuple[int, Optional[int], str]]:
    atoms: List[Tuple[int, Optional[int], str]] = []
    i = 0
    atom_index = 0
    while i < len(smarts):
        ch = smarts[i]
        if ch == "[":
            j = i + 1
            while j < len(smarts) and smarts[j] != "]":
                j += 1
            if j >= len(smarts):
                raise ValueError(f"Unclosed SMARTS atom in {smarts!r}")
            token = smarts[i:j + 1]
            map_match = re.search(r":(\d+)", token)
            map_number = int(map_match.group(1)) if map_match else None
            atoms.append((atom_index, map_number, token))
            atom_index += 1
            i = j + 1
            continue
        matched = False
        for token in UNBRACKETED_SMARTS_TOKENS:
            if smarts.startswith(token, i):
                atoms.append((atom_index, None, token))
                atom_index += 1
                i += len(token)
                matched = True
                break
        if matched:
            continue
        if ch in SINGLE_CHAR_SMARTS_ATOMS:
            atoms.append((atom_index, None, ch))
            atom_index += 1
            i += 1
            continue
        if ch == "%" and i + 2 < len(smarts):
            i += 3
            continue
        i += 1
    return atoms


def _atomic_number_counts(numbers: np.ndarray) -> Dict[int, int]:
    if numbers.size == 0:
        return {}
    unique, counts = np.unique(numbers.astype(int), return_counts=True)
    return {int(atomic_number): int(count) for atomic_number, count in zip(unique, counts)}


def _atomic_number_requirements_satisfied(
    available_counts: Dict[int, int],
    required_counts: Sequence[Tuple[int, int]],
) -> bool:
    for atomic_number, required_count in required_counts:
        if int(available_counts.get(int(atomic_number), 0)) < int(required_count):
            return False
    return True


def _merge_atomic_number_requirements(
    requirements_a: Sequence[Tuple[int, int]],
    requirements_b: Sequence[Tuple[int, int]],
) -> Tuple[Tuple[int, int], ...]:
    merged: Counter[int] = Counter()
    for atomic_number, count in requirements_a:
        merged[int(atomic_number)] += int(count)
    for atomic_number, count in requirements_b:
        merged[int(atomic_number)] += int(count)
    return tuple(sorted((int(atomic_number), int(count)) for atomic_number, count in merged.items()))


def _smarts_token_to_atomic_number(token: str) -> Optional[int]:
    text = str(token).strip()
    if not text:
        return None

    if text.startswith("[") and text.endswith("]"):
        body = text[1:-1].strip()
        if not body or any(ch in body for ch in ",;!$*&~"):
            return None
        body = re.sub(r":\d+", "", body)
        body = re.sub(r"^\d+", "", body)

        atomic_number_match = re.match(r"^#(\d+)", body)
        if atomic_number_match:
            return int(atomic_number_match.group(1))

        symbol_match = re.match(r"^([A-Z][a-z]?|[bcnops])", body)
        if not symbol_match:
            return None
        symbol = symbol_match.group(1)
    else:
        if text == "*":
            return None
        symbol = text

    if symbol in AROMATIC_SMARTS_ATOM_SYMBOLS:
        symbol = AROMATIC_SMARTS_ATOM_SYMBOLS[symbol]
    elif symbol and symbol[0].islower():
        symbol = symbol.capitalize()

    atomic_number = ASE_ATOMIC_NUMBERS.get(symbol) if ASE_AVAILABLE else None
    if atomic_number is None:
        return None
    return int(atomic_number)


def _compile_exact_atomic_number_requirements(
    atom_tokens: Sequence[Tuple[int, Optional[int], str]],
) -> Tuple[Tuple[int, int], ...]:
    requirements: Counter[int] = Counter()
    for _, _, token in atom_tokens:
        atomic_number = _smarts_token_to_atomic_number(token)
        if atomic_number is None:
            continue
        requirements[int(atomic_number)] += 1
    return tuple(
        sorted((int(atomic_number), int(count)) for atomic_number, count in requirements.items())
    )


def compile_fragment(label: str, smarts: str) -> FragmentSpec:
    if not RDKIT_AVAILABLE:
        raise ImportError("RDKit is required for SMARTS search")
    atom_tokens = _parse_smarts_atom_order(smarts)
    if not atom_tokens:
        raise ValueError(f"SMARTS {smarts!r} does not contain atoms")
    exact_atomic_number_requirements = _compile_exact_atomic_number_requirements(atom_tokens)
    mapped = [(query_idx, map_no) for query_idx, map_no, _ in atom_tokens if map_no is not None]
    if mapped:
        mapped_sorted = sorted(mapped, key=lambda item: item[1])
        mapped_query_indices = tuple(item[0] for item in mapped_sorted)
        mapped_numbers = tuple(item[1] for item in mapped_sorted)
        anchor_query_index = mapped_query_indices[-1]
        angle_query_index = mapped_query_indices[-2] if len(mapped_query_indices) >= 2 else None
        torsion_query_index = mapped_query_indices[-3] if len(mapped_query_indices) >= 3 else None
    else:
        mapped_query_indices = tuple()
        mapped_numbers = tuple()
        anchor_query_index = atom_tokens[-1][0]
        angle_query_index = atom_tokens[-2][0] if len(atom_tokens) >= 2 else None
        torsion_query_index = atom_tokens[-3][0] if len(atom_tokens) >= 3 else None
    pattern = Chem.MolFromSmarts(smarts)
    if pattern is None:
        raise ValueError(f"Invalid SMARTS: {smarts!r}")
    return FragmentSpec(
        label=label,
        smarts=smarts,
        pattern=pattern,
        atom_count=len(atom_tokens),
        exact_atomic_number_requirements=exact_atomic_number_requirements,
        mapped_query_indices=mapped_query_indices,
        mapped_numbers=mapped_numbers,
        anchor_query_index=anchor_query_index,
        angle_query_index=angle_query_index,
        torsion_query_index=torsion_query_index,
    )


def _angle_degrees(p1: np.ndarray, p2: np.ndarray, p3: np.ndarray) -> Optional[float]:
    v1 = np.asarray(p1, dtype=float) - np.asarray(p2, dtype=float)
    v2 = np.asarray(p3, dtype=float) - np.asarray(p2, dtype=float)
    n1 = np.linalg.norm(v1)
    n2 = np.linalg.norm(v2)
    if n1 < 1e-12 or n2 < 1e-12:
        return None
    cosine = np.dot(v1, v2) / (n1 * n2)
    cosine = float(np.clip(cosine, -1.0, 1.0))
    return float(np.degrees(np.arccos(cosine)))


def _dihedral_degrees(
    p1: np.ndarray,
    p2: np.ndarray,
    p3: np.ndarray,
    p4: np.ndarray,
) -> Optional[float]:
    b0 = np.asarray(p1, dtype=float) - np.asarray(p2, dtype=float)
    b1 = np.asarray(p3, dtype=float) - np.asarray(p2, dtype=float)
    b2 = np.asarray(p4, dtype=float) - np.asarray(p3, dtype=float)
    n_b1 = np.linalg.norm(b1)
    if n_b1 < 1e-12:
        return None
    b1_unit = b1 / n_b1
    v = b0 - np.dot(b0, b1_unit) * b1_unit
    w = b2 - np.dot(b2, b1_unit) * b1_unit
    n_v = np.linalg.norm(v)
    n_w = np.linalg.norm(w)
    if n_v < 1e-12 or n_w < 1e-12:
        return None
    x = np.dot(v, w)
    y = np.dot(np.cross(b1_unit, v), w)
    return float(np.degrees(np.arctan2(y, x)))


def _build_component_molecule(numbers: np.ndarray, positions: np.ndarray) -> Any:
    if not RDKIT_AVAILABLE:
        raise ImportError("RDKit is required for SMARTS search")
    if positions.shape[0] != len(numbers):
        raise ValueError("numbers and positions must describe the same number of atoms")

    periodic_table = Chem.GetPeriodicTable()
    lines = [str(len(numbers)), "generated from db-v2 component"]
    for atomic_number, (x, y, z) in zip(numbers.tolist(), positions.tolist()):
        symbol = periodic_table.GetElementSymbol(int(atomic_number))
        lines.append(f"{symbol} {float(x):.16f} {float(y):.16f} {float(z):.16f}")
    xyz_block = "\n".join(lines)

    mol = Chem.MolFromXYZBlock(xyz_block)
    if mol is None:
        raise ValueError("RDKit failed to create a molecule from XYZ coordinates")

    try:
        rdDetermineBonds.DetermineBonds(mol, charge=0)
    except Exception:
        try:
            rdDetermineBonds.DetermineConnectivity(mol)
            Chem.SanitizeMol(
                mol,
                sanitizeOps=(
                    Chem.SanitizeFlags.SANITIZE_FINDRADICALS
                    | Chem.SanitizeFlags.SANITIZE_SETAROMATICITY
                    | Chem.SanitizeFlags.SANITIZE_SETCONJUGATION
                    | Chem.SanitizeFlags.SANITIZE_SYMMRINGS
                ),
            )
        except Exception as exc:
            raise ValueError(f"RDKit failed to determine bonds for component: {exc}") from exc
    return mol


def _rdkit_has_defined_valence(atomic_number: int) -> bool:
    if not RDKIT_AVAILABLE:
        return True
    periodic_table = Chem.GetPeriodicTable()
    try:
        return int(periodic_table.GetDefaultValence(int(atomic_number))) >= 0
    except Exception:
        return False


def _connected_components_from_edges(
    node_indices: Iterable[int],
    edges: Iterable[Tuple[int, int]],
) -> List[Tuple[int, ...]]:
    adjacency: Dict[int, Set[int]] = {}
    for node in node_indices:
        adjacency[int(node)] = set()
    for edge_a, edge_b in edges:
        a = int(edge_a)
        b = int(edge_b)
        if a not in adjacency or b not in adjacency or a == b:
            continue
        adjacency[a].add(b)
        adjacency[b].add(a)

    components: List[Tuple[int, ...]] = []
    visited: Set[int] = set()
    for start in sorted(adjacency):
        if start in visited:
            continue
        stack = [start]
        visited.add(start)
        component: List[int] = []
        while stack:
            node = stack.pop()
            component.append(node)
            for neighbor in adjacency[node]:
                if neighbor in visited:
                    continue
                visited.add(neighbor)
                stack.append(neighbor)
        components.append(tuple(sorted(component)))
    return components


def _build_single_bond_molecule(
    numbers: np.ndarray,
    edges: Iterable[Tuple[int, int]],
) -> Any:
    if not RDKIT_AVAILABLE:
        raise ImportError("RDKit is required for SMARTS search")
    editable = Chem.RWMol()
    for atomic_number in numbers.tolist():
        editable.AddAtom(Chem.Atom(int(atomic_number)))
    for edge_a, edge_b in sorted({tuple(sorted((int(a), int(b)))) for a, b in edges}):
        if edge_a == edge_b:
            continue
        if editable.GetBondBetweenAtoms(edge_a, edge_b) is not None:
            continue
        editable.AddBond(edge_a, edge_b, Chem.BondType.SINGLE)
    mol = editable.GetMol()
    mol.UpdatePropertyCache(strict=False)
    return mol


def _build_rdkit_fallback_subcomponents(
    numbers: np.ndarray,
    positions: np.ndarray,
    edges: Iterable[Tuple[int, int]],
) -> List[Tuple[np.ndarray, np.ndarray, Any]]:
    local_edges = {tuple(sorted((int(a), int(b)))) for a, b in edges if int(a) != int(b)}
    undefined_valence = {
        local_idx
        for local_idx, atomic_number in enumerate(numbers.tolist())
        if not _rdkit_has_defined_valence(int(atomic_number))
    }

    if not undefined_valence:
        mol = _build_single_bond_molecule(numbers, local_edges)
        return [(np.arange(len(numbers), dtype=int), positions.copy(), mol)]

    organic_nodes = [idx for idx in range(len(numbers)) if idx not in undefined_valence]
    organic_edges = {
        (a, b)
        for a, b in local_edges
        if a not in undefined_valence and b not in undefined_valence
    }

    fallback_components: List[Tuple[np.ndarray, np.ndarray, Any]] = []

    for local_component in _connected_components_from_edges(organic_nodes, organic_edges):
        component_indices = np.array(local_component, dtype=int)
        component_numbers = numbers[component_indices]
        component_positions = positions[component_indices]
        reindex = {int(old_idx): new_idx for new_idx, old_idx in enumerate(component_indices.tolist())}
        component_edges = {
            (reindex[a], reindex[b])
            for a, b in organic_edges
            if a in reindex and b in reindex
        }
        try:
            component_mol = _build_component_molecule(component_numbers, component_positions)
        except Exception:
            component_mol = _build_single_bond_molecule(component_numbers, component_edges)
        fallback_components.append((component_indices, component_positions, component_mol))

    for local_idx in sorted(undefined_valence):
        component_indices = np.array([local_idx], dtype=int)
        component_positions = positions[component_indices]
        component_mol = _build_single_bond_molecule(numbers[component_indices], [])
        fallback_components.append((component_indices, component_positions, component_mol))

    return fallback_components


class DirectoryStructureDB:
    """Interface for working with db-v2 directory database."""

    def __init__(self, db_root: str = "uspex_db_v2", scan_batch_size: int = 65536):
        self.db_root = Path(db_root).expanduser().resolve()
        if not self.db_root.exists():
            raise FileNotFoundError(f"Database directory not found: {self.db_root}")

        self.manifest_path = self.db_root / "manifest.json"
        if not self.manifest_path.exists():
            raise FileNotFoundError(f"manifest.json not found in {self.db_root}")

        with self.manifest_path.open("r", encoding="utf-8") as fh:
            self.manifest = json.load(fh)

        metadata_relpath = self.manifest.get("metadata", {}).get("path", "metadata")
        lookup_relpath = self.manifest.get("indexes", {}).get(
            "structure_lookup", "indexes/structure_lookup"
        )
        refcode_relpath = self.manifest.get("indexes", {}).get("refcode_lookup")

        self.metadata_path = self.db_root / metadata_relpath
        self.lookup_path = self.db_root / lookup_relpath
        self.refcode_lookup_path = self.db_root / refcode_relpath if refcode_relpath else None

        if not self.metadata_path.exists():
            raise FileNotFoundError(f"Metadata parquet dataset not found: {self.metadata_path}")
        if self.lookup_path.exists():
            self.lookup_dataset = ds.dataset(str(self.lookup_path), format="parquet")
        else:
            self.lookup_dataset = None
        if self.refcode_lookup_path and self.refcode_lookup_path.exists():
            self.refcode_dataset = ds.dataset(str(self.refcode_lookup_path), format="parquet")
        else:
            self.refcode_dataset = None
        self.metadata_dataset = ds.dataset(str(self.metadata_path), format="parquet")

        self.scan_batch_size = int(scan_batch_size)
        self.metadata_fields = list(
            self.manifest.get("metadata", {}).get("metadata_fields_from_source", [])
        )
        self.metadata_columns = list(self.manifest.get("metadata", {}).get("columns", []))
        self.metadata_column_set = set(self.metadata_dataset.schema.names)
        self.lookup_column_set = set(self.lookup_dataset.schema.names) if self.lookup_dataset else set()
        self.float_fields = set(
            self.manifest.get("null_policy", {})
            .get("numeric_negative_one_fields", {})
            .get("float_fields", [])
        )
        self.int_fields = set(
            self.manifest.get("null_policy", {})
            .get("numeric_negative_one_fields", {})
            .get("int_fields", [])
        )

        self._metadata_parts = list(self.manifest.get("metadata", {}).get("parts", []))
        self._metadata_part_starts = [
            int(part["min_structure_id"])
            for part in self._metadata_parts
            if "min_structure_id" in part and "max_structure_id" in part
        ]
        lookup_parts = self.manifest.get("indexes", {}).get("structure_lookup_parts", [])
        self._lookup_parts = list(lookup_parts)
        self._lookup_part_starts = [
            int(part["min_structure_id"])
            for part in self._lookup_parts
            if "min_structure_id" in part and "max_structure_id" in part
        ]

        self._shards = sorted(
            self.manifest.get("structures", {}).get("shards", []),
            key=lambda shard: shard.get("min_structure_id", -1),
        )
        self._shard_range_starts = [
            int(shard["min_structure_id"])
            for shard in self._shards
            if "min_structure_id" in shard and "max_structure_id" in shard
        ]
        self._has_shard_ranges = (
            len(self._shards) > 0
            and len(self._shard_range_starts) == len(self._shards)
        )

        self._formula_cache: Dict[str, Set[str]] = {}
        self._formula_count_cache: Dict[str, Dict[str, int]] = {}
        self._metadata_row_cache: Dict[int, Dict[str, Any]] = {}
        self._lookup_row_cache: Dict[int, Dict[str, Any]] = {}
        self._refcode_cache: Dict[str, int] = {}
        self._source_path_cache: Dict[str, int] = {}
        self._shard_cache: Dict[str, Dict[str, np.ndarray]] = {}
        self._stats_cache: Optional[Dict[str, Any]] = None

    def close(self):
        """Explicitly release caches and mapped shard arrays."""
        self._formula_cache.clear()
        self._formula_count_cache.clear()
        self._metadata_row_cache.clear()
        self._lookup_row_cache.clear()
        self._refcode_cache.clear()
        self._source_path_cache.clear()
        self._shard_cache.clear()
        self._stats_cache = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    # ═══════════════════════════════════════════════════════
    # BASIC METHODS
    # ═══════════════════════════════════════════════════════
    def count(self) -> int:
        return int(self.manifest.get("counts", {}).get("structure_count", 0))

    # ═══════════════════════════════════════════════════════
    # HELPERS
    # ═══════════════════════════════════════════════════════
    def _resolve_field_name(self, field: str) -> str:
        return METADATA_ALIASES.get(field, field)

    def _field_exists(self, field: str) -> bool:
        return self._resolve_field_name(field) in self.metadata_column_set

    def _dataset_filter_for_condition(self, field: str, condition) -> Optional[ds.Expression]:
        if not self._field_exists(field):
            return None

        resolved = self._resolve_field_name(field)
        field_expr = ds.field(resolved)

        if isinstance(condition, tuple):
            if len(condition) == 2 and isinstance(condition[0], str):
                op, value = condition
                if op == "<":
                    return field_expr < value
                if op == ">":
                    return field_expr > value
                if op == "<=":
                    return field_expr <= value
                if op == ">=":
                    return field_expr >= value
                if op == "==":
                    return field_expr == value
                if op == "!=":
                    return field_expr != value
            if len(condition) == 2 and isinstance(condition[0], (int, float)):
                return (field_expr >= condition[0]) & (field_expr <= condition[1])
            if len(condition) == 3 and condition[0] == "range":
                return (field_expr >= condition[1]) & (field_expr <= condition[2])

        if isinstance(condition, (int, float)):
            return field_expr == condition

        if isinstance(condition, str):
            if condition.startswith("~") or "*" in condition:
                return None
            text = condition.strip()
            if text:
                return field_expr == text
        return None

    def _scan_metadata_batches(
        self,
        columns: Sequence[str],
        filter_expr: Optional[ds.Expression] = None,
    ) -> Iterable[pd.DataFrame]:
        cols = []
        for column in columns:
            resolved = self._resolve_field_name(column)
            if resolved in self.metadata_column_set and resolved not in cols:
                cols.append(resolved)
        if "structure_id" not in cols:
            cols.insert(0, "structure_id")

        scanner = self.metadata_dataset.scanner(
            columns=cols,
            filter=filter_expr,
            batch_size=self.scan_batch_size,
            use_threads=True,
        )
        for batch in scanner.to_batches():
            yield batch.to_pandas()

    def _find_part_path(
        self,
        parts: List[Dict[str, Any]],
        starts: List[int],
        structure_id: int,
    ) -> Optional[Path]:
        if not parts or not starts:
            return None

        idx = bisect_right(starts, structure_id) - 1
        if idx < 0 or idx >= len(parts):
            return None
        part = parts[idx]
        if int(part.get("min_structure_id", -1)) <= structure_id <= int(
            part.get("max_structure_id", -1)
        ):
            return self.db_root / part["path"]
        return None

    def _extract_single_record(self, table) -> Optional[Dict[str, Any]]:
        if table is None or table.num_rows == 0:
            return None
        record = table.slice(0, 1).to_pylist()[0]
        return {key: _normalize_scalar(value) for key, value in record.items()}

    def _load_metadata_record(self, structure_id: int) -> Optional[Dict[str, Any]]:
        if structure_id in self._metadata_row_cache:
            return dict(self._metadata_row_cache[structure_id])

        part_path = self._find_part_path(self._metadata_parts, self._metadata_part_starts, structure_id)
        if part_path is not None:
            table = pq.read_table(part_path, filters=[("structure_id", "=", structure_id)])
        else:
            table = self.metadata_dataset.to_table(filter=ds.field("structure_id") == structure_id)

        record = self._extract_single_record(table)
        if record is None:
            return None
        self._metadata_row_cache[structure_id] = record
        return dict(record)

    def _load_refcode_record(self, refcode: str) -> Optional[Dict[str, Any]]:
        if not refcode:
            return None
        if refcode in self._refcode_cache:
            return {"structure_id": self._refcode_cache[refcode], "refcode": refcode}

        if self.refcode_dataset is not None:
            table = self.refcode_dataset.to_table(
                columns=["structure_id", "refcode"],
                filter=ds.field("refcode") == refcode,
            )
            record = self._extract_single_record(table)
            if record is not None:
                self._refcode_cache[refcode] = int(record["structure_id"])
                return record

        table = self.metadata_dataset.to_table(
            columns=["structure_id", "refcode"],
            filter=ds.field("refcode") == refcode,
        )
        record = self._extract_single_record(table)
        if record is not None:
            self._refcode_cache[refcode] = int(record["structure_id"])
            return record
        return None

    def _load_source_path_record(self, source_path: str) -> Optional[Dict[str, Any]]:
        if not source_path:
            return None
        if source_path in self._source_path_cache:
            return {"structure_id": self._source_path_cache[source_path]}

        if "source_hdf5_path" not in self.metadata_column_set:
            return None

        table = self.metadata_dataset.to_table(
            columns=["structure_id", "source_hdf5_path"],
            filter=ds.field("source_hdf5_path") == source_path,
        )
        record = self._extract_single_record(table)
        if record is not None:
            self._source_path_cache[source_path] = int(record["structure_id"])
            return record
        return None

    def _find_shard_info(self, structure_id: int) -> Optional[Dict[str, Any]]:
        if not self._has_shard_ranges:
            return None
        idx = bisect_right(self._shard_range_starts, structure_id) - 1
        if idx < 0 or idx >= len(self._shards):
            return None
        shard = self._shards[idx]
        if int(shard["min_structure_id"]) <= structure_id <= int(shard["max_structure_id"]):
            return shard
        return None

    def _get_shard_arrays(self, shard_relpath: str) -> Dict[str, np.ndarray]:
        if shard_relpath not in self._shard_cache:
            shard_dir = self.db_root / shard_relpath
            self._shard_cache[shard_relpath] = {
                "offsets": np.load(shard_dir / "offsets.npy", allow_pickle=False, mmap_mode="r"),
                "n_atoms": np.load(shard_dir / "n_atoms.npy", allow_pickle=False, mmap_mode="r"),
                "numbers": np.load(shard_dir / "numbers.npy", allow_pickle=False, mmap_mode="r"),
                "positions": np.load(shard_dir / "positions.npy", allow_pickle=False, mmap_mode="r"),
                "cell": np.load(shard_dir / "cell.npy", allow_pickle=False, mmap_mode="r"),
                "pbc": np.load(shard_dir / "pbc.npy", allow_pickle=False, mmap_mode="r"),
            }
        return self._shard_cache[shard_relpath]

    def _lookup_from_shard_ranges(self, structure_id: int) -> Optional[Dict[str, Any]]:
        shard_info = self._find_shard_info(structure_id)
        if shard_info is None:
            return None

        shard_relpath = str(shard_info["path"])
        local_idx = int(structure_id - int(shard_info["min_structure_id"]))
        shard = self._get_shard_arrays(shard_relpath)
        offsets = shard["offsets"]
        n_atoms = shard["n_atoms"]
        if local_idx < 0 or local_idx + 1 >= len(offsets):
            return None

        return {
            "structure_id": structure_id,
            "shard_id": int(shard_info["shard_id"]),
            "shard_relpath": shard_relpath,
            "local_idx": local_idx,
            "n_atoms": int(n_atoms[local_idx]),
            "offset_start": int(offsets[local_idx]),
            "offset_stop": int(offsets[local_idx + 1]),
        }

    def _load_lookup_record(self, structure_id: int) -> Optional[Dict[str, Any]]:
        if structure_id in self._lookup_row_cache:
            return dict(self._lookup_row_cache[structure_id])

        record = self._lookup_from_shard_ranges(structure_id)
        if record is None and self.lookup_dataset is not None:
            part_path = self._find_part_path(self._lookup_parts, self._lookup_part_starts, structure_id)
            if part_path is not None:
                table = pq.read_table(part_path, filters=[("structure_id", "=", structure_id)])
            else:
                table = self.lookup_dataset.to_table(
                    filter=ds.field("structure_id") == structure_id
                )
            record = self._extract_single_record(table)

        if record is None:
            return None
        self._lookup_row_cache[structure_id] = record
        return dict(record)

    def _metadata_record_to_dict(self, record: Dict[str, Any]) -> Dict[str, Any]:
        metadata: Dict[str, Any] = {}
        for key, value in record.items():
            if key in USER_METADATA_EXCLUDED_FIELDS:
                continue
            metadata[key] = _normalize_scalar(value)
        if "legacy_global_idx" in metadata:
            metadata["global_idx"] = metadata["legacy_global_idx"]
        return metadata

    def _match_structure_id(self, identifier: Union[int, str]) -> Optional[int]:
        if isinstance(identifier, (int, np.integer)):
            return int(identifier)

        text = str(identifier).strip()
        if not text:
            return None
        if text.isdigit():
            return int(text)

        refcode_record = self._load_refcode_record(text)
        if refcode_record is not None:
            return int(refcode_record["structure_id"])

        source_record = self._load_source_path_record(text)
        if source_record is not None:
            return int(source_record["structure_id"])
        return None

    def _parse_formula_element_counts(self, formula: str) -> Dict[str, int]:
        if not formula or formula in NULL_STRINGS:
            return {}
        if formula in self._formula_count_cache:
            return dict(self._formula_count_cache[formula])

        text = re.sub(r"\s+", "", str(formula))
        length = len(text)
        element_pattern = re.compile(r"[A-Z][a-z]?")

        def read_number(index: int) -> Tuple[int, int]:
            end = index
            while end < length and text[end].isdigit():
                end += 1
            if end == index:
                return 1, index
            return int(text[index:end]), end

        def parse_segment(index: int, stop_char: Optional[str] = None) -> Tuple[Counter[str], int]:
            counts: Counter[str] = Counter()
            i = index
            while i < length:
                ch = text[i]
                if stop_char is not None and ch == stop_char:
                    return counts, i + 1
                if ch in ",.;·":
                    i += 1
                    continue
                if ch in "+-":
                    i += 1
                    continue
                if ch.isdigit():
                    multiplier, next_i = read_number(i)
                    if next_i < length and text[next_i] == "(":
                        nested_counts, next_i = parse_segment(next_i + 1, ")")
                        suffix_multiplier = 1
                        if next_i < length and text[next_i].isdigit():
                            suffix_multiplier, next_i = read_number(next_i)
                        for element, value in nested_counts.items():
                            counts[element] += int(value) * int(multiplier) * int(suffix_multiplier)
                        i = next_i
                        continue
                    i = next_i
                    while i < length and text[i] in "+-":
                        i += 1
                    continue
                if ch == "(":
                    nested_counts, next_i = parse_segment(i + 1, ")")
                    multiplier = 1
                    if next_i < length and text[next_i].isdigit():
                        multiplier, next_i = read_number(next_i)
                    for element, value in nested_counts.items():
                        counts[element] += int(value) * int(multiplier)
                    i = next_i
                    continue
                match = element_pattern.match(text, i)
                if match:
                    symbol = match.group(0)
                    i = match.end()
                    amount = 1
                    if i < length and text[i].isdigit():
                        amount, i = read_number(i)
                    counts[symbol] += int(amount)
                    continue
                i += 1
            return counts, i

        parsed_counts = {
            element: int(value)
            for element, value in parse_segment(0)[0].items()
            if element and len(element) <= 2
        }
        self._formula_count_cache[formula] = parsed_counts
        return dict(parsed_counts)

    def _parse_formula_elements(self, formula: str) -> Set[str]:
        if not formula or formula in NULL_STRINGS:
            return set()
        if formula in self._formula_cache:
            return set(self._formula_cache[formula])
        counts = self._parse_formula_element_counts(formula)
        parsed = set(counts.keys())
        self._formula_cache[formula] = parsed
        return set(parsed)

    def _normalize_element_symbol(self, element: Any) -> str:
        symbol = str(element).strip()
        if not symbol:
            return ""
        if len(symbol) == 1:
            return symbol.upper()
        return symbol[0].upper() + symbol[1:].lower()

    def _parse_element_filter_values(self, values: Any) -> Set[str]:
        if values is None:
            return set()
        if isinstance(values, str):
            return {
                self._normalize_element_symbol(element)
                for element in self._parse_formula_elements(values)
                if self._normalize_element_symbol(element)
            }
        return {
            self._normalize_element_symbol(element)
            for element in values
            if self._normalize_element_symbol(element)
        }

    def _normalize_required_additional_elements(
        self, condition: Any
    ) -> Tuple[Set[str], Set[str]]:
        if isinstance(condition, dict):
            required = self._parse_element_filter_values(
                condition.get("required", condition.get("elements", condition.get("value")))
            )
            additional = self._parse_element_filter_values(condition.get("additional"))
            return required, additional
        return self._parse_element_filter_values(condition), set()

    def _elements_filter_matches(
        self,
        structure_elements: Set[str],
        required_elements: Set[str],
        additional_elements: Set[str],
    ) -> bool:
        if not required_elements:
            return False
        allowed_elements = required_elements | additional_elements
        if not required_elements.issubset(structure_elements):
            return False
        if additional_elements and not structure_elements.intersection(additional_elements):
            return False
        return structure_elements.issubset(allowed_elements)

    def _preferred_formula_field(self) -> Optional[str]:
        for candidate in ("formula_moiety", "formula", "formula_sum"):
            resolved = self._resolve_field_name(candidate)
            if resolved in self.metadata_column_set:
                return resolved
        return None

    def _prefilter_smarts_structure_ids(
        self,
        candidate_ids: Sequence[int],
        min_atoms: int,
        required_atomic_numbers: Sequence[Tuple[int, int]],
    ) -> List[int]:
        candidate_list = [int(value) for value in candidate_ids]
        if not candidate_list:
            return []
        if min_atoms <= 0 and not required_atomic_numbers:
            return candidate_list

        atom_field = None
        for candidate in ("n_atoms_full", "n_atoms"):
            resolved = self._resolve_field_name(candidate)
            if resolved in self.metadata_column_set:
                atom_field = resolved
                break
        formula_field = self._preferred_formula_field()
        if atom_field is None and formula_field is None:
            return candidate_list

        dataset_filter = None
        if atom_field is not None and min_atoms > 0:
            dataset_filter = ds.field(atom_field) >= int(min_atoms)

        required_symbols: Dict[str, int] = {}
        if ASE_AVAILABLE:
            for atomic_number, required_count in required_atomic_numbers:
                atomic_number = int(atomic_number)
                if 0 < atomic_number < len(ASE_CHEMICAL_SYMBOLS):
                    symbol = ASE_CHEMICAL_SYMBOLS[atomic_number]
                    if symbol:
                        required_symbols[symbol] = int(required_count)

        requested_ids = None
        if len(candidate_list) != self.count():
            requested_ids = set(candidate_list)

        columns = ["structure_id"]
        if atom_field is not None:
            columns.append(atom_field)
        if formula_field is not None and formula_field not in columns:
            columns.append(formula_field)

        matched_ids: Set[int] = set()
        for batch_df in self._scan_metadata_batches(columns, dataset_filter):
            if batch_df.empty:
                continue
            if requested_ids is not None:
                batch_df = batch_df.loc[batch_df["structure_id"].isin(requested_ids)]
                if batch_df.empty:
                    continue
            if required_symbols and formula_field is not None:
                formulas = batch_df[formula_field].fillna("").astype(str).str.strip().tolist()
                keep_mask = []
                for formula in formulas:
                    formula_counts = self._parse_formula_element_counts(formula)
                    keep_mask.append(
                        all(
                            int(formula_counts.get(symbol, 0)) >= int(required_count)
                            for symbol, required_count in required_symbols.items()
                        )
                    )
                batch_df = batch_df.loc[pd.Series(keep_mask, index=batch_df.index)]
                if batch_df.empty:
                    continue
            matched_ids.update(int(value) for value in batch_df["structure_id"].tolist())
        return [structure_id for structure_id in candidate_list if structure_id in matched_ids]

    def _apply_condition_to_series(self, series: pd.Series, condition) -> pd.Series:
        if isinstance(condition, list):
            return series.isin(condition)

        if isinstance(condition, tuple):
            if len(condition) == 2 and isinstance(condition[0], str):
                op, value = condition
                valid = series.notna()
                if op == "<":
                    return valid & (series < value)
                if op == ">":
                    return valid & (series > value)
                if op == "<=":
                    return valid & (series <= value)
                if op == ">=":
                    return valid & (series >= value)
                if op == "==":
                    return valid & (series == value)
                if op == "!=":
                    return valid & (series != value)
            if len(condition) == 2 and isinstance(condition[0], (int, float)):
                return series.notna() & (series >= condition[0]) & (series <= condition[1])
            if len(condition) == 3 and condition[0] == "range":
                return series.notna() & (series >= condition[1]) & (series <= condition[2])

        if isinstance(condition, str):
            text = series.fillna("").astype(str).str.strip()
            if condition.startswith("~"):
                pattern = re.compile(condition[1:], re.IGNORECASE)
                return text.apply(lambda value: bool(pattern.search(value)))
            if "*" in condition:
                parts = [part.strip().lower() for part in condition.split("*") if part.strip()]
                if not parts:
                    return text != ""
                lowered = text.str.lower()
                mask = pd.Series(True, index=series.index)
                for part in parts:
                    mask &= lowered.str.contains(part, regex=False)
                return mask & (text != "")
            target = condition.strip()
            return (text != "") & (text == target)

        if isinstance(condition, (int, float)):
            return series.notna() & (series == condition)

        return pd.Series(True, index=series.index)

    def _collect_with_batch_filters(
        self,
        columns: Sequence[str],
        dataset_filter: Optional[ds.Expression],
        predicates: Sequence[Callable[[pd.DataFrame], pd.Series]],
    ) -> List[int]:
        structure_ids: List[int] = []
        for batch_df in self._scan_metadata_batches(columns, dataset_filter):
            if batch_df.empty:
                continue
            mask = pd.Series(True, index=batch_df.index)
            for predicate in predicates:
                mask &= predicate(batch_df)
            if mask.any():
                structure_ids.extend(batch_df.loc[mask, "structure_id"].astype(int).tolist())
        return structure_ids

    def _load_atoms_by_structure_id(self, structure_id: int) -> Optional[Atoms]:
        lookup_record = self._load_lookup_record(structure_id)
        if lookup_record is None:
            return None

        shard_relpath = str(lookup_record["shard_relpath"])
        shard = self._get_shard_arrays(shard_relpath)
        local_idx = int(lookup_record["local_idx"])
        start = int(lookup_record["offset_start"])
        stop = int(lookup_record["offset_stop"])

        numbers = shard["numbers"][start:stop]
        positions = shard["positions"][start:stop]
        cell = shard["cell"][local_idx]
        pbc = shard["pbc"][local_idx]

        try:
            return Atoms(numbers=numbers, positions=positions, cell=cell, pbc=pbc)
        except Exception as exc:
            print(f"Loading error structure_id={structure_id}: {exc}")
            return None

    def _graph_cache_dir(self, graph_cache_path: Optional[Union[str, Path]] = None) -> Path:
        if graph_cache_path is None:
            return self.db_root / "indexes" / "graph_cache"
        path = Path(graph_cache_path).expanduser()
        if not path.is_absolute():
            path = self.db_root / path
        return path.resolve()

    def _infer_geometry_bond(
        self,
        atomic_number_a: int,
        atomic_number_b: int,
        distance: float,
    ) -> Tuple[str, str, str]:
        element_a = _symbol_from_atomic_number(atomic_number_a)
        element_b = _symbol_from_atomic_number(atomic_number_b)
        if _is_metal_atomic_number(atomic_number_a) or _is_metal_atomic_number(atomic_number_b):
            return "coordination", "coordination", "geometry_metal"

        pair = tuple(sorted((element_a, element_b)))
        reference_lengths = {
            ("C", "C"): {"single": 1.54, "double": 1.34, "triple": 1.20, "aromatic": 1.39},
            ("C", "N"): {"single": 1.47, "double": 1.28, "triple": 1.16, "aromatic": 1.34},
            ("C", "O"): {"single": 1.43, "double": 1.23},
            ("C", "S"): {"single": 1.82, "double": 1.60},
            ("N", "N"): {"single": 1.45, "double": 1.25, "triple": 1.10},
            ("N", "O"): {"single": 1.40, "double": 1.21},
            ("O", "O"): {"single": 1.48, "double": 1.21},
        }
        refs = reference_lengths.get(pair)
        if refs:
            scores = {
                order: abs(float(distance) - ref_length)
                for order, ref_length in refs.items()
            }
            best_order, best_score = min(scores.items(), key=lambda item: item[1])
            ordered_scores = sorted(scores.values())
            if best_score <= 0.08 and (
                len(ordered_scores) == 1 or ordered_scores[1] - ordered_scores[0] >= 0.05
            ):
                edge_type = "aromatic" if best_order == "aromatic" else "covalent"
                return edge_type, best_order, "geometry_high"
            if best_score <= 0.16:
                return "covalent", "unknown", "geometry_low"

        return "covalent", "single", "geometry_default"

    def _infer_graph_node_hybridization(self, graph: Any, node: int) -> str:
        attrs = graph.nodes[node]
        element = str(attrs.get("element", "")).strip().capitalize()
        if not element or element == "H":
            return ""
        if bool(attrs.get("is_metal", False)) or _is_metal_symbol(element):
            return "metal"

        bond_orders: List[str] = []
        heavy_distances: List[float] = []
        heavy_neighbors = 0
        for neighbor in graph.neighbors(node):
            neighbor_attrs = graph.nodes[neighbor]
            neighbor_element = str(neighbor_attrs.get("element", "")).strip().capitalize()
            edge_attrs = graph.edges[node, neighbor]
            order = str(edge_attrs.get("bond_order", "unknown")).strip().lower()
            if order and order != "nan":
                bond_orders.append(order)
            if neighbor_element and neighbor_element != "H":
                heavy_neighbors += 1
                try:
                    distance = float(edge_attrs.get("distance", np.nan))
                except (TypeError, ValueError):
                    distance = float("nan")
                if not np.isnan(distance):
                    heavy_distances.append(distance)

        if any(order == "triple" for order in bond_orders):
            return "sp"
        if any(order == "aromatic" for order in bond_orders):
            return "aromatic"
        if any(order == "double" for order in bond_orders):
            return "sp2"

        if element == "C":
            short_heavy = sum(1 for distance in heavy_distances if distance <= 1.43)
            very_short_heavy = any(distance <= 1.36 for distance in heavy_distances)
            if very_short_heavy or short_heavy >= 2:
                return "sp2"
            return "sp3"

        if element in {"N", "O", "P", "S"}:
            if any(distance <= 1.30 for distance in heavy_distances):
                return "sp2"
            return "sp3" if graph.degree(node) > 0 or heavy_neighbors > 0 else "unknown"

        return "unknown"

    def _annotate_graph_node_features(self, graph: Any) -> None:
        for node in list(graph.nodes):
            heavy_distances: List[float] = []
            has_aromatic_edge = False
            has_double_edge = False
            has_triple_edge = False
            heavy_degree = 0
            for neighbor in graph.neighbors(node):
                neighbor_element = str(graph.nodes[neighbor].get("element", "")).strip().capitalize()
                edge_attrs = graph.edges[node, neighbor]
                order = str(edge_attrs.get("bond_order", "unknown")).strip().lower()
                has_aromatic_edge = has_aromatic_edge or order == "aromatic"
                has_double_edge = has_double_edge or order == "double"
                has_triple_edge = has_triple_edge or order == "triple"
                if neighbor_element and neighbor_element != "H":
                    heavy_degree += 1
                    try:
                        distance = float(edge_attrs.get("distance", np.nan))
                    except (TypeError, ValueError):
                        distance = float("nan")
                    if not np.isnan(distance):
                        heavy_distances.append(distance)

            graph.nodes[node]["graph_degree"] = int(graph.degree(node))
            graph.nodes[node]["heavy_degree"] = int(heavy_degree)
            graph.nodes[node]["geometry_hybridization"] = self._infer_graph_node_hybridization(
                graph,
                int(node),
            )
            graph.nodes[node]["has_aromatic_edge"] = bool(has_aromatic_edge)
            graph.nodes[node]["has_double_edge"] = bool(has_double_edge)
            graph.nodes[node]["has_triple_edge"] = bool(has_triple_edge)
            graph.nodes[node]["min_heavy_bond_distance"] = (
                float(min(heavy_distances)) if heavy_distances else np.nan
            )
            graph.nodes[node]["mean_heavy_bond_distance"] = (
                float(np.mean(heavy_distances)) if heavy_distances else np.nan
            )

    def _build_geometry_graph(
        self,
        atoms: Atoms,
        structure_id: int,
        covalent_scale: float = 1.15,
    ) -> Any:
        if not ASE_AVAILABLE:
            raise ImportError("ASE is required to build graph cache")
        if not NETWORKX_AVAILABLE:
            raise ImportError("networkx is required to build graph cache")

        graph = nx.Graph(structure_id=int(structure_id))
        numbers = np.asarray(atoms.get_atomic_numbers(), dtype=int)
        positions = np.asarray(atoms.get_positions(), dtype=float)
        cell = np.asarray(atoms.get_cell(), dtype=float)

        for atom_idx, atomic_number in enumerate(numbers.tolist()):
            element = _symbol_from_atomic_number(int(atomic_number))
            graph.add_node(
                int(atom_idx),
                structure_id=int(structure_id),
                atom_index=int(atom_idx),
                element=element,
                atomic_number=int(atomic_number),
                is_metal=_is_metal_atomic_number(int(atomic_number)),
                is_hydrogen=int(atomic_number) == 1,
                is_dummy=False,
                x=float(positions[atom_idx][0]),
                y=float(positions[atom_idx][1]),
                z=float(positions[atom_idx][2]),
            )

        if len(atoms) == 0:
            return graph

        cutoffs = natural_cutoffs(atoms, mult=float(covalent_scale))
        neighbor_list = NeighborList(
            cutoffs,
            skin=0.0,
            self_interaction=False,
            bothways=True,
        )
        neighbor_list.update(atoms)

        for atom_idx in range(len(atoms)):
            neighbors, offsets = neighbor_list.get_neighbors(atom_idx)
            for neighbor, offset in zip(neighbors.tolist(), offsets.tolist()):
                neighbor = int(neighbor)
                if neighbor <= atom_idx:
                    continue
                offset_vec = np.asarray(offset, dtype=float)
                vector = positions[neighbor] + np.dot(offset_vec, cell) - positions[atom_idx]
                distance = float(np.linalg.norm(vector))
                edge_type, bond_order, confidence = self._infer_geometry_bond(
                    int(numbers[atom_idx]),
                    int(numbers[neighbor]),
                    distance,
                )
                graph.add_edge(
                    int(atom_idx),
                    int(neighbor),
                    structure_id=int(structure_id),
                    atom_i=int(atom_idx),
                    atom_j=int(neighbor),
                    edge_type=edge_type,
                    bond_order=bond_order,
                    distance=distance,
                    offset_x=int(offset[0]),
                    offset_y=int(offset[1]),
                    offset_z=int(offset[2]),
                    confidence=confidence,
                    source="ase_neighborlist",
                )

        for component_id, component_nodes in enumerate(nx.connected_components(graph)):
            for atom_idx in component_nodes:
                graph.nodes[atom_idx]["component_id"] = int(component_id)
        return graph

    def _pymatgen_0d_component_summary(self, atoms: Atoms) -> Dict[str, Any]:
        if not PYMATGEN_AVAILABLE or not PYMATGEN_GRAPH_AVAILABLE:
            return {
                "status": "not_available",
                "is_0d": True,
                "n_pymatgen_molecules": 0,
                "pymatgen_component_sizes": [],
                "message": "pymatgen CrystalNN is not available",
            }
        try:
            structure = Structure(
                lattice=atoms.get_cell(),
                species=atoms.get_atomic_numbers(),
                coords=atoms.get_scaled_positions(),
                coords_are_cartesian=False,
                validate_proximity=False,
            )
            structure_graph = CrystalNN().get_bonded_structure(structure)
            molecules = structure_graph.get_subgraphs_as_molecules()
            sizes = [int(len(molecule)) for molecule in molecules]
            return {
                "status": "ok",
                "is_0d": True,
                "n_pymatgen_molecules": int(len(molecules)),
                "pymatgen_component_sizes": sizes,
                "message": "",
            }
        except Exception as exc:
            message = str(exc)
            return {
                "status": "extended_or_failed",
                "is_0d": False,
                "n_pymatgen_molecules": 0,
                "pymatgen_component_sizes": [],
                "message": message,
            }

    def _geometry_0d_component_summary(self, graph: Any) -> Dict[str, Any]:
        component_sizes: List[int] = []
        periodic_components: List[Dict[str, Any]] = []
        for component_id, component_nodes in enumerate(nx.connected_components(graph)):
            nodes = sorted(int(node) for node in component_nodes)
            component_sizes.append(int(len(nodes)))
            if not nodes:
                continue

            shifts: Dict[int, Tuple[int, int, int]] = {nodes[0]: (0, 0, 0)}
            stack = [nodes[0]]
            is_periodic = False
            mismatch: Optional[Dict[str, Any]] = None
            node_set = set(nodes)

            while stack and not is_periodic:
                node = stack.pop()
                node_shift = np.asarray(shifts[node], dtype=int)
                for neighbor in graph.neighbors(node):
                    neighbor = int(neighbor)
                    if neighbor not in node_set:
                        continue
                    edge_attrs = graph.edges[node, neighbor]
                    edge_i = int(edge_attrs.get("atom_i", node))
                    edge_j = int(edge_attrs.get("atom_j", neighbor))
                    edge_offset = np.asarray(
                        [
                            int(edge_attrs.get("offset_x", 0)),
                            int(edge_attrs.get("offset_y", 0)),
                            int(edge_attrs.get("offset_z", 0)),
                        ],
                        dtype=int,
                    )
                    if node == edge_i and neighbor == edge_j:
                        expected_shift = node_shift + edge_offset
                    elif node == edge_j and neighbor == edge_i:
                        expected_shift = node_shift - edge_offset
                    else:
                        expected_shift = node_shift
                    expected_tuple = tuple(int(value) for value in expected_shift.tolist())

                    if neighbor not in shifts:
                        shifts[neighbor] = expected_tuple
                        stack.append(neighbor)
                    elif shifts[neighbor] != expected_tuple:
                        is_periodic = True
                        mismatch = {
                            "component_id": int(component_id),
                            "atom_i": int(node),
                            "atom_j": int(neighbor),
                            "existing_shift": list(shifts[neighbor]),
                            "expected_shift": list(expected_tuple),
                        }
                        break

            if is_periodic:
                periodic_components.append(
                    {
                        "component_id": int(component_id),
                        "size": int(len(nodes)),
                        "mismatch": mismatch,
                    }
                )

        is_0d = len(periodic_components) == 0
        return {
            "status": "ok" if is_0d else "periodic_geometry",
            "is_0d": bool(is_0d),
            "n_pymatgen_molecules": int(len(component_sizes)),
            "pymatgen_component_sizes": component_sizes,
            "message": "" if is_0d else _json_dumps_sorted(periodic_components[:5]),
        }

    def _apply_rdkit_bond_orders_to_graph(
        self,
        graph: Any,
        atoms: Atoms,
        max_component_atoms: int = 256,
    ) -> int:
        if not RDKIT_AVAILABLE:
            return 0
        positions = np.asarray(atoms.get_positions(), dtype=float)
        numbers = np.asarray(atoms.get_atomic_numbers(), dtype=int)
        updated = 0
        for component_nodes in nx.connected_components(graph):
            component_indices = np.array(sorted(int(node) for node in component_nodes), dtype=int)
            if component_indices.size == 0 or component_indices.size > int(max_component_atoms):
                continue
            if any(_is_metal_atomic_number(int(numbers[idx])) for idx in component_indices.tolist()):
                continue
            try:
                rdkit_mol = _build_component_molecule(
                    numbers[component_indices],
                    positions[component_indices],
                )
            except Exception:
                continue
            for bond in rdkit_mol.GetBonds():
                local_a = int(bond.GetBeginAtomIdx())
                local_b = int(bond.GetEndAtomIdx())
                atom_a = int(component_indices[local_a])
                atom_b = int(component_indices[local_b])
                if not graph.has_edge(atom_a, atom_b):
                    continue
                bond_order = "aromatic" if bond.GetIsAromatic() else RDKIT_BOND_ORDER_MAP.get(
                    bond.GetBondType(),
                    "unknown",
                )
                graph.edges[atom_a, atom_b]["bond_order"] = bond_order
                graph.edges[atom_a, atom_b]["edge_type"] = (
                    "aromatic" if bond_order == "aromatic" else "covalent"
                )
                graph.edges[atom_a, atom_b]["confidence"] = "rdkit"
                graph.edges[atom_a, atom_b]["source"] = (
                    str(graph.edges[atom_a, atom_b].get("source", ""))
                    + "+rdkit_bond_order"
                ).strip("+")
                updated += 1
        return updated

    def _apply_openbabel_bond_orders_to_graph(
        self,
        graph: Any,
        atoms: Atoms,
        max_component_atoms: int = 256,
    ) -> int:
        if not OPENBABEL_AVAILABLE:
            return 0
        positions = np.asarray(atoms.get_positions(), dtype=float)
        numbers = np.asarray(atoms.get_atomic_numbers(), dtype=int)
        updated = 0
        for component_nodes in nx.connected_components(graph):
            component_indices = np.array(sorted(int(node) for node in component_nodes), dtype=int)
            if component_indices.size == 0 or component_indices.size > int(max_component_atoms):
                continue
            if any(_is_metal_atomic_number(int(numbers[idx])) for idx in component_indices.tolist()):
                continue
            try:
                mol = OB.OBMol()
                for atom_idx in component_indices.tolist():
                    atom = mol.NewAtom()
                    atom.SetAtomicNum(int(numbers[atom_idx]))
                    x, y, z = positions[atom_idx].tolist()
                    atom.SetVector(float(x), float(y), float(z))
                mol.ConnectTheDots()
                mol.PerceiveBondOrders()
                for bond in OB.OBMolBondIter(mol):
                    atom_a = int(component_indices[int(bond.GetBeginAtomIdx()) - 1])
                    atom_b = int(component_indices[int(bond.GetEndAtomIdx()) - 1])
                    if not graph.has_edge(atom_a, atom_b):
                        continue
                    if bond.IsAromatic():
                        bond_order = "aromatic"
                    else:
                        order = int(bond.GetBondOrder())
                        bond_order = {1: "single", 2: "double", 3: "triple"}.get(order, "unknown")
                    graph.edges[atom_a, atom_b]["bond_order"] = bond_order
                    graph.edges[atom_a, atom_b]["edge_type"] = (
                        "aromatic" if bond_order == "aromatic" else "covalent"
                    )
                    graph.edges[atom_a, atom_b]["confidence"] = "openbabel"
                    graph.edges[atom_a, atom_b]["source"] = (
                        str(graph.edges[atom_a, atom_b].get("source", ""))
                        + "+openbabel_bond_order"
                    ).strip("+")
                    updated += 1
            except Exception:
                continue
        return updated

    def _apply_bond_order_backend(
        self,
        graph: Any,
        atoms: Atoms,
        backend: str = "rdkit",
        max_component_atoms: int = 256,
    ) -> Tuple[str, int]:
        backend = str(backend or "none").strip().lower()
        if backend in {"none", "off", "false"}:
            return "none", 0
        if backend == "rdkit":
            updated = self._apply_rdkit_bond_orders_to_graph(graph, atoms, max_component_atoms)
            return ("rdkit" if updated else "rdkit_no_updates", updated)
        if backend in {"openbabel", "obabel"}:
            updated = self._apply_openbabel_bond_orders_to_graph(graph, atoms, max_component_atoms)
            return ("openbabel" if updated else "openbabel_no_updates", updated)
        if backend == "auto":
            updated = self._apply_rdkit_bond_orders_to_graph(graph, atoms, max_component_atoms)
            if updated:
                return "rdkit", updated
            updated = self._apply_openbabel_bond_orders_to_graph(graph, atoms, max_component_atoms)
            return ("openbabel" if updated else "auto_no_updates", updated)
        return backend, 0

    def _interatomic_distance_quality_summary(
        self,
        atoms: Atoms,
        min_distance: Optional[float] = 0.6,
        max_examples: int = 10,
    ) -> Dict[str, Any]:
        threshold = float(min_distance or 0.0)
        if threshold <= 0:
            return {
                "status": "not_requested",
                "threshold": threshold,
                "close_pair_count": 0,
                "min_distance": np.nan,
                "min_atom_i": None,
                "min_atom_j": None,
                "min_elements": "",
                "examples": [],
                "message": "",
            }
        if len(atoms) < 2:
            return {
                "status": "ok",
                "threshold": threshold,
                "close_pair_count": 0,
                "min_distance": np.nan,
                "min_atom_i": None,
                "min_atom_j": None,
                "min_elements": "",
                "examples": [],
                "message": "",
            }

        try:
            positions = np.asarray(atoms.get_positions(), dtype=float)
            cell = np.asarray(atoms.get_cell(), dtype=float)
            symbols = atoms.get_chemical_symbols()
            neighbor_list = NeighborList(
                [threshold / 2.0] * len(atoms),
                skin=0.0,
                self_interaction=False,
                bothways=False,
            )
            neighbor_list.update(atoms)

            pair_best: Dict[Tuple[int, int], Tuple[float, Tuple[int, int, int]]] = {}
            for atom_i in range(len(atoms)):
                neighbors, offsets = neighbor_list.get_neighbors(atom_i)
                for atom_j, offset in zip(neighbors.tolist(), offsets.tolist()):
                    atom_j = int(atom_j)
                    if atom_j == atom_i:
                        continue
                    pair_key = tuple(sorted((int(atom_i), atom_j)))
                    offset_vec = np.asarray(offset, dtype=float)
                    vector = positions[atom_j] + np.dot(offset_vec, cell) - positions[atom_i]
                    distance = float(np.linalg.norm(vector))
                    if distance >= threshold:
                        continue
                    offset_tuple = tuple(int(value) for value in np.asarray(offset, dtype=int).tolist())
                    if pair_key not in pair_best or distance < pair_best[pair_key][0]:
                        pair_best[pair_key] = (distance, offset_tuple)

            if not pair_best:
                return {
                    "status": "ok",
                    "threshold": threshold,
                    "close_pair_count": 0,
                    "min_distance": np.nan,
                    "min_atom_i": None,
                    "min_atom_j": None,
                    "min_elements": "",
                    "examples": [],
                    "message": "",
                }

            sorted_pairs = sorted(
                (
                    distance,
                    atom_i,
                    atom_j,
                    symbols[atom_i],
                    symbols[atom_j],
                    offset,
                )
                for (atom_i, atom_j), (distance, offset) in pair_best.items()
            )
            min_distance_value, min_atom_i, min_atom_j, min_el_i, min_el_j, _ = sorted_pairs[0]
            examples = [
                {
                    "distance": float(distance),
                    "atom_i": int(atom_i + 1),
                    "atom_j": int(atom_j + 1),
                    "element_i": str(el_i),
                    "element_j": str(el_j),
                    "offset": [int(value) for value in offset],
                }
                for distance, atom_i, atom_j, el_i, el_j, offset in sorted_pairs[:max_examples]
            ]
            message = (
                f"min interatomic distance {min_distance_value:.3f} A < {threshold:.3f} A "
                f"between {min_el_i}{min_atom_i + 1}-{min_el_j}{min_atom_j + 1}; "
                f"close_pairs={len(pair_best)}"
            )
            return {
                "status": "bad_geometry",
                "threshold": threshold,
                "close_pair_count": int(len(pair_best)),
                "min_distance": float(min_distance_value),
                "min_atom_i": int(min_atom_i + 1),
                "min_atom_j": int(min_atom_j + 1),
                "min_elements": f"{min_el_i}-{min_el_j}",
                "examples": examples,
                "message": message,
            }
        except Exception as exc:
            return {
                "status": "failed",
                "threshold": threshold,
                "close_pair_count": 0,
                "min_distance": np.nan,
                "min_atom_i": None,
                "min_atom_j": None,
                "min_elements": "",
                "examples": [],
                "message": str(exc),
            }

    def _graph_cache_quality_fields(
        self,
        quality_summary: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        quality_summary = quality_summary or {}
        min_distance = quality_summary.get("min_distance", np.nan)
        try:
            min_distance = float(min_distance)
        except (TypeError, ValueError):
            min_distance = np.nan
        return {
            "geometry_quality_status": str(quality_summary.get("status", "not_checked")),
            "geometry_quality_threshold": float(quality_summary.get("threshold", 0.0) or 0.0),
            "close_pair_count": int(quality_summary.get("close_pair_count", 0) or 0),
            "min_close_distance": min_distance,
            "min_close_atom_i": (
                int(quality_summary["min_atom_i"])
                if quality_summary.get("min_atom_i") is not None
                else -1
            ),
            "min_close_atom_j": (
                int(quality_summary["min_atom_j"])
                if quality_summary.get("min_atom_j") is not None
                else -1
            ),
            "min_close_elements": str(quality_summary.get("min_elements", "")),
            "close_pairs_json": _json_dumps_sorted(quality_summary.get("examples", [])),
        }

    def _empty_graph_cache_summary_row(
        self,
        structure_id: int,
        metadata: Optional[Dict[str, Any]],
        status: str,
        skip_reason: str,
        n_atoms: int = 0,
        component_filter_summary: Optional[Dict[str, Any]] = None,
        bond_order_backend: str = "none",
        bond_order_updates: int = 0,
        quality_summary: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        component_filter_summary = component_filter_summary or {}
        return {
            "structure_id": int(structure_id),
            "refcode": metadata.get("refcode") if metadata else None,
            "graph_status": status,
            "skip_reason": str(skip_reason),
            "n_atoms": int(n_atoms),
            "n_edges": 0,
            "n_components": 0,
            "has_metal": False,
            "has_aromatic": False,
            "element_counts_json": "{}",
            "node_keys_json": "[]",
            "edge_keys_json": "[]",
            "generic_edge_keys_json": "[]",
            "bond_type_counts_json": "{}",
            "component_filter_status": str(component_filter_summary.get("status", "")),
            "n_pymatgen_molecules": int(component_filter_summary.get("n_pymatgen_molecules", 0)),
            "pymatgen_component_sizes_json": _json_dumps_sorted(
                component_filter_summary.get("pymatgen_component_sizes", [])
            ),
            "bond_order_backend": str(bond_order_backend),
            "bond_order_updates": int(bond_order_updates),
            **self._graph_cache_quality_fields(quality_summary),
        }

    def _graph_to_cache_rows(
        self,
        graph: Any,
        structure_id: int,
        metadata: Dict[str, Any],
        status: str = "ok",
        skip_reason: str = "",
        component_filter_summary: Optional[Dict[str, Any]] = None,
        bond_order_backend: str = "geometry",
        bond_order_updates: int = 0,
        quality_summary: Optional[Dict[str, Any]] = None,
    ) -> Tuple[Dict[str, Any], List[Dict[str, Any]], List[Dict[str, Any]]]:
        self._annotate_graph_node_features(graph)
        node_rows: List[Dict[str, Any]] = []
        edge_rows: List[Dict[str, Any]] = []
        for atom_idx, attrs in graph.nodes(data=True):
            row = dict(attrs)
            row["structure_id"] = int(structure_id)
            row["atom_index"] = int(atom_idx)
            node_rows.append(row)
        for atom_i, atom_j, attrs in graph.edges(data=True):
            row = dict(attrs)
            row["structure_id"] = int(structure_id)
            row["atom_i"] = int(atom_i)
            row["atom_j"] = int(atom_j)
            edge_rows.append(row)

        fingerprint = graph_fingerprint(graph, ignore_dummy=True)
        bond_type_counts = Counter(str(row.get("bond_order", "unknown")) for row in edge_rows)
        component_ids = {
            int(row.get("component_id", -1))
            for row in node_rows
            if row.get("component_id", -1) is not None
        }
        summary_row = {
            "structure_id": int(structure_id),
            "refcode": metadata.get("refcode"),
            "graph_status": status,
            "skip_reason": skip_reason,
            "n_atoms": len(node_rows),
            "n_edges": len(edge_rows),
            "n_components": len(component_ids),
            "has_metal": bool(fingerprint["has_metal"]),
            "has_aromatic": bool(fingerprint["has_aromatic"]),
            "element_counts_json": _json_dumps_sorted(fingerprint["element_counts"]),
            "node_keys_json": _json_dumps_sorted(fingerprint["node_keys"]),
            "edge_keys_json": _json_dumps_sorted(fingerprint["edge_keys"]),
            "generic_edge_keys_json": _json_dumps_sorted(fingerprint["generic_edge_keys"]),
            "bond_type_counts_json": _json_dumps_sorted(dict(sorted(bond_type_counts.items()))),
            "component_filter_status": (component_filter_summary or {}).get("status", ""),
            "n_pymatgen_molecules": int(
                (component_filter_summary or {}).get("n_pymatgen_molecules", 0)
            ),
            "pymatgen_component_sizes_json": _json_dumps_sorted(
                (component_filter_summary or {}).get("pymatgen_component_sizes", [])
            ),
            "bond_order_backend": str(bond_order_backend),
            "bond_order_updates": int(bond_order_updates),
            **self._graph_cache_quality_fields(quality_summary),
        }
        return summary_row, node_rows, edge_rows

    def _build_graph_cache_structure_rows(
        self,
        structure_id: int,
        max_atoms: Optional[int] = 1000,
        covalent_scale: float = 1.15,
        min_nonbonded_distance: float = 0.6,
        skip_extended_networks: bool = True,
        component_filter_backend: str = "pymatgen",
        bond_order_backend: str = "rdkit",
    ) -> Dict[str, Any]:
        structure_id = int(structure_id)
        atoms, metadata = self.get_structure(structure_id)
        if atoms is None or metadata is None:
            return {
                "status": "load_failed",
                "summary_row": self._empty_graph_cache_summary_row(
                    structure_id=structure_id,
                    metadata=None,
                    status="load_failed",
                    skip_reason="structure_load_failed",
                    n_atoms=0,
                ),
                "node_rows": [],
                "edge_rows": [],
            }
        if max_atoms is not None and len(atoms) > int(max_atoms):
            return {
                "status": "skipped",
                "summary_row": self._empty_graph_cache_summary_row(
                    structure_id=structure_id,
                    metadata=metadata,
                    status="skipped",
                    skip_reason=f"n_atoms>{int(max_atoms)}",
                    n_atoms=len(atoms),
                    component_filter_summary={"status": "skipped_before_filter"},
                ),
                "node_rows": [],
                "edge_rows": [],
            }

        geometry_quality = self._interatomic_distance_quality_summary(
            atoms,
            min_distance=min_nonbonded_distance,
        )
        if geometry_quality.get("status") == "bad_geometry":
            return {
                "status": "bad_geometry",
                "summary_row": self._empty_graph_cache_summary_row(
                    structure_id=structure_id,
                    metadata=metadata,
                    status="bad_geometry",
                    skip_reason=str(geometry_quality.get("message", ""))[:500],
                    n_atoms=len(atoms),
                    component_filter_summary={"status": "skipped_bad_geometry"},
                    quality_summary=geometry_quality,
                ),
                "node_rows": [],
                "edge_rows": [],
            }

        component_backend = str(component_filter_backend or "none").strip().lower()
        component_summary = {
            "status": "not_requested",
            "is_0d": True,
            "n_pymatgen_molecules": 0,
            "pymatgen_component_sizes": [],
            "message": "",
        }
        if component_backend == "pymatgen":
            component_summary = self._pymatgen_0d_component_summary(atoms)
            if skip_extended_networks and not bool(component_summary.get("is_0d", True)):
                return {
                    "status": "polymeric_or_extended",
                    "summary_row": self._empty_graph_cache_summary_row(
                        structure_id=structure_id,
                        metadata=metadata,
                        status="polymeric_or_extended",
                        skip_reason=str(component_summary.get("message", ""))[:500],
                        n_atoms=len(atoms),
                        component_filter_summary=component_summary,
                        quality_summary=geometry_quality,
                    ),
                    "node_rows": [],
                    "edge_rows": [],
                }

        try:
            graph = self._build_geometry_graph(
                atoms=atoms,
                structure_id=structure_id,
                covalent_scale=float(covalent_scale),
            )
            if component_backend in {"geometry", "ase", "fast"}:
                component_summary = self._geometry_0d_component_summary(graph)
                if skip_extended_networks and not bool(component_summary.get("is_0d", True)):
                    return {
                        "status": "polymeric_or_extended",
                        "summary_row": self._empty_graph_cache_summary_row(
                            structure_id=structure_id,
                            metadata=metadata,
                            status="polymeric_or_extended",
                            skip_reason=str(component_summary.get("message", ""))[:500],
                            n_atoms=len(atoms),
                            component_filter_summary=component_summary,
                            quality_summary=geometry_quality,
                        ),
                        "node_rows": [],
                        "edge_rows": [],
                    }
            backend_used, bond_order_updates = self._apply_bond_order_backend(
                graph=graph,
                atoms=atoms,
                backend=bond_order_backend,
            )
            summary_row, graph_node_rows, graph_edge_rows = self._graph_to_cache_rows(
                graph=graph,
                structure_id=structure_id,
                metadata=metadata,
                status="ok",
                skip_reason="",
                component_filter_summary=component_summary,
                bond_order_backend=backend_used,
                bond_order_updates=bond_order_updates,
                quality_summary=geometry_quality,
            )
            return {
                "status": "ok",
                "summary_row": summary_row,
                "node_rows": graph_node_rows,
                "edge_rows": graph_edge_rows,
            }
        except Exception as exc:
            return {
                "status": "graph_failed",
                "summary_row": self._empty_graph_cache_summary_row(
                    structure_id=structure_id,
                    metadata=metadata,
                    status="graph_failed",
                    skip_reason=str(exc),
                    n_atoms=len(atoms),
                    component_filter_summary=component_summary,
                    quality_summary=geometry_quality,
                ),
                "node_rows": [],
                "edge_rows": [],
            }

    def build_graph_cache(
        self,
        graph_cache_path: Optional[Union[str, Path]] = None,
        structure_ids: Optional[Sequence[int]] = None,
        max_structures: Optional[int] = None,
        max_atoms: Optional[int] = 1000,
        covalent_scale: float = 1.15,
        min_nonbonded_distance: float = 0.6,
        skip_extended_networks: bool = True,
        component_filter_backend: str = "pymatgen",
        bond_order_backend: str = "rdkit",
        overwrite: bool = False,
        progress_every: int = 100,
        flush_every: int = 250,
        workers: int = 1,
        worker_chunk_size: int = 10,
    ) -> Dict[str, Any]:
        min_nonbonded_distance = float(min_nonbonded_distance or 0.0)
        cache_dir = self._graph_cache_dir(graph_cache_path)
        manifest_path = cache_dir / "manifest.json"
        summary_path = cache_dir / "graph_summary.parquet"
        nodes_path = cache_dir / "graph_nodes.parquet"
        edges_path = cache_dir / "graph_edges.parquet"
        if summary_path.exists() and nodes_path.exists() and edges_path.exists() and not overwrite:
            try:
                existing_summary = pd.read_parquet(summary_path)
                rebuild_reason = self._graph_cache_rebuild_reason(
                    cache_dir,
                    existing_summary,
                    min_nonbonded_distance,
                )
            except Exception as exc:
                rebuild_reason = f"cannot validate existing graph cache: {exc}"
            if not rebuild_reason:
                return {
                    "cache_dir": str(cache_dir),
                    "status": "exists",
                    "summary_path": str(summary_path),
                    "nodes_path": str(nodes_path),
                    "edges_path": str(edges_path),
                }
            print(
                "Existing graph cache is not compatible with the requested geometry filter; "
                f"rebuilding ({rebuild_reason})",
                flush=True,
            )

        if structure_ids:
            scan_ids = [int(value) for value in structure_ids]
        else:
            scan_ids = list(range(self.count()))
        if max_structures is not None:
            scan_ids = scan_ids[: int(max_structures)]

        summary_columns = [
            "structure_id", "refcode", "graph_status", "skip_reason", "n_atoms",
            "n_edges", "n_components", "has_metal", "has_aromatic",
            "element_counts_json", "node_keys_json", "edge_keys_json",
            "generic_edge_keys_json", "bond_type_counts_json",
            "component_filter_status", "n_pymatgen_molecules",
            "pymatgen_component_sizes_json", "bond_order_backend", "bond_order_updates",
            "geometry_quality_status", "geometry_quality_threshold", "close_pair_count",
            "min_close_distance", "min_close_atom_i", "min_close_atom_j",
            "min_close_elements", "close_pairs_json",
        ]
        node_columns = [
            "structure_id", "atom_index", "element", "atomic_number", "is_metal",
            "is_hydrogen", "is_dummy", "x", "y", "z", "component_id",
            "graph_degree", "heavy_degree", "geometry_hybridization",
            "has_aromatic_edge", "has_double_edge", "has_triple_edge",
            "min_heavy_bond_distance", "mean_heavy_bond_distance",
        ]
        edge_columns = [
            "structure_id", "atom_i", "atom_j", "edge_type", "bond_order",
            "distance", "offset_x", "offset_y", "offset_z", "confidence", "source",
        ]
        summary_schema = pa.schema(
            [
                ("structure_id", pa.int64()),
                ("refcode", pa.string()),
                ("graph_status", pa.string()),
                ("skip_reason", pa.string()),
                ("n_atoms", pa.int64()),
                ("n_edges", pa.int64()),
                ("n_components", pa.int64()),
                ("has_metal", pa.bool_()),
                ("has_aromatic", pa.bool_()),
                ("element_counts_json", pa.string()),
                ("node_keys_json", pa.string()),
                ("edge_keys_json", pa.string()),
                ("generic_edge_keys_json", pa.string()),
                ("bond_type_counts_json", pa.string()),
                ("component_filter_status", pa.string()),
                ("n_pymatgen_molecules", pa.int64()),
                ("pymatgen_component_sizes_json", pa.string()),
                ("bond_order_backend", pa.string()),
                ("bond_order_updates", pa.int64()),
                ("geometry_quality_status", pa.string()),
                ("geometry_quality_threshold", pa.float64()),
                ("close_pair_count", pa.int64()),
                ("min_close_distance", pa.float64()),
                ("min_close_atom_i", pa.int64()),
                ("min_close_atom_j", pa.int64()),
                ("min_close_elements", pa.string()),
                ("close_pairs_json", pa.string()),
            ]
        )
        node_schema = pa.schema(
            [
                ("structure_id", pa.int64()),
                ("atom_index", pa.int64()),
                ("element", pa.string()),
                ("atomic_number", pa.int64()),
                ("is_metal", pa.bool_()),
                ("is_hydrogen", pa.bool_()),
                ("is_dummy", pa.bool_()),
                ("x", pa.float64()),
                ("y", pa.float64()),
                ("z", pa.float64()),
                ("component_id", pa.int64()),
                ("graph_degree", pa.int64()),
                ("heavy_degree", pa.int64()),
                ("geometry_hybridization", pa.string()),
                ("has_aromatic_edge", pa.bool_()),
                ("has_double_edge", pa.bool_()),
                ("has_triple_edge", pa.bool_()),
                ("min_heavy_bond_distance", pa.float64()),
                ("mean_heavy_bond_distance", pa.float64()),
            ]
        )
        edge_schema = pa.schema(
            [
                ("structure_id", pa.int64()),
                ("atom_i", pa.int64()),
                ("atom_j", pa.int64()),
                ("edge_type", pa.string()),
                ("bond_order", pa.string()),
                ("distance", pa.float64()),
                ("offset_x", pa.int64()),
                ("offset_y", pa.int64()),
                ("offset_z", pa.int64()),
                ("confidence", pa.string()),
                ("source", pa.string()),
            ]
        )

        flush_every_int = int(flush_every or 0)
        if flush_every_int < 0:
            flush_every_int = 0
        progress_every_int = int(progress_every or 0)
        if progress_every_int < 0:
            progress_every_int = 0
        workers_int = int(workers or 1)
        if workers_int < 1:
            workers_int = 1
        if scan_ids:
            workers_int = min(workers_int, len(scan_ids))
        worker_chunk_size_int = int(worker_chunk_size or 1)
        if worker_chunk_size_int < 1:
            worker_chunk_size_int = 1

        cache_dir.parent.mkdir(parents=True, exist_ok=True)
        build_dir = Path(
            tempfile.mkdtemp(
                prefix=f"{cache_dir.name}.tmp-build-",
                dir=str(cache_dir.parent),
            )
        )
        build_manifest_path = build_dir / manifest_path.name
        build_summary_path = build_dir / summary_path.name
        build_nodes_path = build_dir / nodes_path.name
        build_edges_path = build_dir / edges_path.name

        summary_rows: List[Dict[str, Any]] = []
        node_rows: List[Dict[str, Any]] = []
        edge_rows: List[Dict[str, Any]] = []
        status_counts: Counter[str] = Counter()
        summary_writer: Optional[pq.ParquetWriter] = None
        node_writer: Optional[pq.ParquetWriter] = None
        edge_writer: Optional[pq.ParquetWriter] = None
        flush_count = 0
        build_completed = False

        def _table_from_rows(
            rows: Sequence[Dict[str, Any]],
            columns: Sequence[str],
            schema: pa.Schema,
        ) -> pa.Table:
            frame = pd.DataFrame(rows, columns=columns)
            table = pa.Table.from_pandas(frame, schema=schema, preserve_index=False)
            return table.cast(schema)

        def _empty_table(columns: Sequence[str], schema: pa.Schema) -> pa.Table:
            arrays = [pa.array([], type=schema.field(column).type) for column in columns]
            return pa.Table.from_arrays(arrays, schema=schema)

        def _write_rows(
            writer: Optional[pq.ParquetWriter],
            path: Path,
            rows: List[Dict[str, Any]],
            columns: Sequence[str],
            schema: pa.Schema,
        ) -> Optional[pq.ParquetWriter]:
            if not rows:
                return writer
            table = _table_from_rows(rows, columns, schema)
            if writer is None:
                writer = pq.ParquetWriter(path, schema)
            writer.write_table(table)
            rows.clear()
            return writer

        def _flush_graph_cache_rows() -> None:
            nonlocal edge_writer, flush_count, node_writer, summary_writer
            wrote_any = bool(summary_rows or node_rows or edge_rows)
            summary_writer = _write_rows(
                summary_writer,
                build_summary_path,
                summary_rows,
                summary_columns,
                summary_schema,
            )
            node_writer = _write_rows(
                node_writer,
                build_nodes_path,
                node_rows,
                node_columns,
                node_schema,
            )
            edge_writer = _write_rows(
                edge_writer,
                build_edges_path,
                edge_rows,
                edge_columns,
                edge_schema,
            )
            if wrote_any:
                flush_count += 1

        def _close_graph_cache_writers() -> None:
            nonlocal edge_writer, node_writer, summary_writer
            if summary_writer is not None:
                summary_writer.close()
                summary_writer = None
            if node_writer is not None:
                node_writer.close()
                node_writer = None
            if edge_writer is not None:
                edge_writer.close()
                edge_writer = None

        def _write_empty_cache_files_if_needed() -> None:
            if not build_summary_path.exists():
                pq.write_table(
                    _empty_table(summary_columns, summary_schema),
                    build_summary_path,
                )
            if not build_nodes_path.exists():
                pq.write_table(
                    _empty_table(node_columns, node_schema),
                    build_nodes_path,
                )
            if not build_edges_path.exists():
                pq.write_table(
                    _empty_table(edge_columns, edge_schema),
                    build_edges_path,
                )

        def _post_graph_cache_structure(idx: int) -> None:
            if flush_every_int and idx % flush_every_int == 0:
                _flush_graph_cache_rows()
            if progress_every_int and idx % progress_every_int == 0:
                print(f"Built graph cache for {idx}/{len(scan_ids)} structures", flush=True)

        def _remove_path(path: Path) -> None:
            if path.is_dir():
                shutil.rmtree(path)
            elif path.exists():
                path.unlink()

        def _replace_cache_dir() -> None:
            backup_dir: Optional[Path] = None
            if cache_dir.exists():
                backup_dir = Path(
                    tempfile.mkdtemp(
                        prefix=f"{cache_dir.name}.old-",
                        dir=str(cache_dir.parent),
                    )
                )
                backup_dir.rmdir()
                cache_dir.rename(backup_dir)
            try:
                build_dir.rename(cache_dir)
            except Exception:
                if backup_dir is not None and backup_dir.exists() and not cache_dir.exists():
                    backup_dir.rename(cache_dir)
                raise
            else:
                if backup_dir is not None and backup_dir.exists():
                    try:
                        _remove_path(backup_dir)
                    except Exception:
                        pass

        def _consume_graph_cache_result(result: Dict[str, Any], idx: int) -> None:
            summary_rows.append(result["summary_row"])
            node_rows.extend(result.get("node_rows", []))
            edge_rows.extend(result.get("edge_rows", []))
            status_counts[str(result.get("status", "unknown"))] += 1
            _post_graph_cache_structure(idx)

        worker_settings = {
            "max_atoms": max_atoms,
            "covalent_scale": float(covalent_scale),
            "min_nonbonded_distance": float(min_nonbonded_distance),
            "skip_extended_networks": bool(skip_extended_networks),
            "component_filter_backend": str(component_filter_backend),
            "bond_order_backend": str(bond_order_backend),
        }

        try:
            if workers_int <= 1 or len(scan_ids) <= 1:
                for idx, structure_id in enumerate(scan_ids, start=1):
                    result = self._build_graph_cache_structure_rows(
                        structure_id=int(structure_id),
                        **worker_settings,
                    )
                    _consume_graph_cache_result(result, idx)
            else:
                print(
                    f"Building graph cache with {workers_int} worker processes",
                    flush=True,
                )
                with ProcessPoolExecutor(
                    max_workers=workers_int,
                    initializer=_graph_cache_worker_init,
                    initargs=(str(self.db_root), worker_settings),
                ) as executor:
                    results_iter = executor.map(
                        _graph_cache_worker_process,
                        scan_ids,
                        chunksize=worker_chunk_size_int,
                    )
                    for idx, result in enumerate(results_iter, start=1):
                        _consume_graph_cache_result(result, idx)

            _flush_graph_cache_rows()
            _close_graph_cache_writers()
            _write_empty_cache_files_if_needed()
            manifest = {
                "format": "uspex_graph_cache_v0",
                "database": str(self.db_root),
                "count_requested": len(scan_ids),
                "status_counts": dict(status_counts),
                "files": {
                    "summary": summary_path.name,
                    "nodes": nodes_path.name,
                    "edges": edges_path.name,
                },
                "settings": {
                    "max_structures": max_structures,
                    "max_atoms": max_atoms,
                    "covalent_scale": float(covalent_scale),
                    "min_nonbonded_distance": float(min_nonbonded_distance),
                    "skip_extended_networks": bool(skip_extended_networks),
                    "component_filter_backend": str(component_filter_backend),
                    "bond_order_backend": str(bond_order_backend),
                    "flush_every": flush_every_int,
                    "workers": workers_int,
                    "worker_chunk_size": worker_chunk_size_int,
                },
            }
            with build_manifest_path.open("w", encoding="utf-8") as fh:
                json.dump(manifest, fh, indent=2, ensure_ascii=False)
            _replace_cache_dir()
            build_completed = True
        finally:
            _close_graph_cache_writers()
            if not build_completed and build_dir.exists():
                shutil.rmtree(build_dir, ignore_errors=True)

        return {
            "cache_dir": str(cache_dir),
            "status": "built",
            "summary_path": str(summary_path),
            "nodes_path": str(nodes_path),
            "edges_path": str(edges_path),
            "status_counts": dict(status_counts),
            "flush_count": int(flush_count),
            "workers": int(workers_int),
            "worker_chunk_size": int(worker_chunk_size_int),
        }

    def _load_graph_cache(
        self,
        graph_cache_path: Optional[Union[str, Path]] = None,
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        cache_dir = self._graph_cache_dir(graph_cache_path)
        summary_path = cache_dir / "graph_summary.parquet"
        nodes_path = cache_dir / "graph_nodes.parquet"
        edges_path = cache_dir / "graph_edges.parquet"
        missing = [path for path in (summary_path, nodes_path, edges_path) if not path.exists()]
        if missing:
            raise FileNotFoundError(
                "Graph cache is missing: " + ", ".join(str(path) for path in missing)
            )
        return (
            pd.read_parquet(summary_path),
            pd.read_parquet(nodes_path),
            pd.read_parquet(edges_path),
        )

    def _graph_cache_rebuild_reason(
        self,
        graph_cache_path: Optional[Union[str, Path]],
        summary_df: pd.DataFrame,
        min_nonbonded_distance: Optional[float],
    ) -> str:
        threshold = float(min_nonbonded_distance or 0.0)
        if threshold <= 0:
            return ""

        required_columns = {
            "geometry_quality_status",
            "geometry_quality_threshold",
            "close_pair_count",
            "min_close_distance",
            "min_close_atom_i",
            "min_close_atom_j",
            "min_close_elements",
            "close_pairs_json",
        }
        missing_columns = sorted(required_columns.difference(summary_df.columns))
        if missing_columns:
            return "missing geometry-quality columns: " + ", ".join(missing_columns)

        manifest_path = self._graph_cache_dir(graph_cache_path) / "manifest.json"
        try:
            with manifest_path.open("r", encoding="utf-8") as fh:
                manifest = json.load(fh)
            built_threshold = float(
                manifest.get("settings", {}).get("min_nonbonded_distance", 0.0) or 0.0
            )
        except Exception:
            return "missing or unreadable graph-cache manifest min_nonbonded_distance"

        if built_threshold + 1e-12 < threshold:
            return (
                f"cache was built with min_nonbonded_distance={built_threshold:g}, "
                f"requested {threshold:g}"
            )
        return ""

    def _graph_from_cache_rows(
        self,
        node_rows: pd.DataFrame,
        edge_rows: pd.DataFrame,
    ) -> Any:
        if not NETWORKX_AVAILABLE:
            raise ImportError("networkx is required for graph search")
        graph = nx.Graph()
        for row in node_rows.to_dict("records"):
            atom_index = int(row["atom_index"])
            graph.add_node(atom_index, **row)
        for row in edge_rows.to_dict("records"):
            graph.add_edge(int(row["atom_i"]), int(row["atom_j"]), **row)
        self._annotate_graph_node_features(graph)
        return graph

    def _split_into_molecules(
        self,
        atoms: Atoms,
        covalent_scale: float = 1.15,
    ) -> List[MoleculeComponent]:
        if not ASE_AVAILABLE:
            raise ImportError("ASE is required for SMARTS contact search")
        if not RDKIT_AVAILABLE:
            raise ImportError("RDKit is required for SMARTS contact search")

        cutoffs = natural_cutoffs(atoms, mult=float(covalent_scale))
        neighbor_list = NeighborList(
            cutoffs,
            skin=0.0,
            self_interaction=False,
            bothways=True,
        )
        neighbor_list.update(atoms)

        positions = atoms.get_positions()
        cell = np.asarray(atoms.get_cell(), dtype=float)
        numbers = np.asarray(atoms.get_atomic_numbers(), dtype=int)
        visited = set()
        components: List[MoleculeComponent] = []
        parent_molecule_id = 0

        for start in range(len(atoms)):
            if start in visited:
                continue
            stack = [start]
            shifts: Dict[int, np.ndarray] = {start: np.zeros(3, dtype=int)}
            visited.add(start)
            while stack:
                node = stack.pop()
                neighbors, offsets = neighbor_list.get_neighbors(node)
                for neighbor, offset in zip(neighbors.tolist(), offsets.tolist()):
                    proposed = shifts[node] + np.asarray(offset, dtype=int)
                    if neighbor not in shifts:
                        shifts[neighbor] = proposed
                        visited.add(neighbor)
                        stack.append(neighbor)
            global_indices = np.array(sorted(shifts), dtype=int)
            component_shifts = np.array([shifts[idx] for idx in global_indices], dtype=float)
            positions_unwrapped = positions[global_indices] + np.dot(component_shifts, cell)
            global_to_local = {int(global_idx): local_idx for local_idx, global_idx in enumerate(global_indices)}
            component_edges: Set[Tuple[int, int]] = set()
            for global_idx in global_indices.tolist():
                local_idx = global_to_local[int(global_idx)]
                neighbors, _ = neighbor_list.get_neighbors(int(global_idx))
                for neighbor in neighbors.tolist():
                    if neighbor not in global_to_local:
                        continue
                    neighbor_local = global_to_local[int(neighbor)]
                    if local_idx == neighbor_local:
                        continue
                    component_edges.add(tuple(sorted((local_idx, neighbor_local))))

            try:
                component_numbers = numbers[global_indices]
                rdkit_mol = _build_component_molecule(component_numbers, positions_unwrapped)
                components.append(
                    MoleculeComponent(
                        search_component_id=len(components),
                        molecule_id=parent_molecule_id,
                        global_indices=global_indices,
                        positions_unwrapped=positions_unwrapped,
                        atomic_number_counts=_atomic_number_counts(component_numbers),
                        rdkit_mol=rdkit_mol,
                    )
                )
            except Exception as exc:
                fallback_components = _build_rdkit_fallback_subcomponents(
                    numbers=numbers[global_indices],
                    positions=positions_unwrapped,
                    edges=component_edges,
                )
                if not fallback_components:
                    raise ValueError(
                        f"Failed to construct SMARTS-search molecule for component {parent_molecule_id}: {exc}"
                    ) from exc
                for local_indices, local_positions, rdkit_mol in fallback_components:
                    components.append(
                        MoleculeComponent(
                            search_component_id=len(components),
                            molecule_id=parent_molecule_id,
                            global_indices=global_indices[local_indices],
                            positions_unwrapped=local_positions,
                            atomic_number_counts=_atomic_number_counts(
                                numbers[global_indices[local_indices]]
                            ),
                            rdkit_mol=rdkit_mol,
                        )
                    )
            parent_molecule_id += 1
        return components

    def _find_fragment_matches(
        self,
        component: MoleculeComponent,
        spec: FragmentSpec,
    ) -> List[FragmentMatch]:
        if component.global_indices.size < spec.atom_count:
            return []
        if not _atomic_number_requirements_satisfied(
            component.atomic_number_counts,
            spec.exact_atomic_number_requirements,
        ):
            return []
        matches = component.rdkit_mol.GetSubstructMatches(spec.pattern, uniquify=True)
        unique: Dict[Tuple[int, ...], FragmentMatch] = {}
        for match in matches:
            query_atoms = tuple(int(idx) for idx in match)
            if len(query_atoms) != spec.atom_count:
                continue
            anchor_local = query_atoms[spec.anchor_query_index]
            prev_local = (
                query_atoms[spec.angle_query_index]
                if spec.angle_query_index is not None
                else None
            )
            prev2_local = (
                query_atoms[spec.torsion_query_index]
                if spec.torsion_query_index is not None
                else None
            )
            global_atoms = tuple(int(component.global_indices[idx]) for idx in query_atoms)
            unique[query_atoms] = FragmentMatch(
                query_atoms=query_atoms,
                global_atoms=global_atoms,
                anchor_global=int(component.global_indices[anchor_local]),
                anchor_local=anchor_local,
                prev_global=(
                    int(component.global_indices[prev_local]) if prev_local is not None else None
                ),
                prev_local=prev_local,
                prev2_global=(
                    int(component.global_indices[prev2_local]) if prev2_local is not None else None
                ),
                prev2_local=prev2_local,
            )
        return list(unique.values())

    def _contact_geometry(
        self,
        atoms: Atoms,
        component_a: MoleculeComponent,
        match_a: FragmentMatch,
        component_b: MoleculeComponent,
        match_b: FragmentMatch,
    ) -> Dict[str, Optional[float]]:
        anchor_a_global = match_a.anchor_global
        anchor_b_global = match_b.anchor_global
        raw_vector = atoms.positions[anchor_b_global] - atoms.positions[anchor_a_global]
        mic_vector, _ = find_mic(raw_vector, atoms.cell, atoms.pbc)
        mic_vector = np.asarray(mic_vector, dtype=float)
        distance = float(np.linalg.norm(mic_vector))

        anchor_a_position = component_a.positions_unwrapped[match_a.anchor_local]
        local_a = component_a.positions_unwrapped - anchor_a_position

        anchor_b_position = component_b.positions_unwrapped[match_b.anchor_local]
        local_b = component_b.positions_unwrapped - anchor_b_position + mic_vector

        origin = np.zeros(3, dtype=float)
        geometry = {
            "distance": distance,
            "angle_a": None,
            "angle_b": None,
            "torsion_a": None,
            "torsion_b": None,
        }
        if match_a.prev_local is not None:
            geometry["angle_a"] = _angle_degrees(
                local_a[match_a.prev_local],
                origin,
                mic_vector,
            )
        if match_b.prev_local is not None:
            geometry["angle_b"] = _angle_degrees(
                origin,
                mic_vector,
                local_b[match_b.prev_local],
            )
        if match_a.prev2_local is not None and match_a.prev_local is not None:
            geometry["torsion_a"] = _dihedral_degrees(
                local_a[match_a.prev2_local],
                local_a[match_a.prev_local],
                origin,
                mic_vector,
            )
        if match_b.prev2_local is not None and match_b.prev_local is not None:
            geometry["torsion_b"] = _dihedral_degrees(
                origin,
                mic_vector,
                local_b[match_b.prev_local],
                local_b[match_b.prev2_local],
            )
        return geometry

    def _canonical_contact_key(
        self,
        fragment_a_smarts: str,
        fragment_b_smarts: str,
        component_a: MoleculeComponent,
        match_a: FragmentMatch,
        component_b: MoleculeComponent,
        match_b: FragmentMatch,
    ) -> Tuple[Any, ...]:
        side_a = (
            int(component_a.molecule_id),
            tuple(sorted(int(value) for value in match_a.global_atoms)),
            int(match_a.anchor_global),
        )
        side_b = (
            int(component_b.molecule_id),
            tuple(sorted(int(value) for value in match_b.global_atoms)),
            int(match_b.anchor_global),
        )
        if fragment_a_smarts == fragment_b_smarts:
            ordered_sides = tuple(sorted((side_a, side_b)))
            return ("symmetric", fragment_a_smarts, ordered_sides)
        return ("directed", fragment_a_smarts, fragment_b_smarts, side_a, side_b)

    def _target_hybridization_matches(
        self,
        target_attrs: Dict[str, Any],
        query_hybridization: Optional[str],
    ) -> bool:
        query_value = str(query_hybridization or "").strip().lower()
        if not query_value:
            return True
        target_value = str(target_attrs.get("geometry_hybridization", "")).strip().lower()
        if not target_value or target_value in {"nan", "unknown"}:
            return False
        if query_value == "aromatic":
            return target_value in {"aromatic", "sp2"}
        if query_value == "sp2":
            return target_value in {"sp2", "aromatic"}
        return target_value == query_value

    def _query_node_matches(
        self,
        target_attrs: Dict[str, Any],
        query_attrs: Dict[str, Any],
        allow_hydrogen_wildcards: bool = True,
        strict_atom_types: bool = True,
    ) -> bool:
        if query_attrs.get("is_dummy"):
            return True
        query_element = str(query_attrs.get("element", ""))
        if query_element in ("", "*"):
            return True
        target_element = str(target_attrs.get("element", ""))
        if allow_hydrogen_wildcards and query_element == "H":
            return True
        if target_element != query_element:
            return False
        if strict_atom_types and not self._target_hybridization_matches(
            target_attrs,
            query_attrs.get("hybridization"),
        ):
            return False
        return True

    def _query_edge_matches(
        self,
        target_attrs: Dict[str, Any],
        query_attrs: Dict[str, Any],
        strict_bonds: bool = False,
    ) -> bool:
        if not strict_bonds:
            return True
        query_order = str(query_attrs.get("bond_order", "unknown"))
        target_order = str(target_attrs.get("bond_order", "unknown"))
        if query_order in ("unknown", "dummy", "not_connected"):
            return True
        if target_order == query_order:
            return True
        if query_order == "aromatic" and target_order in {"aromatic", "unknown"}:
            return True
        return target_order == "unknown"

    def _mol2_candidate_ids_from_cache(
        self,
        summary_df: pd.DataFrame,
        query_graph: Any,
        allow_hydrogen_wildcards: bool = True,
        strict_bonds: bool = False,
    ) -> List[int]:
        query_fp = graph_fingerprint(query_graph, ignore_dummy=True)
        required_counts = Counter(query_fp["element_counts"])
        if allow_hydrogen_wildcards and "H" in required_counts:
            del required_counts["H"]
        required_edges = set(query_fp["edge_keys"] if strict_bonds else query_fp["generic_edge_keys"])

        candidates: List[int] = []
        for row in summary_df.to_dict("records"):
            if row.get("graph_status") != "ok":
                continue
            structure_counts = Counter(
                _json_loads_or_empty(row.get("element_counts_json"), {})
            )
            if any(structure_counts.get(element, 0) < count for element, count in required_counts.items()):
                continue
            edge_field = "edge_keys_json" if strict_bonds else "generic_edge_keys_json"
            structure_edges = set(_json_loads_or_empty(row.get(edge_field), []))
            if required_edges and not required_edges.issubset(structure_edges):
                continue
            candidates.append(int(row["structure_id"]))
        return candidates

    def _fragment_graph_from_mol2_for_contact(
        self,
        mol2_path: Union[str, Path],
    ) -> Tuple[Any, Optional[int], Set[str]]:
        query = parse_mol2_file(mol2_path)
        full_graph = mol2_query_to_graph(query, drop_dummy=False)
        dummy_nodes = [
            node
            for node, attrs in full_graph.nodes(data=True)
            if attrs.get("is_dummy")
        ]
        contact_elements: Set[str] = set()
        anchor_query_node: Optional[int] = None
        fragment_graph = full_graph.copy()
        if dummy_nodes:
            dummy_node = dummy_nodes[0]
            neighbors = list(full_graph.neighbors(dummy_node))
            if neighbors:
                anchor_query_node = int(neighbors[0])
            for node in dummy_nodes:
                for neighbor in full_graph.neighbors(node):
                    element = str(full_graph.nodes[node].get("element", "*"))
                    if element and element != "*":
                        contact_elements.add(element)
                if node in fragment_graph:
                    fragment_graph.remove_node(node)
        if anchor_query_node is None and fragment_graph.nodes:
            anchor_query_node = int(next(iter(fragment_graph.nodes)))
        return fragment_graph, anchor_query_node, contact_elements

    def _component_id_for_node(self, graph: Any, node: int) -> int:
        value = graph.nodes[node].get("component_id")
        if value is not None and not (isinstance(value, float) and np.isnan(value)):
            return int(value)
        for component_id, component_nodes in enumerate(nx.connected_components(graph)):
            if node in component_nodes:
                return int(component_id)
        return -1

    def _query_predecessor_nodes(
        self,
        query_graph: Any,
        anchor_query_node: int,
    ) -> Tuple[Optional[int], Optional[int]]:
        neighbors = sorted(int(node) for node in query_graph.neighbors(anchor_query_node))
        prev = neighbors[0] if neighbors else None
        prev2 = None
        if prev is not None:
            second_neighbors = sorted(
                int(node)
                for node in query_graph.neighbors(prev)
                if int(node) != int(anchor_query_node)
            )
            prev2 = second_neighbors[0] if second_neighbors else None
        return prev, prev2

    def _contact_base_label(
        self,
        query_graph: Any,
        anchor_query_node: int,
        prev_query_node: Optional[int],
    ) -> str:
        anchor_element = str(query_graph.nodes[anchor_query_node].get("element", "*"))
        if prev_query_node is None:
            return anchor_element
        prev_element = str(query_graph.nodes[prev_query_node].get("element", "*"))
        return f"{prev_element}-{anchor_element}"

    def _safe_label(self, label: str) -> str:
        safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(label).strip())
        return safe.strip("_") or "unknown"

    def _query_match_order(self, query_graph: Any, anchor_query_node: int) -> List[int]:
        order: List[int] = [int(anchor_query_node)]
        in_order: Set[int] = {int(anchor_query_node)}
        while len(order) < query_graph.number_of_nodes():
            candidates = [
                int(node)
                for node in query_graph.nodes
                if int(node) not in in_order
                and any(int(neighbor) in in_order for neighbor in query_graph.neighbors(node))
            ]
            if not candidates:
                return []
            candidates.sort(
                key=lambda node: (
                    -sum(1 for neighbor in query_graph.neighbors(node) if int(neighbor) in in_order),
                    -int(query_graph.degree(node)),
                    str(query_graph.nodes[node].get("element", "")),
                    int(node),
                )
            )
            chosen = int(candidates[0])
            order.append(chosen)
            in_order.add(chosen)
        return order

    def _fast_anchor_fragment_matches(
        self,
        target_graph: Any,
        query_graph: Any,
        anchor_query_node: int,
        allow_hydrogen_wildcards: bool = True,
        strict_atom_types: bool = True,
        strict_bonds: bool = False,
    ) -> Iterable[Dict[int, int]]:
        order = self._query_match_order(query_graph, int(anchor_query_node))
        if not order:
            return

        target_nodes = [int(node) for node in target_graph.nodes]
        candidate_cache: Dict[int, Set[int]] = {}
        for query_node in order:
            query_attrs = query_graph.nodes[int(query_node)]
            candidate_cache[int(query_node)] = {
                int(target_node)
                for target_node in target_nodes
                if self._query_node_matches(
                    target_graph.nodes[int(target_node)],
                    query_attrs,
                    allow_hydrogen_wildcards=allow_hydrogen_wildcards,
                    strict_atom_types=strict_atom_types,
                )
            }
            if not candidate_cache[int(query_node)]:
                return

        mapping: Dict[int, int] = {}
        used_targets: Set[int] = set()

        def recurse(position: int) -> Iterable[Dict[int, int]]:
            if position >= len(order):
                yield dict(mapping)
                return

            query_node = int(order[position])
            mapped_neighbors = [
                int(neighbor)
                for neighbor in query_graph.neighbors(query_node)
                if int(neighbor) in mapping
            ]
            if mapped_neighbors:
                candidate_targets: Optional[Set[int]] = None
                for query_neighbor in mapped_neighbors:
                    target_neighbor = int(mapping[query_neighbor])
                    neighbor_targets = {int(node) for node in target_graph.neighbors(target_neighbor)}
                    candidate_targets = (
                        neighbor_targets
                        if candidate_targets is None
                        else candidate_targets & neighbor_targets
                    )
                candidates = candidate_targets or set()
                candidates &= candidate_cache[query_node]
            else:
                candidates = set(candidate_cache[query_node])

            for target_node in sorted(candidates):
                target_node = int(target_node)
                if target_node in used_targets:
                    continue
                edge_ok = True
                for query_neighbor in mapped_neighbors:
                    target_neighbor = int(mapping[query_neighbor])
                    if not target_graph.has_edge(target_node, target_neighbor):
                        edge_ok = False
                        break
                    if not self._query_edge_matches(
                        target_graph.edges[target_node, target_neighbor],
                        query_graph.edges[query_node, query_neighbor],
                        strict_bonds=strict_bonds,
                    ):
                        edge_ok = False
                        break
                if not edge_ok:
                    continue

                mapping[query_node] = target_node
                used_targets.add(target_node)
                yield from recurse(position + 1)
                used_targets.remove(target_node)
                del mapping[query_node]

        yield from recurse(0)

    def _contact_candidate_nodes(
        self,
        target_graph: Any,
        contact_element_set: Set[str],
        wildcard_contact: bool,
    ) -> List[Tuple[int, str, int]]:
        candidates: List[Tuple[int, str, int]] = []
        for node, attrs in target_graph.nodes(data=True):
            element = str(attrs.get("element", ""))
            if not wildcard_contact and element not in contact_element_set:
                continue
            candidates.append(
                (
                    int(node),
                    element,
                    self._component_id_for_node(target_graph, int(node)),
                )
            )
        return candidates

    def _normalize_contact_scope(self, contact_scope: Optional[str] = "intermolecular") -> str:
        value = str(contact_scope or "intermolecular").strip().lower()
        if value in {"intermolecular", "inter", "intercomponent", "different_component"}:
            return "intermolecular"
        if value in {"all", "any", "both", "inter_and_intra", "intermolecular_and_intramolecular"}:
            return "all"
        raise ValueError(
            "contact_scope must be 'intermolecular' or 'all', "
            f"got {contact_scope!r}"
        )

    def _search_mol2_contacts_fast_anchor(
        self,
        fragment_mol2: Union[str, Path],
        cache_dir: Path,
        nodes_df: pd.DataFrame,
        edges_df: pd.DataFrame,
        candidate_ids: Sequence[int],
        query_graph: Any,
        anchor_query_node: int,
        contact_element_set: Set[str],
        wildcard_contact: bool,
        radius_max: float = 4.0,
        strict_bonds: bool = False,
        strict_atom_types: bool = True,
        allow_hydrogen_wildcards: bool = True,
        contact_scope: str = "intermolecular",
        progress_every: int = 100,
    ) -> Dict[str, Any]:
        contact_scope = self._normalize_contact_scope(contact_scope)
        results: List[Dict[str, Any]] = []
        structures_with_hits = 0
        prev_query_node, prev2_query_node = self._query_predecessor_nodes(
            query_graph,
            int(anchor_query_node),
        )
        contact_base_label = self._contact_base_label(
            query_graph,
            int(anchor_query_node),
            prev_query_node,
        )

        candidate_id_set = {int(structure_id) for structure_id in candidate_ids}
        node_groups = {
            int(structure_id): group
            for structure_id, group in nodes_df[
                nodes_df["structure_id"].isin(candidate_id_set)
            ].groupby("structure_id", sort=False)
        }
        edge_groups = {
            int(structure_id): group
            for structure_id, group in edges_df[
                edges_df["structure_id"].isin(candidate_id_set)
            ].groupby("structure_id", sort=False)
        }

        for idx, structure_id in enumerate(candidate_ids, start=1):
            structure_nodes = node_groups.get(int(structure_id))
            structure_edges = edge_groups.get(int(structure_id))
            if structure_nodes is None:
                continue
            if structure_edges is None:
                structure_edges = edges_df.iloc[0:0]
            if structure_nodes.empty:
                continue
            target_graph = self._graph_from_cache_rows(structure_nodes, structure_edges)
            contact_candidates = self._contact_candidate_nodes(
                target_graph,
                contact_element_set,
                wildcard_contact,
            )
            if not contact_candidates:
                continue

            atoms: Optional[Atoms] = None
            metadata: Optional[Dict[str, Any]] = None
            positions: Optional[np.ndarray] = None
            structure_hits = 0
            seen_keys: Set[Tuple[int, int, Tuple[int, ...]]] = set()
            seen_fragment_keys: Set[Tuple[Tuple[int, int], ...]] = set()

            for inverse_mapping in self._fast_anchor_fragment_matches(
                target_graph=target_graph,
                query_graph=query_graph,
                anchor_query_node=int(anchor_query_node),
                allow_hydrogen_wildcards=allow_hydrogen_wildcards,
                strict_atom_types=strict_atom_types,
                strict_bonds=strict_bonds,
            ):
                fragment_key = tuple(sorted((int(q), int(t)) for q, t in inverse_mapping.items()))
                if fragment_key in seen_fragment_keys:
                    continue
                seen_fragment_keys.add(fragment_key)

                if int(anchor_query_node) not in inverse_mapping:
                    continue
                matched_atoms = tuple(sorted(int(value) for value in inverse_mapping.values()))
                anchor_global = int(inverse_mapping[int(anchor_query_node)])
                anchor_component = self._component_id_for_node(target_graph, anchor_global)
                prev_global = (
                    inverse_mapping.get(prev_query_node)
                    if prev_query_node is not None
                    else None
                )
                prev2_global = (
                    inverse_mapping.get(prev2_query_node)
                    if prev2_query_node is not None
                    else None
                )

                if atoms is None or metadata is None or positions is None:
                    atoms, metadata = self.get_structure(int(structure_id))
                    if atoms is None or metadata is None:
                        break
                    positions = atoms.get_positions()

                eligible_contacts = [
                    (int(contact_node), str(contact_element), int(contact_component))
                    for contact_node, contact_element, contact_component in contact_candidates
                    if int(contact_node) not in matched_atoms
                    and (
                        contact_scope == "all"
                        or int(contact_component) != int(anchor_component)
                    )
                ]
                if not eligible_contacts:
                    continue

                contact_nodes = np.asarray(
                    [contact_node for contact_node, _, _ in eligible_contacts],
                    dtype=int,
                )
                raw_vectors = positions[contact_nodes] - positions[anchor_global]
                mic_vectors, _ = find_mic(raw_vectors, atoms.cell, atoms.pbc)
                distances = np.linalg.norm(mic_vectors, axis=1)
                close_indices = np.flatnonzero(distances <= float(radius_max))

                for close_idx in close_indices.tolist():
                    contact_node, contact_element, contact_component = eligible_contacts[int(close_idx)]
                    same_component = int(contact_component) == int(anchor_component)
                    contact_relation = "intracomponent" if same_component else "intercomponent"
                    mic_vector = np.asarray(mic_vectors[int(close_idx)], dtype=float)
                    raw_vector = np.asarray(raw_vectors[int(close_idx)], dtype=float)
                    distance = float(distances[int(close_idx)])

                    contact_key = (anchor_global, int(contact_node), matched_atoms)
                    if contact_key in seen_keys:
                        continue
                    seen_keys.add(contact_key)

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
                    donor_contact_distance = None
                    prev_position = None
                    if prev_global is not None:
                        prev_vector, _ = find_mic(
                            positions[int(prev_global)] - anchor_position,
                            atoms.cell,
                            atoms.pbc,
                        )
                        prev_position = anchor_position + prev_vector
                        angle = _angle_degrees(
                            prev_position,
                            anchor_position,
                            contact_position,
                        )
                        donor_contact_distance = float(
                            np.linalg.norm(contact_position - prev_position)
                        )
                    if prev_position is not None and prev2_global is not None:
                        prev2_vector, _ = find_mic(
                            positions[int(prev2_global)] - prev_position,
                            atoms.cell,
                            atoms.pbc,
                        )
                        prev2_position = prev_position + prev2_vector
                        torsion = _dihedral_degrees(
                            prev2_position,
                            prev_position,
                            anchor_position,
                            contact_position,
                        )

                    contact_label = f"{contact_base_label}...{contact_element}"
                    structure_hits += 1
                    results.append(
                        {
                            "structure_id": int(structure_id),
                            "refcode": metadata.get("refcode") if metadata else None,
                            "fragment_mol2": str(fragment_mol2),
                            "matched_atoms": [int(value) + 1 for value in matched_atoms],
                            "matched_atoms_zero_based": [int(value) for value in matched_atoms],
                            "matched_elements": [
                                str(target_graph.nodes[int(value)].get("element", ""))
                                for value in matched_atoms
                            ],
                            "prev_atom": int(prev_global + 1) if prev_global is not None else None,
                            "prev_atom_zero_based": (
                                int(prev_global) if prev_global is not None else None
                            ),
                            "prev_element": (
                                str(target_graph.nodes[int(prev_global)].get("element", ""))
                                if prev_global is not None
                                else None
                            ),
                            "anchor_atom": int(anchor_global + 1),
                            "anchor_atom_zero_based": int(anchor_global),
                            "anchor_element": str(
                                target_graph.nodes[int(anchor_global)].get("element", "")
                            ),
                            "anchor_component": int(anchor_component),
                            "contact_atom": int(contact_node + 1),
                            "contact_atom_zero_based": int(contact_node),
                            "contact_element": contact_element,
                            "contact_component": int(contact_component),
                            "contact_scope": contact_scope,
                            "same_component": bool(same_component),
                            "contact_relation": contact_relation,
                            "contact_label": contact_label,
                            "distance": distance,
                            "anchor_contact_distance": distance,
                            "donor_contact_distance": donor_contact_distance,
                            "contact_offset_x": int(contact_offset[0]),
                            "contact_offset_y": int(contact_offset[1]),
                            "contact_offset_z": int(contact_offset[2]),
                            "prev_x": (
                                float(positions[int(prev_global)][0])
                                if prev_global is not None
                                else None
                            ),
                            "prev_y": (
                                float(positions[int(prev_global)][1])
                                if prev_global is not None
                                else None
                            ),
                            "prev_z": (
                                float(positions[int(prev_global)][2])
                                if prev_global is not None
                                else None
                            ),
                            "prev_image_x": (
                                float(prev_position[0]) if prev_position is not None else None
                            ),
                            "prev_image_y": (
                                float(prev_position[1]) if prev_position is not None else None
                            ),
                            "prev_image_z": (
                                float(prev_position[2]) if prev_position is not None else None
                            ),
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
                    )

            if structure_hits > 0:
                structures_with_hits += 1
            if progress_every and idx % int(progress_every) == 0:
                print(
                    f"Processed {idx}/{len(candidate_ids)} graph candidates, "
                    f"contacts so far: {len(results)}",
                    flush=True,
                )

        return {
            "summary": {
                "mode": "mol2_contact",
                "search_backend": "fast_anchor",
                "graph_cache": str(cache_dir),
                "fragment_mol2": str(fragment_mol2),
                "scanned_structures": len(candidate_ids),
                "structures_with_hits": structures_with_hits,
                "contacts_found": len(results),
                "radius_max": float(radius_max),
                "contact_scope": contact_scope,
                "contact_elements": sorted(contact_element_set) if contact_element_set else ["*"],
                "strict_bonds": bool(strict_bonds),
                "strict_atom_types": bool(strict_atom_types),
                "allow_hydrogen_wildcards": bool(allow_hydrogen_wildcards),
            },
            "results": results,
        }

    def search_mol2_contacts(
        self,
        fragment_mol2: Union[str, Path],
        graph_cache_path: Optional[Union[str, Path]] = None,
        build_cache_if_missing: bool = True,
        rebuild_cache: bool = False,
        cache_max_structures: Optional[int] = 100,
        cache_max_atoms: Optional[int] = 1000,
        cache_min_nonbonded_distance: float = 0.6,
        cache_skip_extended_networks: bool = True,
        cache_component_filter_backend: str = "pymatgen",
        cache_bond_order_backend: str = "rdkit",
        cache_flush_every: int = 250,
        cache_workers: int = 1,
        cache_worker_chunk_size: int = 10,
        structure_ids: Optional[Sequence[int]] = None,
        refcodes: Optional[Sequence[str]] = None,
        radius_max: float = 4.0,
        contact_elements: Optional[Sequence[str]] = None,
        contact_scope: str = "intermolecular",
        covalent_scale: float = 1.15,
        strict_bonds: bool = False,
        strict_atom_types: bool = True,
        allow_hydrogen_wildcards: bool = True,
        distance_min: Optional[float] = None,
        angle_min: Optional[float] = None,
        search_backend: str = "fast_anchor",
        max_structures: Optional[int] = None,
        progress_every: int = 100,
    ) -> Dict[str, Any]:
        del covalent_scale
        contact_scope = self._normalize_contact_scope(contact_scope)
        if not NETWORKX_AVAILABLE:
            raise ImportError("networkx is required for MOL2 graph search")

        cache_dir = self._graph_cache_dir(graph_cache_path)
        if rebuild_cache:
            self.build_graph_cache(
                graph_cache_path=cache_dir,
                max_structures=cache_max_structures,
                max_atoms=cache_max_atoms,
                min_nonbonded_distance=cache_min_nonbonded_distance,
                skip_extended_networks=cache_skip_extended_networks,
                component_filter_backend=cache_component_filter_backend,
                bond_order_backend=cache_bond_order_backend,
                overwrite=True,
                progress_every=progress_every,
                flush_every=cache_flush_every,
                workers=cache_workers,
                worker_chunk_size=cache_worker_chunk_size,
            )
            summary_df, nodes_df, edges_df = self._load_graph_cache(cache_dir)
        else:
            try:
                summary_df, nodes_df, edges_df = self._load_graph_cache(cache_dir)
            except FileNotFoundError:
                if not build_cache_if_missing:
                    raise
                self.build_graph_cache(
                    graph_cache_path=cache_dir,
                    max_structures=cache_max_structures,
                    max_atoms=cache_max_atoms,
                    min_nonbonded_distance=cache_min_nonbonded_distance,
                    skip_extended_networks=cache_skip_extended_networks,
                    component_filter_backend=cache_component_filter_backend,
                    bond_order_backend=cache_bond_order_backend,
                    overwrite=False,
                    progress_every=progress_every,
                    flush_every=cache_flush_every,
                    workers=cache_workers,
                    worker_chunk_size=cache_worker_chunk_size,
                )
                summary_df, nodes_df, edges_df = self._load_graph_cache(cache_dir)

        rebuild_reason = self._graph_cache_rebuild_reason(
            cache_dir,
            summary_df,
            cache_min_nonbonded_distance,
        )
        if rebuild_reason:
            if not build_cache_if_missing:
                raise RuntimeError(
                    "Existing graph cache is not compatible with the requested geometry filter: "
                    f"{rebuild_reason}"
                )
            print(
                "Existing graph cache is not compatible with the requested geometry filter; "
                f"rebuilding ({rebuild_reason})",
                flush=True,
            )
            self.build_graph_cache(
                graph_cache_path=cache_dir,
                max_structures=cache_max_structures,
                max_atoms=cache_max_atoms,
                min_nonbonded_distance=cache_min_nonbonded_distance,
                skip_extended_networks=cache_skip_extended_networks,
                component_filter_backend=cache_component_filter_backend,
                bond_order_backend=cache_bond_order_backend,
                overwrite=True,
                progress_every=progress_every,
                flush_every=cache_flush_every,
                workers=cache_workers,
                worker_chunk_size=cache_worker_chunk_size,
            )
            summary_df, nodes_df, edges_df = self._load_graph_cache(cache_dir)

        query_graph, anchor_query_node, dummy_contact_elements = (
            self._fragment_graph_from_mol2_for_contact(fragment_mol2)
        )
        if not query_graph.nodes:
            raise ValueError("MOL2 contact query has no searchable fragment atoms")
        if anchor_query_node is None:
            raise ValueError("MOL2 contact query has no anchor atom")
        contact_element_set = {
            str(element).strip()
            for element in (contact_elements or [])
            if str(element).strip()
        }
        if not contact_element_set:
            contact_element_set = dummy_contact_elements
        wildcard_contact = not contact_element_set or "*" in contact_element_set

        candidate_ids = self._mol2_candidate_ids_from_cache(
            summary_df=summary_df,
            query_graph=query_graph,
            allow_hydrogen_wildcards=allow_hydrogen_wildcards,
            strict_bonds=strict_bonds,
        )
        requested_ids = set(self._iter_smarts_structure_ids(structure_ids, refcodes, max_structures))
        if structure_ids or refcodes or max_structures is not None:
            candidate_ids = [structure_id for structure_id in candidate_ids if structure_id in requested_ids]

        backend_name = str(search_backend or "fast_anchor").strip().lower()
        if backend_name in {"fast", "fast_anchor", "anchor"}:
            return self._search_mol2_contacts_fast_anchor(
                fragment_mol2=fragment_mol2,
                cache_dir=cache_dir,
                nodes_df=nodes_df,
                edges_df=edges_df,
                candidate_ids=candidate_ids,
                query_graph=query_graph,
                anchor_query_node=int(anchor_query_node),
                contact_element_set=contact_element_set,
                wildcard_contact=wildcard_contact,
                radius_max=radius_max,
                strict_bonds=strict_bonds,
                strict_atom_types=strict_atom_types,
                allow_hydrogen_wildcards=allow_hydrogen_wildcards,
                contact_scope=contact_scope,
                progress_every=progress_every,
            )

        results: List[Dict[str, Any]] = []
        structures_with_hits = 0
        prev_query_node, prev2_query_node = self._query_predecessor_nodes(
            query_graph,
            int(anchor_query_node),
        )
        contact_base_label = self._contact_base_label(
            query_graph,
            int(anchor_query_node),
            prev_query_node,
        )

        for idx, structure_id in enumerate(candidate_ids, start=1):
            structure_nodes = nodes_df.loc[nodes_df["structure_id"] == int(structure_id)]
            structure_edges = edges_df.loc[edges_df["structure_id"] == int(structure_id)]
            if structure_nodes.empty:
                continue
            target_graph = self._graph_from_cache_rows(structure_nodes, structure_edges)
            matcher = nx.algorithms.isomorphism.GraphMatcher(
                target_graph,
                query_graph,
                node_match=lambda target_attrs, query_attrs: self._query_node_matches(
                    target_attrs,
                    query_attrs,
                    allow_hydrogen_wildcards=allow_hydrogen_wildcards,
                    strict_atom_types=strict_atom_types,
                ),
                edge_match=lambda target_attrs, query_attrs: self._query_edge_matches(
                    target_attrs,
                    query_attrs,
                    strict_bonds=strict_bonds,
                ),
            )
            atoms, metadata = self.get_structure(int(structure_id))
            if atoms is None or metadata is None:
                continue
            positions = atoms.get_positions()
            structure_hits = 0
            seen_keys: Set[Tuple[int, int, Tuple[int, ...]]] = set()
            for mapping in matcher.subgraph_isomorphisms_iter():
                inverse_mapping = {int(query_node): int(target_node) for target_node, query_node in mapping.items()}
                if int(anchor_query_node) not in inverse_mapping:
                    continue
                matched_atoms = tuple(sorted(int(value) for value in inverse_mapping.values()))
                anchor_global = inverse_mapping[int(anchor_query_node)]
                anchor_component = self._component_id_for_node(target_graph, anchor_global)
                prev_global = inverse_mapping.get(prev_query_node) if prev_query_node is not None else None
                prev2_global = inverse_mapping.get(prev2_query_node) if prev2_query_node is not None else None

                for contact_node, contact_attrs in target_graph.nodes(data=True):
                    contact_node = int(contact_node)
                    if contact_node in matched_atoms:
                        continue
                    contact_component = self._component_id_for_node(target_graph, contact_node)
                    same_component = int(contact_component) == int(anchor_component)
                    if contact_scope == "intermolecular" and same_component:
                        continue
                    contact_element = str(contact_attrs.get("element", ""))
                    if not wildcard_contact and contact_element not in contact_element_set:
                        continue
                    contact_relation = "intracomponent" if same_component else "intercomponent"
                    contact_label = f"{contact_base_label}...{contact_element}"
                    raw_vector = positions[contact_node] - positions[anchor_global]
                    mic_vector, _ = find_mic(raw_vector, atoms.cell, atoms.pbc)
                    distance = float(np.linalg.norm(mic_vector))
                    if distance_min is not None and distance < float(distance_min):
                        continue
                    if distance > float(radius_max):
                        continue
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
                    donor_contact_distance = None
                    prev_position = None
                    if prev_global is not None:
                        prev_vector, _ = find_mic(
                            positions[int(prev_global)] - anchor_position,
                            atoms.cell,
                            atoms.pbc,
                        )
                        prev_position = anchor_position + prev_vector
                        angle = _angle_degrees(
                            prev_position,
                            anchor_position,
                            contact_position,
                        )
                        donor_contact_distance = float(
                            np.linalg.norm(contact_position - prev_position)
                        )
                    if angle_min is not None and (angle is None or angle < float(angle_min)):
                        continue
                    if prev_position is not None and prev2_global is not None:
                        prev2_vector, _ = find_mic(
                            positions[int(prev2_global)] - prev_position,
                            atoms.cell,
                            atoms.pbc,
                        )
                        prev2_position = prev_position + prev2_vector
                        torsion = _dihedral_degrees(
                            prev2_position,
                            prev_position,
                            anchor_position,
                            contact_position,
                        )
                    contact_key = (anchor_global, contact_node, matched_atoms)
                    if contact_key in seen_keys:
                        continue
                    seen_keys.add(contact_key)
                    structure_hits += 1
                    results.append(
                        {
                            "structure_id": int(structure_id),
                            "refcode": metadata.get("refcode"),
                            "fragment_mol2": str(fragment_mol2),
                            "matched_atoms": [int(value) + 1 for value in matched_atoms],
                            "matched_atoms_zero_based": [int(value) for value in matched_atoms],
                            "matched_elements": [
                                str(target_graph.nodes[int(value)].get("element", ""))
                                for value in matched_atoms
                            ],
                            "prev_atom": int(prev_global + 1) if prev_global is not None else None,
                            "prev_atom_zero_based": (
                                int(prev_global) if prev_global is not None else None
                            ),
                            "prev_element": (
                                str(target_graph.nodes[int(prev_global)].get("element", ""))
                                if prev_global is not None
                                else None
                            ),
                            "anchor_atom": int(anchor_global + 1),
                            "anchor_atom_zero_based": int(anchor_global),
                            "anchor_element": str(
                                target_graph.nodes[int(anchor_global)].get("element", "")
                            ),
                            "anchor_component": int(anchor_component),
                            "contact_atom": int(contact_node + 1),
                            "contact_atom_zero_based": int(contact_node),
                            "contact_element": contact_element,
                            "contact_component": int(contact_component),
                            "contact_scope": contact_scope,
                            "same_component": bool(same_component),
                            "contact_relation": contact_relation,
                            "contact_label": contact_label,
                            "distance": distance,
                            "anchor_contact_distance": distance,
                            "donor_contact_distance": donor_contact_distance,
                            "contact_offset_x": int(contact_offset[0]),
                            "contact_offset_y": int(contact_offset[1]),
                            "contact_offset_z": int(contact_offset[2]),
                            "prev_x": (
                                float(positions[int(prev_global)][0])
                                if prev_global is not None
                                else None
                            ),
                            "prev_y": (
                                float(positions[int(prev_global)][1])
                                if prev_global is not None
                                else None
                            ),
                            "prev_z": (
                                float(positions[int(prev_global)][2])
                                if prev_global is not None
                                else None
                            ),
                            "prev_image_x": (
                                float(prev_position[0]) if prev_position is not None else None
                            ),
                            "prev_image_y": (
                                float(prev_position[1]) if prev_position is not None else None
                            ),
                            "prev_image_z": (
                                float(prev_position[2]) if prev_position is not None else None
                            ),
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
                    )
            if structure_hits > 0:
                structures_with_hits += 1
            if progress_every and idx % int(progress_every) == 0:
                print(
                    f"Processed {idx}/{len(candidate_ids)} graph candidates, "
                    f"contacts so far: {len(results)}"
                )

        return {
            "summary": {
                "mode": "mol2_contact",
                "search_backend": "graphmatcher",
                "graph_cache": str(cache_dir),
                "fragment_mol2": str(fragment_mol2),
                "scanned_structures": len(candidate_ids),
                "structures_with_hits": structures_with_hits,
                "contacts_found": len(results),
                "radius_max": float(radius_max),
                "distance_min": float(distance_min) if distance_min is not None else None,
                "angle_min": float(angle_min) if angle_min is not None else None,
                "contact_scope": contact_scope,
                "contact_elements": sorted(contact_element_set) if contact_element_set else ["*"],
                "strict_bonds": bool(strict_bonds),
                "strict_atom_types": bool(strict_atom_types),
                "allow_hydrogen_wildcards": bool(allow_hydrogen_wildcards),
            },
            "results": results,
        }

    def _normalize_smarts_fragment_filter(self, condition: Any) -> Dict[str, Any]:
        if isinstance(condition, str):
            return {
                "smarts": condition.strip(),
                "covalent_scale": 1.15,
                "progress_every": 0,
            }
        if isinstance(condition, dict):
            smarts = str(condition.get("smarts", condition.get("value", ""))).strip()
            return {
                "smarts": smarts,
                "covalent_scale": float(condition.get("covalent_scale", 1.15)),
                "progress_every": int(condition.get("progress_every", 0)),
            }
        raise ValueError("smarts_fragment must be a SMARTS string or object with 'smarts'")

    def search_smarts_fragment(
        self,
        smarts: str,
        structure_ids: Optional[Sequence[int]] = None,
        refcodes: Optional[Sequence[str]] = None,
        covalent_scale: float = 1.15,
        max_structures: Optional[int] = None,
        progress_every: int = 100,
        return_matches: bool = False,
    ) -> Union[List[int], Dict[str, Any]]:
        fragment_spec = compile_fragment("F", smarts)
        scan_ids = self._iter_smarts_structure_ids(
            structure_ids=structure_ids,
            refcodes=refcodes,
            max_structures=max_structures,
        )
        scan_ids = self._prefilter_smarts_structure_ids(
            scan_ids,
            min_atoms=int(fragment_spec.atom_count),
            required_atomic_numbers=fragment_spec.exact_atomic_number_requirements,
        )
        matched_ids: List[int] = []
        match_payloads: List[Dict[str, Any]] = []

        for idx, structure_id in enumerate(scan_ids, start=1):
            atoms, metadata = self.get_structure(int(structure_id))
            if atoms is None or metadata is None:
                continue
            numbers = np.asarray(atoms.get_atomic_numbers(), dtype=int)
            if numbers.size < fragment_spec.atom_count:
                continue
            if not _atomic_number_requirements_satisfied(
                _atomic_number_counts(numbers),
                fragment_spec.exact_atomic_number_requirements,
            ):
                continue
            components = self._split_into_molecules(atoms, covalent_scale=covalent_scale)
            structure_has_match = False
            for component in components:
                matches = self._find_fragment_matches(component, fragment_spec)
                if not matches:
                    continue
                structure_has_match = True
                if return_matches:
                    for match in matches:
                        match_payloads.append(
                            {
                                "structure_id": int(structure_id),
                                "refcode": metadata.get("refcode"),
                                "molecule_id": int(component.molecule_id),
                                "smarts": fragment_spec.smarts,
                                "matched_atoms": [int(value) + 1 for value in match.global_atoms],
                                "anchor_atom": int(match.anchor_global + 1),
                            }
                        )
            if structure_has_match:
                matched_ids.append(int(structure_id))
            if progress_every and idx % int(progress_every) == 0:
                print(
                    f"Processed {idx}/{len(scan_ids)} structures for SMARTS fragment, "
                    f"hits so far: {len(matched_ids)}"
                )

        if return_matches:
            return {
                "summary": {
                    "mode": "smarts_fragment",
                    "scanned_structures": len(scan_ids),
                    "structures_with_hits": len(matched_ids),
                    "matches_found": len(match_payloads),
                    "smarts": fragment_spec.smarts,
                    "covalent_scale": float(covalent_scale),
                },
                "results": match_payloads,
                "structure_ids": matched_ids,
            }
        return matched_ids

    def _iter_smarts_structure_ids(
        self,
        structure_ids: Optional[Sequence[int]] = None,
        refcodes: Optional[Sequence[str]] = None,
        max_structures: Optional[int] = None,
    ) -> List[int]:
        structure_ids = structure_ids or []
        refcodes = refcodes or []
        if structure_ids:
            result = [int(value) for value in structure_ids]
        elif refcodes:
            result = []
            seen: Set[int] = set()
            for refcode in refcodes:
                record = self._load_refcode_record(str(refcode).strip())
                if record is None:
                    continue
                structure_id = int(record["structure_id"])
                if structure_id not in seen:
                    seen.add(structure_id)
                    result.append(structure_id)
        else:
            result = list(range(self.count()))
        if max_structures is not None:
            result = result[: int(max_structures)]
        return result

    def search_smarts_contacts(
        self,
        fragment_a: str,
        fragment_b: str,
        radius_max: float = 4.0,
        structure_ids: Optional[Sequence[int]] = None,
        refcodes: Optional[Sequence[str]] = None,
        covalent_scale: float = 1.15,
        max_structures: Optional[int] = None,
        progress_every: int = 100,
    ) -> Dict[str, Any]:
        fragment_a_spec = compile_fragment("A", fragment_a)
        fragment_b_spec = compile_fragment("B", fragment_b)
        same_fragment = fragment_a_spec.smarts == fragment_b_spec.smarts
        min_structure_atoms = int(fragment_a_spec.atom_count + fragment_b_spec.atom_count)
        combined_atomic_requirements = _merge_atomic_number_requirements(
            fragment_a_spec.exact_atomic_number_requirements,
            fragment_b_spec.exact_atomic_number_requirements,
        )
        scan_ids = self._iter_smarts_structure_ids(
            structure_ids=structure_ids,
            refcodes=refcodes,
            max_structures=max_structures,
        )
        scan_ids = self._prefilter_smarts_structure_ids(
            scan_ids,
            min_atoms=min_structure_atoms,
            required_atomic_numbers=combined_atomic_requirements,
        )
        results: List[Dict[str, Any]] = []
        structures_with_hits = 0

        for idx, structure_id in enumerate(scan_ids, start=1):
            atoms, metadata = self.get_structure(int(structure_id))
            if atoms is None or metadata is None:
                continue
            numbers = np.asarray(atoms.get_atomic_numbers(), dtype=int)
            if numbers.size < min_structure_atoms:
                continue
            if not _atomic_number_requirements_satisfied(
                _atomic_number_counts(numbers),
                combined_atomic_requirements,
            ):
                continue
            components = self._split_into_molecules(atoms, covalent_scale=covalent_scale)
            seen_contact_keys: Set[Tuple[Any, ...]] = set()
            matches_a_by_component: Dict[int, List[FragmentMatch]] = {}
            matches_b_by_component: Dict[int, List[FragmentMatch]] = {}
            for component in components:
                matches_a = self._find_fragment_matches(component, fragment_a_spec)
                if matches_a:
                    matches_a_by_component[component.search_component_id] = matches_a
                if same_fragment:
                    if matches_a:
                        matches_b_by_component[component.search_component_id] = matches_a
                else:
                    matches_b = self._find_fragment_matches(component, fragment_b_spec)
                    if matches_b:
                        matches_b_by_component[component.search_component_id] = matches_b

            structure_hits = 0
            for component_index, component_a in enumerate(components):
                comp_a_matches = matches_a_by_component.get(component_a.search_component_id)
                if not comp_a_matches:
                    continue
                component_b_iterable = (
                    components[component_index + 1:] if same_fragment else components
                )
                for component_b in component_b_iterable:
                    if component_b.molecule_id == component_a.molecule_id:
                        continue
                    comp_b_matches = matches_b_by_component.get(component_b.search_component_id)
                    if not comp_b_matches:
                        continue
                    for match_a in comp_a_matches:
                        for match_b in comp_b_matches:
                            contact_key = self._canonical_contact_key(
                                fragment_a_spec.smarts,
                                fragment_b_spec.smarts,
                                component_a,
                                match_a,
                                component_b,
                                match_b,
                            )
                            if contact_key in seen_contact_keys:
                                continue
                            geometry = self._contact_geometry(
                                atoms,
                                component_a,
                                match_a,
                                component_b,
                                match_b,
                            )
                            if geometry["distance"] > float(radius_max):
                                continue
                            seen_contact_keys.add(contact_key)
                            structure_hits += 1
                            results.append(
                                {
                                    "structure_id": int(structure_id),
                                    "refcode": metadata.get("refcode"),
                                    "molecule_a": int(component_a.molecule_id),
                                    "molecule_b": int(component_b.molecule_id),
                                    "fragment_a_smarts": fragment_a_spec.smarts,
                                    "fragment_b_smarts": fragment_b_spec.smarts,
                                    "fragment_a_atoms": [value + 1 for value in match_a.global_atoms],
                                    "fragment_b_atoms": [value + 1 for value in match_b.global_atoms],
                                    "anchor_a_atom": int(match_a.anchor_global + 1),
                                    "anchor_b_atom": int(match_b.anchor_global + 1),
                                    "distance": geometry["distance"],
                                    "angle_a": geometry["angle_a"],
                                    "angle_b": geometry["angle_b"],
                                    "torsion_a": geometry["torsion_a"],
                                    "torsion_b": geometry["torsion_b"],
                                }
                            )
            if structure_hits > 0:
                structures_with_hits += 1
            if progress_every and idx % int(progress_every) == 0:
                print(
                    f"Processed {idx}/{len(scan_ids)} structures, "
                    f"hits so far: {len(results)}"
                )

        return {
            "summary": {
                "mode": "smarts_contact",
                "scanned_structures": len(scan_ids),
                "structures_with_hits": structures_with_hits,
                "contacts_found": len(results),
                "radius_max": float(radius_max),
                "fragment_a": fragment_a_spec.smarts,
                "fragment_b": fragment_b_spec.smarts,
                "covalent_scale": float(covalent_scale),
            },
            "results": results,
        }

    # ═══════════════════════════════════════════════════════
    # SEARCH FUNCTIONS
    # ═══════════════════════════════════════════════════════
    def search_by_name_keyword(
        self,
        keyword: str,
        field: str = "name_systematic",
        case_sensitive: bool = False,
    ) -> List[int]:
        field = self._resolve_field_name(field)
        if field not in self.metadata_column_set:
            return []
        target = keyword.strip()
        if not target:
            return []

        def predicate(batch_df: pd.DataFrame) -> pd.Series:
            text = batch_df[field].fillna("").astype(str).str.strip()
            if case_sensitive:
                return (text != "") & text.str.contains(target, regex=False)
            return (text != "") & text.str.lower().str.contains(target.lower(), regex=False)

        return self._collect_with_batch_filters(["structure_id", field], None, [predicate])

    def search_by_elements(
        self,
        elements: Union[str, List[str], Set[str], Dict[str, Any]],
    ) -> List[int]:
        formula_field = self._resolve_field_name("formula_moiety")
        if formula_field not in self.metadata_column_set:
            return []

        required_elements, additional_elements = (
            self._normalize_required_additional_elements(elements)
        )
        if not required_elements:
            return []

        def predicate(batch_df: pd.DataFrame) -> pd.Series:
            values = batch_df[formula_field].fillna("").astype(str).str.strip()
            result = []
            for formula in values.tolist():
                if not formula or formula in NULL_STRINGS:
                    result.append(False)
                    continue
                structure_elements = self._parse_formula_elements(formula)
                result.append(
                    self._elements_filter_matches(
                        structure_elements,
                        required_elements,
                        additional_elements,
                    )
                )
            return pd.Series(result, index=batch_df.index)

        return self._collect_with_batch_filters(
            ["structure_id", formula_field], None, [predicate]
        )

    def search_by_year(
        self, year_min: Optional[int] = None, year_max: Optional[int] = None
    ) -> List[int]:
        if "year" not in self.metadata_column_set:
            return []
        filters = []
        if year_min is not None:
            filters.append(ds.field("year") >= year_min)
        if year_max is not None:
            filters.append(ds.field("year") <= year_max)
        filter_expr = None
        for expr in filters:
            filter_expr = expr if filter_expr is None else filter_expr & expr
        table = self.metadata_dataset.to_table(columns=["structure_id"], filter=filter_expr)
        return [int(value) for value in table.column("structure_id").to_pylist()]

    def search_by_n_atoms(
        self, min_atoms: Optional[int] = None, max_atoms: Optional[int] = None
    ) -> List[int]:
        field = "n_atoms_full" if "n_atoms_full" in self.metadata_column_set else "n_atoms"
        if field not in self.metadata_column_set:
            return []

        filters = []
        if min_atoms is not None:
            filters.append(ds.field(field) >= min_atoms)
        if max_atoms is not None:
            filters.append(ds.field(field) <= max_atoms)
        filter_expr = None
        for expr in filters:
            filter_expr = expr if filter_expr is None else filter_expr & expr
        table = self.metadata_dataset.to_table(columns=["structure_id"], filter=filter_expr)
        return [int(value) for value in table.column("structure_id").to_pylist()]

    def search_by_refcode(self, refcode: str, case_sensitive: bool = False) -> List[int]:
        if "refcode" not in self.metadata_column_set:
            return []
        target = refcode.strip()
        if not target:
            return []

        def predicate(batch_df: pd.DataFrame) -> pd.Series:
            text = batch_df["refcode"].fillna("").astype(str).str.strip()
            if case_sensitive:
                return (text != "") & text.str.contains(target, regex=False)
            return (text != "") & text.str.lower().str.contains(target.lower(), regex=False)

        return self._collect_with_batch_filters(["structure_id", "refcode"], None, [predicate])

    def search_by_spacegroup(
        self,
        name: Optional[str] = None,
        number: Optional[int] = None,
        system: Optional[Union[str, int]] = None,
    ) -> List[int]:
        filter_expr = None
        predicates: List[Callable[[pd.DataFrame], pd.Series]] = []
        columns = ["structure_id"]

        if name is not None:
            if "spacegroup" not in self.metadata_column_set:
                return []
            columns.append("spacegroup")
            target = str(name).strip().replace(" ", "").lower()

            def name_predicate(batch_df: pd.DataFrame) -> pd.Series:
                text = (
                    batch_df["spacegroup"]
                    .fillna("")
                    .astype(str)
                    .str.strip()
                    .str.replace(" ", "", regex=False)
                    .str.lower()
                )
                return text == target

            predicates.append(name_predicate)

        if number is not None:
            if "spacegroup_number" not in self.metadata_column_set:
                return []
            expr = ds.field("spacegroup_number") == number
            filter_expr = expr if filter_expr is None else filter_expr & expr

        if system is not None:
            if "spacegroup_system" not in self.metadata_column_set:
                return []
            if isinstance(system, (int, np.integer)) or str(system).strip().isdigit():
                expr = ds.field("spacegroup_system") == int(system)
                filter_expr = expr if filter_expr is None else filter_expr & expr
            else:
                columns.append("spacegroup_system")
                target = str(system).strip().lower()

                def system_predicate(batch_df: pd.DataFrame) -> pd.Series:
                    text = batch_df["spacegroup_system"].fillna("").astype(str).str.strip().str.lower()
                    return text == target

                predicates.append(system_predicate)

        if not predicates:
            table = self.metadata_dataset.to_table(columns=["structure_id"], filter=filter_expr)
            return [int(value) for value in table.column("structure_id").to_pylist()]
        return self._collect_with_batch_filters(columns, filter_expr, predicates)

    def search_by_r_factor(
        self, max_r: float, min_r: float = 0.0, exclude_none: bool = True
    ) -> List[int]:
        if "r_factor" not in self.metadata_column_set:
            return []
        filter_expr = (ds.field("r_factor") >= min_r) & (ds.field("r_factor") <= max_r)
        if exclude_none:
            filter_expr = filter_expr & (ds.field("r_factor") >= 0)
        table = self.metadata_dataset.to_table(columns=["structure_id"], filter=filter_expr)
        return [int(value) for value in table.column("structure_id").to_pylist()]

    # ═══════════════════════════════════════════════════════
    # UNIVERSAL SEARCH
    # ═══════════════════════════════════════════════════════
    def search(self, **filters) -> List[int]:
        if not filters:
            return []

        smarts_fragment_condition = filters.pop("smarts_fragment", None)
        dataset_filter = None
        batch_predicates: List[Callable[[pd.DataFrame], pd.Series]] = []
        columns = ["structure_id"]

        for field, condition in list(filters.items()):
            if condition is None:
                continue

            if field == "elements":
                formula_field = self._resolve_field_name("formula_moiety")
                if formula_field not in self.metadata_column_set:
                    return []
                columns.append(formula_field)
                required_elements, additional_elements = (
                    self._normalize_required_additional_elements(condition)
                )
                if not required_elements:
                    return []

                def elements_predicate(
                    batch_df: pd.DataFrame,
                    formula_field: str = formula_field,
                    required_elements: Set[str] = required_elements,
                    additional_elements: Set[str] = additional_elements,
                ) -> pd.Series:
                    values = batch_df[formula_field].fillna("").astype(str).str.strip()
                    result = []
                    for formula in values.tolist():
                        if not formula or formula in NULL_STRINGS:
                            result.append(False)
                            continue
                        structure_elements = self._parse_formula_elements(formula)
                        result.append(
                            self._elements_filter_matches(
                                structure_elements,
                                required_elements,
                                additional_elements,
                            )
                        )
                    return pd.Series(result, index=batch_df.index)

                batch_predicates.append(elements_predicate)
                continue

            if field == "spacegroup_name":
                if "spacegroup" not in self.metadata_column_set:
                    return []
                columns.append("spacegroup")
                target = str(condition).strip().replace(" ", "").lower()

                def spacegroup_name_predicate(batch_df: pd.DataFrame, target: str = target) -> pd.Series:
                    text = (
                        batch_df["spacegroup"]
                        .fillna("")
                        .astype(str)
                        .str.strip()
                        .str.replace(" ", "", regex=False)
                        .str.lower()
                    )
                    return text == target

                batch_predicates.append(spacegroup_name_predicate)
                continue

            if field == "spacegroup_number":
                expr = self._dataset_filter_for_condition("spacegroup_number", condition)
                if expr is not None:
                    dataset_filter = expr if dataset_filter is None else dataset_filter & expr
                else:
                    columns.append("spacegroup_number")
                    batch_predicates.append(
                        lambda batch_df, condition=condition: self._apply_condition_to_series(
                            batch_df["spacegroup_number"], condition
                        )
                    )
                continue

            if field == "name_keyword":
                if "name_systematic" not in self.metadata_column_set:
                    return []
                columns.append("name_systematic")
                target = str(condition).strip().lower()

                def name_keyword_predicate(batch_df: pd.DataFrame, target: str = target) -> pd.Series:
                    text = batch_df["name_systematic"].fillna("").astype(str).str.strip().str.lower()
                    return (text != "") & text.str.contains(target, regex=False)

                batch_predicates.append(name_keyword_predicate)
                continue

            if field == "n_atoms":
                n_atoms_field = "n_atoms_full" if "n_atoms_full" in self.metadata_column_set else "n_atoms"
                if n_atoms_field not in self.metadata_column_set:
                    return []
                expr = self._dataset_filter_for_condition(n_atoms_field, condition)
                if expr is not None:
                    dataset_filter = expr if dataset_filter is None else dataset_filter & expr
                else:
                    columns.append(n_atoms_field)
                    batch_predicates.append(
                        lambda batch_df, condition=condition, field_name=n_atoms_field: (
                            self._apply_condition_to_series(batch_df[field_name], condition)
                        )
                    )
                continue

            if field == "refcode":
                if "refcode" not in self.metadata_column_set:
                    return []
                columns.append("refcode")
                target = str(condition).strip().lower()

                def refcode_predicate(batch_df: pd.DataFrame, target: str = target) -> pd.Series:
                    text = batch_df["refcode"].fillna("").astype(str).str.strip().str.lower()
                    return (text != "") & text.str.contains(target, regex=False)

                batch_predicates.append(refcode_predicate)
                continue

            if field in {"year", "temperature", "r_factor"}:
                resolved = self._resolve_field_name(field)
                if resolved not in self.metadata_column_set:
                    return []
                expr = self._dataset_filter_for_condition(resolved, condition)
                if expr is not None:
                    dataset_filter = expr if dataset_filter is None else dataset_filter & expr
                else:
                    columns.append(resolved)
                    batch_predicates.append(
                        lambda batch_df, condition=condition, field_name=resolved: (
                            self._apply_condition_to_series(batch_df[field_name], condition)
                        )
                    )
                continue

            resolved = self._resolve_field_name(field)
            if resolved not in self.metadata_column_set:
                print(f"⚠️ Field {field} not found")
                continue
            expr = self._dataset_filter_for_condition(resolved, condition)
            if expr is not None:
                dataset_filter = expr if dataset_filter is None else dataset_filter & expr
            else:
                columns.append(resolved)
                batch_predicates.append(
                    lambda batch_df, condition=condition, field_name=resolved: (
                        self._apply_condition_to_series(batch_df[field_name], condition)
                    )
                )

        if not batch_predicates:
            table = self.metadata_dataset.to_table(columns=["structure_id"], filter=dataset_filter)
            structure_ids = [int(value) for value in table.column("structure_id").to_pylist()]
        else:
            structure_ids = self._collect_with_batch_filters(columns, dataset_filter, batch_predicates)

        if smarts_fragment_condition is None:
            return structure_ids

        fragment_filter = self._normalize_smarts_fragment_filter(smarts_fragment_condition)
        return self.search_smarts_fragment(
            smarts=fragment_filter["smarts"],
            structure_ids=structure_ids,
            covalent_scale=float(fragment_filter.get("covalent_scale", 1.15)),
            progress_every=int(fragment_filter.get("progress_every", 0)),
        )

    # ═══════════════════════════════════════════════════════
    # STRUCTURE LOADING
    # ═══════════════════════════════════════════════════════
    def get_structure(self, identifier: Union[int, str]) -> Tuple[Optional[Atoms], Optional[Dict]]:
        structure_id = self._match_structure_id(identifier)
        if structure_id is None:
            return None, None

        atoms = self._load_atoms_by_structure_id(structure_id)
        if atoms is None:
            return None, None

        metadata_record = self._load_metadata_record(structure_id)
        if metadata_record is None:
            return None, None

        lookup_record = self._load_lookup_record(structure_id)
        metadata = self._metadata_record_to_dict(metadata_record)
        metadata["structure_id"] = structure_id
        metadata["n_atoms"] = len(atoms)
        metadata["volume"] = atoms.get_volume()
        if lookup_record is not None:
            metadata["shard_relpath"] = lookup_record.get("shard_relpath")
        return atoms, metadata

    def get_by_refcode(self, refcode: str) -> Tuple[Optional[Atoms], Optional[Dict]]:
        record = self._load_refcode_record(refcode.strip())
        if record is None:
            return None, None
        return self.get_structure(int(record["structure_id"]))

    def get_by_index(self, idx: int) -> Tuple[Optional[Atoms], Optional[Dict]]:
        return self.get_structure(int(idx))

    # ═══════════════════════════════════════════════════════
    # EXPORT: CIF
    # ═══════════════════════════════════════════════════════
    def export_to_cif(
        self,
        atoms: Atoms,
        metadata: Dict,
        output_path: str,
        include_symmetry: bool = True,
    ) -> bool:
        if not PYMATGEN_AVAILABLE:
            print("⚠️ pymatgen not found, using ASE")
            return self._export_cif_with_metadata(atoms, metadata, output_path, include_symmetry)

        try:
            from pymatgen.symmetry.analyzer import SymmetryUndeterminedError

            try:
                structure = Structure(
                    lattice=atoms.get_cell(),
                    species=atoms.get_atomic_numbers(),
                    coords=atoms.get_scaled_positions(),
                    coords_are_cartesian=False,
                    validate_proximity=False,
                )
            except Exception as exc:
                print(f"⚠️ Structure creation error: {exc}, fallback to ASE")
                return self._export_cif_with_metadata(atoms, metadata, output_path, include_symmetry)

            def write_pymatgen_cif(symprec: Optional[float]):
                cif_writer = CifWriter(structure, symprec=symprec, significant_figures=6)
                return str(cif_writer).split("\n")

            try:
                symprec = 0.15 if include_symmetry else None
                lines = write_pymatgen_cif(symprec)
            except SymmetryUndeterminedError:
                print(
                    f"⚠️ SymmetryUndeterminedError for {os.path.basename(output_path)}, retry without pymatgen symmetry"
                )
                try:
                    lines = write_pymatgen_cif(None)
                except Exception as retry_exc:
                    print(f"⚠️ CifWriter retry error: {retry_exc}, fallback to ASE")
                    return self._export_cif_with_metadata(atoms, metadata, output_path, include_symmetry)
            except AttributeError as exc:
                if "SpglibCppError" not in str(exc):
                    print(f"⚠️ CifWriter error: {exc}, fallback to ASE")
                    return self._export_cif_with_metadata(atoms, metadata, output_path, include_symmetry)
                print(
                    f"⚠️ spglib compatibility error for {os.path.basename(output_path)}, retry without pymatgen symmetry"
                )
                try:
                    lines = write_pymatgen_cif(None)
                except Exception as retry_exc:
                    print(f"⚠️ CifWriter retry error: {retry_exc}, fallback to ASE")
                    return self._export_cif_with_metadata(atoms, metadata, output_path, include_symmetry)
            except Exception as exc:
                print(f"⚠️ CifWriter error: {exc}, fallback to ASE")
                return self._export_cif_with_metadata(atoms, metadata, output_path, include_symmetry)

            insert_idx = 0
            for i, line in enumerate(lines):
                if line.strip().startswith("data_"):
                    insert_idx = i + 1
                    break

            extra_tags = []
            if include_symmetry:
                refcode = str(metadata.get("refcode", "")).strip()
                if refcode and refcode not in NULL_STRINGS:
                    extra_tags.append(f"_database_code_dccds             '{refcode}'")

                formula = str(metadata.get("formula_sum", "")).strip()
                if formula and formula not in NULL_STRINGS:
                    extra_tags.append(f"_chemical_formula_sum            '{formula}'")

                z_value = metadata.get("z_value")
                if z_value not in (None, -1, "None", ""):
                    try:
                        extra_tags.append(f"_cell_formula_units_Z            {int(z_value)}")
                    except Exception:
                        pass

                temperature = metadata.get("temperature")
                if temperature not in (None, -1):
                    try:
                        extra_tags.append(
                            f"_diffrn_ambient_temperature      {float(temperature):.2f}"
                        )
                    except Exception:
                        pass

                r_factor = metadata.get("r_factor")
                if r_factor not in (None, -1):
                    try:
                        extra_tags.append(f"_refine_ls_R_factor_all          {float(r_factor):.5f}")
                    except Exception:
                        pass

                doi = str(metadata.get("doi", "")).strip()
                if doi and doi not in NULL_STRINGS:
                    extra_tags.append(f"_publ_section_doi                '{doi}'")

            if extra_tags:
                lines = lines[:insert_idx] + [""] + extra_tags + [""] + lines[insert_idx:]

            with open(output_path, "w", encoding="utf-8") as fh:
                fh.write("\n".join(lines))
            return True
        except ImportError as exc:
            print(f"⚠️ ImportError: {exc}, using ASE")
            return self._export_cif_with_metadata(atoms, metadata, output_path, include_symmetry)
        except Exception as exc:
            print(f"⚠️ CIF export error {output_path}: {exc}")
            return self._export_cif_with_metadata(atoms, metadata, output_path, include_symmetry)

    def _export_cif_with_metadata(
        self,
        atoms: Atoms,
        metadata: Dict,
        output_path: str,
        include_symmetry: bool = True,
    ) -> bool:
        if not ASE_AVAILABLE:
            print("⚠️ ASE not available")
            return False

        try:
            write(output_path, atoms, format="cif")
            with open(output_path, "r", encoding="utf-8") as fh:
                lines = fh.readlines()

            insert_idx = 0
            for i, line in enumerate(lines):
                if line.strip().startswith("data_"):
                    insert_idx = i + 1
                    break

            extra_tags = []
            if include_symmetry:
                sg = str(metadata.get("spacegroup", "")).strip()
                if sg and sg not in NULL_STRINGS:
                    extra_tags.append(f"_symmetry_space_group_name_H-M   '{sg}'")

                sg_num = metadata.get("spacegroup_number")
                if sg_num not in (None, -1, "None", ""):
                    extra_tags.append(f"_symmetry_Int_Tables_number      {int(sg_num)}")

                hall = str(metadata.get("spacegroup_hall", "")).strip()
                if hall and hall not in NULL_STRINGS:
                    extra_tags.append(f"_symmetry_space_group_name_Hall  '{hall}'")

                refcode = str(metadata.get("refcode", "")).strip()
                if refcode and refcode not in NULL_STRINGS:
                    extra_tags.append(f"_database_code_dccds             '{refcode}'")

                formula = str(metadata.get("formula_sum", "")).strip()
                if formula and formula not in NULL_STRINGS:
                    extra_tags.append(f"_chemical_formula_sum            '{formula}'")

                z_value = metadata.get("z_value")
                if z_value not in (None, -1, "None", ""):
                    try:
                        extra_tags.append(f"_cell_formula_units_Z            {int(z_value)}")
                    except Exception:
                        pass

                temperature = metadata.get("temperature")
                if temperature not in (None, -1):
                    try:
                        extra_tags.append(
                            f"_diffrn_ambient_temperature      {float(temperature):.2f}"
                        )
                    except Exception:
                        pass

                r_factor = metadata.get("r_factor")
                if r_factor not in (None, -1):
                    try:
                        extra_tags.append(f"_refine_ls_R_factor_all          {float(r_factor):.5f}")
                    except Exception:
                        pass

                doi = str(metadata.get("doi", "")).strip()
                if doi and doi not in NULL_STRINGS:
                    extra_tags.append(f"_publ_section_doi                '{doi}'")

            if extra_tags:
                lines = lines[:insert_idx] + ["\n"] + extra_tags + ["\n"] + lines[insert_idx:]

            with open(output_path, "w", encoding="utf-8") as fh:
                fh.writelines(lines)
            return True
        except Exception as exc:
            print(f"⚠️ _export_cif_with_metadata error {output_path}: {exc}")
            return False

    # ═══════════════════════════════════════════════════════
    # EXPORT: POSCAR / JSON
    # ═══════════════════════════════════════════════════════
    def export_to_poscar(
        self,
        atoms: Atoms,
        metadata: Dict,
        output_path: str,
        direct: bool = True,
        sort: bool = False,
    ) -> bool:
        if not ASE_AVAILABLE:
            print("⚠️ ASE not available")
            return False

        try:
            write(output_path, atoms, format="vasp", direct=direct, sort=sort)
            return True
        except Exception as exc:
            print(f"⚠️ POSCAR export error {output_path}: {exc}")
            return False

    def export_to_json(
        self,
        atoms: Atoms,
        metadata: Dict,
        output_path: str,
        include_structure: bool = True,
        indent: int = 2,
    ) -> bool:
        try:
            output = {"metadata": {}, "structure": {}}
            for key, value in metadata.items():
                if key in USER_METADATA_EXCLUDED_FIELDS:
                    continue
                if isinstance(value, (np.integer, np.floating)):
                    output["metadata"][key] = value.item()
                elif isinstance(value, np.ndarray):
                    output["metadata"][key] = value.tolist()
                else:
                    output["metadata"][key] = value

            if include_structure:
                output["structure"] = {
                    "n_atoms": len(atoms),
                    "chemical_symbols": atoms.get_chemical_symbols(),
                    "atomic_numbers": atoms.get_atomic_numbers().tolist(),
                    "positions": atoms.get_positions().tolist(),
                    "scaled_positions": atoms.get_scaled_positions().tolist(),
                    "cell": atoms.get_cell().array.tolist(),
                    "pbc": atoms.get_pbc().tolist(),
                    "volume": atoms.get_volume(),
                    "mass": atoms.get_masses().sum(),
                }

            with open(output_path, "w", encoding="utf-8") as fh:
                json.dump(output, fh, indent=indent, ensure_ascii=False, default=str)
            return True
        except Exception as exc:
            print(f"⚠️ JSON export error {output_path}: {exc}")
            return False

    # ═══════════════════════════════════════════════════════
    # MASS EXPORT
    # ═══════════════════════════════════════════════════════
    def export_structures(
        self,
        indices: List[int],
        output_dir: str,
        format: str = "cif",
        limit: Optional[int] = None,
        **export_kwargs,
    ) -> Dict[str, Any]:
        os.makedirs(output_dir, exist_ok=True)
        stats = {"success": 0, "failed": 0, "errors": []}
        count = 0

        for idx in indices:
            if limit and count >= limit:
                break

            atoms, meta = self.get_by_index(idx)
            if atoms is None:
                stats["failed"] += 1
                stats["errors"].append(f"ID {idx}: failed to load structure")
                continue

            refcode = meta.get("refcode", f"id_{idx}")
            if not refcode or refcode in NULL_STRINGS:
                refcode = f"id_{idx}"
            safe_name = "".join(c if c.isalnum() else "_" for c in str(refcode))[:50]
            if not safe_name:
                safe_name = f"id_{idx}"

            if format == "cif":
                output_path = f"{output_dir}/{safe_name}.cif"
                success = self.export_to_cif(atoms, meta, output_path, **export_kwargs)
            elif format == "poscar":
                output_path = f"{output_dir}/{safe_name}_POSCAR"
                success = self.export_to_poscar(atoms, meta, output_path, **export_kwargs)
            elif format == "json":
                output_path = f"{output_dir}/{safe_name}.json"
                success = self.export_to_json(atoms, meta, output_path, **export_kwargs)
            else:
                print(f"⚠️ Unknown format: {format}. Available: cif, poscar, json")
                success = False

            if success:
                stats["success"] += 1
                count += 1
            else:
                stats["failed"] += 1
                stats["errors"].append(f"ID {idx}: export error")

        return stats

    def export_all_filtered(
        self,
        output_dir: str,
        format: str = "cif",
        limit: int = 100,
        **filters,
    ) -> Dict[str, Any]:
        indices = self.search(**filters)
        return self.export_structures(indices, output_dir, format=format, limit=limit)

    # ═══════════════════════════════════════════════════════
    # ITERATION
    # ═══════════════════════════════════════════════════════
    def iterate_all(self, batch_size: int = 100):
        columns = list(dict.fromkeys(self.metadata_fields + ["structure_id"]))
        total = self.count()
        processed = 0

        for batch_df in self._scan_metadata_batches(columns, None):
            for record in batch_df.to_dict("records"):
                structure_id = int(record["structure_id"])
                atoms = self._load_atoms_by_structure_id(structure_id)
                if atoms is None:
                    processed += 1
                    continue
                metadata = self._metadata_record_to_dict(record)
                metadata["structure_id"] = structure_id
                metadata["n_atoms"] = len(atoms)
                metadata["volume"] = atoms.get_volume()
                yield atoms, metadata
                processed += 1
                if batch_size and processed % batch_size == 0:
                    print(f"🔄 Processed {processed}/{total} structures")

    def iterate_filtered(self, batch_size: int = 100, **filters):
        indices = self.search(**filters)
        total = len(indices)
        for i, idx in enumerate(indices, start=1):
            atoms, metadata = self.get_by_index(idx)
            if atoms is not None:
                yield atoms, metadata
            if batch_size and i % batch_size == 0:
                print(f"🔄 Processed {i}/{total} filtered structures")

    # ═══════════════════════════════════════════════════════
    # PANDAS DATAFRAME / STATS
    # ═══════════════════════════════════════════════════════
    def get_metadata_dataframe(
        self,
        columns: Optional[List[str]] = None,
        chunk_size: Optional[int] = None,
    ) -> pd.DataFrame:
        del chunk_size
        if columns is None:
            columns = self.metadata_fields + [
                "structure_id",
                "legacy_global_idx",
            ]
        resolved_columns = []
        for column in columns:
            resolved = self._resolve_field_name(column)
            if resolved in self.metadata_column_set and resolved not in resolved_columns:
                resolved_columns.append(resolved)
        if not resolved_columns:
            return pd.DataFrame()
        return pd.read_parquet(self.metadata_path, columns=resolved_columns)

    def get_statistics(self) -> Dict[str, Any]:
        if self._stats_cache is None:
            counts = self.manifest.get("counts", {})
            total_structures = int(counts.get("structure_count", 0))
            total_atoms = int(counts.get("atom_count", 0))
            if self.refcode_dataset is not None:
                refcodes_indexed = int(self.refcode_dataset.count_rows())
            elif "refcode" in self.metadata_column_set:
                refcodes_indexed = int(
                    self.metadata_dataset.count_rows(filter=ds.field("refcode").is_valid())
                )
            else:
                refcodes_indexed = 0
            file_size_mb = sum(
                path.stat().st_size for path in self.db_root.rglob("*") if path.is_file()
            ) / (1024 ** 2)
            self._stats_cache = {
                "total_structures": total_structures,
                "total_atoms": total_atoms,
                "build_date": self.manifest.get("build_date", "N/A"),
                "file_size_mb": file_size_mb,
                "metadata_fields": len(self.metadata_fields),
                "refcodes_indexed": refcodes_indexed,
                "avg_atoms": total_atoms / total_structures if total_structures > 0 else 0,
            }
        return dict(self._stats_cache)

    def get_field_statistics(self, field: str) -> Dict[str, Any]:
        resolved = self._resolve_field_name(field)
        if resolved not in self.metadata_column_set:
            return {}

        df = pd.read_parquet(self.metadata_path, columns=[resolved])
        series = df[resolved]
        if resolved in self.float_fields or resolved in self.int_fields or pd.api.types.is_numeric_dtype(series):
            data = series.dropna()
            if len(data) == 0:
                return {}
            return {
                "field": resolved,
                "count": int(len(data)),
                "min": float(data.min()),
                "max": float(data.max()),
                "mean": float(data.mean()),
                "median": float(data.median()),
                "std": float(data.std(ddof=0)),
            }

        text = series.fillna("").astype(str).str.strip()
        text = text[text != ""]
        if len(text) == 0:
            return {}
        top = text.value_counts().head(10)
        return {
            "field": resolved,
            "count": int(len(text)),
            "unique_count": int(text.nunique()),
            "top_10": [(int(count), value) for value, count in top.items()],
        }


_GRAPH_CACHE_WORKER_DB: Optional[DirectoryStructureDB] = None
_GRAPH_CACHE_WORKER_SETTINGS: Dict[str, Any] = {}


def _graph_cache_worker_init(db_root: str, settings: Dict[str, Any]) -> None:
    global _GRAPH_CACHE_WORKER_DB, _GRAPH_CACHE_WORKER_SETTINGS
    _GRAPH_CACHE_WORKER_DB = DirectoryStructureDB(db_root)
    _GRAPH_CACHE_WORKER_SETTINGS = dict(settings)


def _graph_cache_worker_process(structure_id: int) -> Dict[str, Any]:
    if _GRAPH_CACHE_WORKER_DB is None:
        raise RuntimeError("Graph-cache worker database is not initialized")
    return _GRAPH_CACHE_WORKER_DB._build_graph_cache_structure_rows(
        structure_id=int(structure_id),
        **_GRAPH_CACHE_WORKER_SETTINGS,
    )


class ConfigManager:
    def __init__(self, config_path: str = "config.json"):
        self.config_path = config_path
        self.config = self._load_config()

    def _load_config(self) -> Dict:
        config_file = Path(self.config_path).expanduser().resolve()
        if config_file.exists() and config_file.is_dir():
            raise IsADirectoryError(
                f"Expected path to config.json, but got directory: {config_file}\n"
                f"Pass a JSON config file or create one with:\n"
                f"  python {Path(__file__).name} --init-config ./config_v2.json --db {config_file}"
            )

        if not config_file.exists():
            print(f"Config-file not found: {self.config_path}")
            self._create_default_config()

        with config_file.open("r", encoding="utf-8") as fh:
            raw_config = json.load(fh)
        return self._strip_dict_keys(raw_config)

    def _strip_dict_keys(self, d: Dict) -> Dict:
        if not isinstance(d, dict):
            return d

        result = {}
        for key, value in d.items():
            clean_key = key.strip() if isinstance(key, str) else key
            if isinstance(value, dict):
                result[clean_key] = self._strip_dict_keys(value)
            elif isinstance(value, list):
                result[clean_key] = [
                    self._strip_dict_keys(item) if isinstance(item, dict) else
                    (item.strip() if isinstance(item, str) else item)
                    for item in value
                ]
            elif isinstance(value, str):
                result[clean_key] = value.strip()
            else:
                result[clean_key] = value
        return result

    def _create_default_config(self):
        default_config = build_default_config("uspex_db_v2")
        with open(self.config_path, "w", encoding="utf-8") as fh:
            json.dump(default_config, fh, indent=4, ensure_ascii=False)
        self.config = default_config

    def get_search_filters(self) -> Dict:
        filters = {}
        search_config = self.config.get("search", {})

        name_keyword = search_config.get("name_keyword")
        if name_keyword and isinstance(name_keyword, str) and name_keyword.strip():
            filters["name_keyword"] = name_keyword.strip()

        elements_cfg = search_config.get("elements")
        if elements_cfg:
            if isinstance(elements_cfg, dict):
                filters["elements"] = elements_cfg
            elif isinstance(elements_cfg, (list, str)):
                filters["elements"] = {
                    "required": elements_cfg,
                    "additional": [],
                }

        year_cfg = search_config.get("year")
        if year_cfg:
            filters["year"] = self._parse_numeric_filter(year_cfg)

        sg_cfg = search_config.get("spacegroup")
        if sg_cfg:
            if isinstance(sg_cfg, dict):
                sg_type = sg_cfg.get("type", "number")
                sg_value = sg_cfg.get("value")
                if sg_type == "number":
                    if isinstance(sg_value, list):
                        filters["spacegroup_number"] = sg_value[0] if len(sg_value) == 1 else sg_value
                    else:
                        filters["spacegroup_number"] = sg_value
                elif sg_type == "name":
                    filters["spacegroup_name"] = sg_value
            elif isinstance(sg_cfg, (int, list)):
                if isinstance(sg_cfg, list):
                    filters["spacegroup_number"] = sg_cfg[0] if len(sg_cfg) == 1 else sg_cfg
                else:
                    filters["spacegroup_number"] = sg_cfg

        temp_cfg = search_config.get("temperature")
        if temp_cfg:
            filters["temperature"] = self._parse_numeric_filter(temp_cfg)

        r_cfg = search_config.get("r_factor")
        if r_cfg is not None:
            filters["r_factor"] = self._parse_numeric_filter(r_cfg)

        n_atoms_cfg = search_config.get("n_atoms")
        if n_atoms_cfg:
            filters["n_atoms"] = self._parse_numeric_filter(n_atoms_cfg)

        refcode_cfg = search_config.get("refcode")
        if refcode_cfg and isinstance(refcode_cfg, str) and refcode_cfg.strip():
            filters["refcode"] = refcode_cfg.strip()

        smarts_fragment_cfg = search_config.get("smarts_fragment")
        if smarts_fragment_cfg:
            filters["smarts_fragment"] = smarts_fragment_cfg

        return filters

    def _parse_numeric_filter(self, value: Any) -> Optional[Tuple]:
        if isinstance(value, (int, float)):
            return ("<=", value)
        if isinstance(value, list) and len(value) == 2:
            return ("range", value[0], value[1])
        if isinstance(value, dict):
            operators = ["<", ">", "<=", ">=", "==", "!="]
            for op in operators:
                if op in value:
                    return (op, value[op])
            if "type" in value and "value" in value:
                type_map = {"max": "<=", "min": ">=", "exact": "==", "range": "range"}
                op = type_map.get(value["type"], "==")
                if op == "range" and isinstance(value["value"], list) and len(value["value"]) == 2:
                    return ("range", value["value"][0], value["value"][1])
                return (op, value["value"])
            if "min" in value or "max" in value:
                min_val = value.get("min")
                max_val = value.get("max")
                if min_val is not None and max_val is not None:
                    return ("range", min_val, max_val)
                if min_val is not None:
                    return (">=", min_val)
                if max_val is not None:
                    return ("<=", max_val)
        return None

    def get_export_settings(self) -> Dict:
        export_config = self.config.get("export", {})
        return {
            "output_dir": export_config.get("output_dir", "export_output"),
            "limit": export_config.get("limit", 100),
            "formats": export_config.get("formats", {}),
        }

    def get_logging_settings(self) -> Dict:
        return self.config.get(
            "logging",
            {"verbose": True, "save_stats": True, "stats_file": "export_stats.json"},
        )


class QueryManager:
    def __init__(self, query_path: str = "query.json"):
        self.query_path = query_path
        self.query = self._load_query()

    def _load_query(self) -> Dict:
        query_file = Path(self.query_path).expanduser().resolve()
        if query_file.exists() and query_file.is_dir():
            raise IsADirectoryError(
                f"Expected path to query.json, but got directory: {query_file}\n"
                f"Pass a JSON query file or create one with:\n"
                f"  python {Path(__file__).name} --type query --init-query ./query.json --db {query_file}"
            )
        if not query_file.exists():
            print(f"Query-file not found: {self.query_path}")
            self._create_default_query()
        with query_file.open("r", encoding="utf-8") as fh:
            raw_query = json.load(fh)
        return self._strip_dict_keys(raw_query)

    def _strip_dict_keys(self, d: Dict) -> Dict:
        if not isinstance(d, dict):
            return d
        result = {}
        for key, value in d.items():
            clean_key = key.strip() if isinstance(key, str) else key
            if isinstance(value, dict):
                result[clean_key] = self._strip_dict_keys(value)
            elif isinstance(value, list):
                result[clean_key] = [
                    self._strip_dict_keys(item) if isinstance(item, dict) else
                    (item.strip() if isinstance(item, str) else item)
                    for item in value
                ]
            elif isinstance(value, str):
                result[clean_key] = value.strip()
            else:
                result[clean_key] = value
        return result

    def _create_default_query(self):
        default_query = build_default_query("uspex_db_v2")
        with open(self.query_path, "w", encoding="utf-8") as fh:
            json.dump(default_query, fh, indent=4, ensure_ascii=False)
        self.query = default_query

    def get_query_settings(self) -> Dict[str, Any]:
        query_cfg = self.query.get("query", {})
        fragment_mol2 = query_cfg.get("fragment_mol2")
        if fragment_mol2:
            fragment_path = Path(str(fragment_mol2)).expanduser()
            if not fragment_path.is_absolute():
                fragment_path = Path(self.query_path).expanduser().resolve().parent / fragment_path
            fragment_mol2 = str(fragment_path)
        return {
            "mode": query_cfg.get("mode", "mol2_contact"),
            "search_backend": query_cfg.get("search_backend", "fast_anchor"),
            "fragment_mol2": fragment_mol2,
            "fragment_a": query_cfg.get("fragment_a"),
            "fragment_b": query_cfg.get("fragment_b"),
            "radius_max": float(query_cfg.get("radius_max", 4.0)),
            "distance_min": _optional_float(query_cfg.get("distance_min")),
            "angle_min": _optional_float(query_cfg.get("angle_min")),
            "contact_elements": query_cfg.get("contact_elements", []),
            "contact_scope": query_cfg.get("contact_scope", "intermolecular"),
            "structure_ids": query_cfg.get("structure_ids", []),
            "refcodes": query_cfg.get("refcodes", []),
            "max_structures": query_cfg.get("max_structures"),
            "covalent_scale": float(query_cfg.get("covalent_scale", 1.15)),
            "strict_bonds": bool(query_cfg.get("strict_bonds", False)),
            "strict_atom_types": bool(query_cfg.get("strict_atom_types", True)),
            "allow_hydrogen_wildcards": bool(query_cfg.get("allow_hydrogen_wildcards", True)),
            "progress_every": int(query_cfg.get("progress_every", 100)),
        }

    def get_graph_cache_settings(self) -> Dict[str, Any]:
        cache_cfg = self.query.get("graph_cache", {})
        return {
            "enabled": bool(cache_cfg.get("enabled", True)),
            "path": cache_cfg.get("path", "indexes/graph_cache"),
            "build_if_missing": bool(cache_cfg.get("build_if_missing", True)),
            "rebuild": bool(cache_cfg.get("rebuild", False)),
            "max_structures": cache_cfg.get("max_structures", 100),
            "max_atoms": cache_cfg.get("max_atoms", 1000),
            "covalent_scale": float(cache_cfg.get("covalent_scale", 1.15)),
            "min_nonbonded_distance": float(cache_cfg.get("min_nonbonded_distance", 0.6)),
            "flush_every": int(cache_cfg.get("flush_every", 250)),
            "workers": int(cache_cfg.get("workers", 1)),
            "worker_chunk_size": int(cache_cfg.get("worker_chunk_size", 10)),
            "skip_extended_networks": bool(cache_cfg.get("skip_extended_networks", True)),
            "component_filter_backend": cache_cfg.get("component_filter_backend", "pymatgen"),
            "bond_order_backend": cache_cfg.get("bond_order_backend", "rdkit"),
        }

    def get_output_settings(self) -> Dict[str, Any]:
        output_cfg = self.query.get("output", {})
        plots_cfg = output_cfg.get("plots", {})
        formats = output_cfg.get("formats", ["json"])
        if isinstance(formats, str):
            formats = [formats]
        return {
            "output_dir": output_cfg.get("output_dir", "query_output"),
            "basename": output_cfg.get("basename", "mol2_contact_results"),
            "formats": formats,
            "export_structures": bool(output_cfg.get("export_structures", True)),
            "clean_output": bool(output_cfg.get("clean_output", False)),
            "html_report": bool(output_cfg.get("html_report", True)),
            "plots": {
                "enabled": plots_cfg.get("enabled", True),
                "bins": int(plots_cfg.get("bins", 50)),
                "dpi": int(plots_cfg.get("dpi", 160)),
            },
        }

    def get_logging_settings(self) -> Dict[str, Any]:
        return self.query.get(
            "logging",
            {"verbose": True, "save_summary": True},
        )


class QueryRunner:
    def __init__(self, query_path: str = "query.json"):
        self.query_manager = QueryManager(query_path)
        self.db: Optional[DirectoryStructureDB] = None
        self.payload: Optional[Dict[str, Any]] = None

    def connect(self):
        db_path = self.query_manager.query.get("database", {}).get("path", "uspex_db_v2")
        if not os.path.exists(db_path):
            raise FileNotFoundError(f"Database not found, check path to db: {db_path}")
        self.db = DirectoryStructureDB(db_path)
        if self.query_manager.get_logging_settings().get("verbose", True):
            print(f"Connection completed: {db_path}")
            print(f"Total structures in db: {self.db.count()}")

    def run_query(self) -> Dict[str, Any]:
        if self.db is None:
            raise RuntimeError("Database is not connected")
        query_settings = self.query_manager.get_query_settings()
        graph_cache_settings = self.query_manager.get_graph_cache_settings()
        mode = query_settings.get("mode", "mol2_contact")
        if mode == "mol2_contact":
            fragment_mol2 = query_settings.get("fragment_mol2")
            if not fragment_mol2:
                raise ValueError("fragment_mol2 must be specified in query.json")
            payload = self.db.search_mol2_contacts(
                fragment_mol2=fragment_mol2,
                graph_cache_path=graph_cache_settings.get("path"),
                build_cache_if_missing=graph_cache_settings.get("build_if_missing", True),
                rebuild_cache=graph_cache_settings.get("rebuild", False),
                cache_max_structures=graph_cache_settings.get("max_structures"),
                cache_max_atoms=graph_cache_settings.get("max_atoms"),
                cache_min_nonbonded_distance=graph_cache_settings.get(
                    "min_nonbonded_distance", 0.6
                ),
                cache_skip_extended_networks=graph_cache_settings.get(
                    "skip_extended_networks", True
                ),
                cache_component_filter_backend=graph_cache_settings.get(
                    "component_filter_backend", "pymatgen"
                ),
                cache_bond_order_backend=graph_cache_settings.get("bond_order_backend", "rdkit"),
                cache_flush_every=graph_cache_settings.get("flush_every", 250),
                cache_workers=graph_cache_settings.get("workers", 1),
                cache_worker_chunk_size=graph_cache_settings.get("worker_chunk_size", 10),
                structure_ids=query_settings.get("structure_ids", []),
                refcodes=query_settings.get("refcodes", []),
                radius_max=float(query_settings.get("radius_max", 4.0)),
                distance_min=query_settings.get("distance_min"),
                angle_min=query_settings.get("angle_min"),
                contact_elements=query_settings.get("contact_elements", []),
                contact_scope=query_settings.get("contact_scope", "intermolecular"),
                covalent_scale=float(query_settings.get("covalent_scale", 1.15)),
                strict_bonds=bool(query_settings.get("strict_bonds", False)),
                strict_atom_types=bool(query_settings.get("strict_atom_types", True)),
                allow_hydrogen_wildcards=bool(
                    query_settings.get("allow_hydrogen_wildcards", True)
                ),
                search_backend=query_settings.get("search_backend", "fast_anchor"),
                max_structures=query_settings.get("max_structures"),
                progress_every=int(query_settings.get("progress_every", 100)),
            )
            self.payload = payload
            return payload

        if mode != "smarts_contact":
            raise ValueError(f"Unsupported query mode: {mode}")
        fragment_a = query_settings.get("fragment_a")
        fragment_b = query_settings.get("fragment_b")
        if not fragment_a or not fragment_b:
            raise ValueError("fragment_a and fragment_b must be specified in query.json")
        payload = self.db.search_smarts_contacts(
            fragment_a=fragment_a,
            fragment_b=fragment_b,
            radius_max=float(query_settings.get("radius_max", 4.0)),
            structure_ids=query_settings.get("structure_ids", []),
            refcodes=query_settings.get("refcodes", []),
            covalent_scale=float(query_settings.get("covalent_scale", 1.15)),
            max_structures=query_settings.get("max_structures"),
            progress_every=int(query_settings.get("progress_every", 100)),
        )
        self.payload = payload
        return payload

    def _results_dataframe(self) -> pd.DataFrame:
        if not self.payload:
            return pd.DataFrame()
        return pd.DataFrame(self.payload.get("results", []))

    def _write_json(self, output_path: Path):
        if self.payload is None:
            return
        with output_path.open("w", encoding="utf-8") as fh:
            json.dump(self.payload, fh, indent=2, ensure_ascii=False, default=str)

    def _write_csv(self, output_path: Path):
        df = self._results_dataframe()
        if df.empty:
            with output_path.open("w", encoding="utf-8", newline="") as fh:
                writer = csv.writer(fh)
                writer.writerow(["no_results"])
            return
        df.to_csv(output_path, index=False)

    def _numeric_summary(self, values: Sequence[float]) -> Dict[str, Optional[float]]:
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

    def _safe_label(self, label: str) -> str:
        safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(label).strip())
        return safe.strip("_") or "unknown"

    def _result_statistics(self) -> Dict[str, Dict[str, Optional[float]]]:
        df = self._results_dataframe()
        if df.empty:
            return {
                "distance": self._numeric_summary([]),
                "donor_contact_distance": self._numeric_summary([]),
                "angle": self._numeric_summary([]),
                "torsion": self._numeric_summary([]),
                "by_contact_label": {},
            }
        angle_values: List[float] = []
        torsion_values: List[float] = []
        for field in ("angle_a", "angle_b"):
            if field in df:
                angle_values.extend(df[field].dropna().astype(float).tolist())
        for field in ("torsion_a", "torsion_b"):
            if field in df:
                torsion_values.extend(df[field].dropna().astype(float).tolist())
        distance_values = df["distance"].dropna().astype(float).tolist() if "distance" in df else []
        donor_contact_distance_values = (
            df["donor_contact_distance"].dropna().astype(float).tolist()
            if "donor_contact_distance" in df
            else []
        )
        by_contact_label: Dict[str, Dict[str, Dict[str, Optional[float]]]] = {}
        if "contact_label" in df:
            for label, group_df in df.groupby("contact_label", dropna=False):
                group_angle_values: List[float] = []
                group_torsion_values: List[float] = []
                for field in ("angle_a", "angle_b"):
                    if field in group_df:
                        group_angle_values.extend(group_df[field].dropna().astype(float).tolist())
                for field in ("torsion_a", "torsion_b"):
                    if field in group_df:
                        group_torsion_values.extend(group_df[field].dropna().astype(float).tolist())
                group_distance_values = (
                    group_df["distance"].dropna().astype(float).tolist()
                    if "distance" in group_df
                    else []
                )
                group_donor_contact_values = (
                    group_df["donor_contact_distance"].dropna().astype(float).tolist()
                    if "donor_contact_distance" in group_df
                    else []
                )
                by_contact_label[str(label)] = {
                    "distance": self._numeric_summary(group_distance_values),
                    "donor_contact_distance": self._numeric_summary(group_donor_contact_values),
                    "angle": self._numeric_summary(group_angle_values),
                    "torsion": self._numeric_summary(group_torsion_values),
                }
        return {
            "distance": self._numeric_summary(distance_values),
            "donor_contact_distance": self._numeric_summary(donor_contact_distance_values),
            "angle": self._numeric_summary(angle_values),
            "torsion": self._numeric_summary(torsion_values),
            "by_contact_label": by_contact_label,
        }

    def _export_hit_structures(self, output_dir: Path) -> Dict[int, Dict[str, str]]:
        if self.db is None or self.payload is None:
            return {}
        results = self.payload.get("results", [])
        structure_ids = sorted({int(result["structure_id"]) for result in results if "structure_id" in result})
        if not structure_ids:
            return {}
        cif_dir = output_dir / "cif"
        poscar_dir = output_dir / "poscar"
        cif_dir.mkdir(parents=True, exist_ok=True)
        poscar_dir.mkdir(parents=True, exist_ok=True)
        exported: Dict[int, Dict[str, str]] = {}
        for structure_id in structure_ids:
            atoms, metadata = self.db.get_structure(int(structure_id))
            if atoms is None or metadata is None:
                continue
            refcode = str(metadata.get("refcode") or f"id_{structure_id}")
            safe_name = "".join(ch if ch.isalnum() else "_" for ch in refcode)[:50]
            if not safe_name:
                safe_name = f"id_{structure_id}"
            cif_path = cif_dir / f"{safe_name}.cif"
            poscar_path = poscar_dir / f"{safe_name}_POSCAR"
            self.db.export_to_cif(atoms, metadata, str(cif_path), include_symmetry=True)
            self.db.export_to_poscar(atoms, metadata, str(poscar_path))
            exported[int(structure_id)] = {
                "cif_path": str(cif_path),
                "poscar_path": str(poscar_path),
            }
        for result in results:
            paths = exported.get(int(result.get("structure_id", -1)))
            if paths:
                result.update(paths)
        return exported

    def _write_html_report(
        self,
        output_path: Path,
        summary: Dict[str, Any],
        plot_files: Sequence[str],
    ):
        df = self._results_dataframe()
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
        by_label_stats = summary.get("statistics", {}).get("by_contact_label", {})
        by_label_rows = []
        for label, stats in by_label_stats.items():
            distance_stats = stats.get("distance", {})
            donor_contact_stats = stats.get("donor_contact_distance", {})
            angle_stats = stats.get("angle", {})
            torsion_stats = stats.get("torsion", {})
            by_label_rows.append(
                "<tr>"
                f"<td>{html.escape(str(label))}</td>"
                f"<td>{html.escape(str(distance_stats.get('count')))}</td>"
                f"<td>{html.escape(str(distance_stats.get('mean')))}</td>"
                f"<td>{html.escape(str(donor_contact_stats.get('mean')))}</td>"
                f"<td>{html.escape(str(angle_stats.get('mean')))}</td>"
                f"<td>{html.escape(str(torsion_stats.get('mean')))}</td>"
                "</tr>"
            )
        by_label_table = ""
        if by_label_rows:
            by_label_table = (
                "<h2>Statistics by contact type</h2>"
                "<table><tr><th>contact</th><th>n</th><th>distance mean</th>"
                "<th>donor-contact mean</th><th>angle mean</th><th>torsion mean</th></tr>"
                + "\n".join(by_label_rows)
                + "</table>"
            )
        summary_items = "\n".join(
            f"<tr><th>{html.escape(str(key))}</th><td>{html.escape(str(value))}</td></tr>"
            for key, value in summary.items()
            if key not in {"written_files", "plot_files", "summary_file"}
        )
        document = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>USPEX graph query report</title>
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
  <h1>USPEX graph query report</h1>
  <h2>Summary</h2>
  <table>{summary_items}</table>
  {by_label_table}
  <h2>Distributions</h2>
  {plots_html or "<p>No plots generated.</p>"}
  <h2>Results</h2>
  {table_html}
</body>
</html>
"""
        with output_path.open("w", encoding="utf-8") as fh:
            fh.write(document)

    def _plot_density(
        self,
        values: Sequence[float],
        title: str,
        xlabel: str,
        output_path: Path,
        bins: int,
        dpi: int,
    ):
        if not MATPLOTLIB_AVAILABLE:
            return
        series = np.asarray(values, dtype=float)
        series = series[~np.isnan(series)]
        if series.size == 0:
            fig, ax = plt.subplots(figsize=(7, 4.5))
            ax.text(0.5, 0.5, "No data", ha="center", va="center", fontsize=14)
            ax.set_title(title)
            ax.set_xlabel(xlabel)
            ax.set_yticks([])
            ax.set_xticks([])
            fig.tight_layout()
            fig.savefig(output_path, dpi=dpi)
            plt.close(fig)
            return
        hist_bins = max(5, int(bins))
        hist_range = None
        if series.size == 1 or np.allclose(series, series[0]):
            center = float(series[0])
            spread = max(abs(center) * 0.05, 0.5)
            hist_range = (center - spread, center + spread)
            hist_bins = min(hist_bins, 10)
        else:
            hist_bins = min(hist_bins, max(10, min(100, series.size)))

        fig, ax = plt.subplots(figsize=(7, 4.5))
        hist, bin_edges, _ = ax.hist(
            series,
            bins=hist_bins,
            range=hist_range,
            density=True,
            alpha=0.45,
            color="#4c78a8",
            edgecolor="white",
            linewidth=0.8,
        )
        centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
        ax.plot(centers, hist, color="#1f3b63", linewidth=1.8)
        ax.set_title(title)
        ax.set_xlabel(xlabel)
        ax.set_ylabel("Probability density")
        ax.grid(alpha=0.25, linewidth=0.5)
        fig.tight_layout()
        fig.savefig(output_path, dpi=dpi)
        plt.close(fig)

    def _plot_overlay_density(
        self,
        grouped_values: Dict[str, Sequence[float]],
        title: str,
        xlabel: str,
        output_path: Path,
        bins: int,
        dpi: int,
    ):
        if not MATPLOTLIB_AVAILABLE:
            return
        cleaned = {
            label: np.asarray(values, dtype=float)[~np.isnan(np.asarray(values, dtype=float))]
            for label, values in grouped_values.items()
        }
        cleaned = {label: values for label, values in cleaned.items() if values.size > 0}
        fig, ax = plt.subplots(figsize=(7, 4.5))
        if not cleaned:
            ax.text(0.5, 0.5, "No data", ha="center", va="center", fontsize=14)
            ax.set_yticks([])
            ax.set_xticks([])
        else:
            hist_bins = max(5, int(bins))
            for label, series in cleaned.items():
                if series.size == 1 or np.allclose(series, series[0]):
                    center = float(series[0])
                    spread = max(abs(center) * 0.05, 0.5)
                    hist_range = (center - spread, center + spread)
                    bins_for_series = min(hist_bins, 10)
                else:
                    hist_range = None
                    bins_for_series = min(hist_bins, max(10, min(100, series.size)))
                hist, bin_edges = np.histogram(
                    series,
                    bins=bins_for_series,
                    range=hist_range,
                    density=True,
                )
                centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
                ax.plot(centers, hist, linewidth=1.8, label=str(label))
            ax.set_ylabel("Probability density")
            ax.grid(alpha=0.25, linewidth=0.5)
            ax.legend(fontsize=8)
        ax.set_title(title)
        ax.set_xlabel(xlabel)
        fig.tight_layout()
        fig.savefig(output_path, dpi=dpi)
        plt.close(fig)

    def _build_plots(self, output_dir: Path, basename: str, bins: int, dpi: int) -> List[str]:
        df = self._results_dataframe()
        if df.empty or not MATPLOTLIB_AVAILABLE:
            return []
        plots_dir = output_dir / f"{basename}_plots"
        plots_dir.mkdir(parents=True, exist_ok=True)
        generated: List[str] = []

        distance_values = df["distance"].dropna().astype(float).tolist() if "distance" in df else []
        donor_contact_distance_values = (
            df["donor_contact_distance"].dropna().astype(float).tolist()
            if "donor_contact_distance" in df
            else []
        )
        angle_values: List[float] = []
        torsion_values: List[float] = []
        for field in ("angle_a", "angle_b"):
            if field in df:
                angle_values.extend(df[field].dropna().astype(float).tolist())
        for field in ("torsion_a", "torsion_b"):
            if field in df:
                torsion_values.extend(df[field].dropna().astype(float).tolist())

        plot_specs = [
            (
                "distance_density.png",
                distance_values,
                "Anchor-Contact Distance Density",
                "Anchor...contact distance, Angstrom",
            ),
            (
                "donor_contact_distance_density.png",
                donor_contact_distance_values,
                "Donor-Contact Distance Density",
                "Donor...contact distance, Angstrom",
            ),
            ("angle_density.png", angle_values, "Angle Density", "Angle, degrees"),
            ("torsion_density.png", torsion_values, "Torsion Density", "Torsion angle, degrees"),
        ]
        for filename, values, title, xlabel in plot_specs:
            output_path = plots_dir / filename
            try:
                self._plot_density(values, title, xlabel, output_path, bins=bins, dpi=dpi)
            except Exception as exc:
                print(f"⚠️ Plot generation error {output_path.name}: {exc}")
                continue
            if output_path.exists():
                generated.append(str(output_path))

        if "contact_label" in df:
            by_label_dir = plots_dir / "by_contact_label"
            by_label_dir.mkdir(parents=True, exist_ok=True)
            grouped_distance: Dict[str, List[float]] = {}
            grouped_donor_contact_distance: Dict[str, List[float]] = {}
            grouped_angle: Dict[str, List[float]] = {}
            grouped_torsion: Dict[str, List[float]] = {}
            for label, group_df in df.groupby("contact_label", dropna=False):
                label = str(label)
                grouped_distance[label] = (
                    group_df["distance"].dropna().astype(float).tolist()
                    if "distance" in group_df
                    else []
                )
                grouped_donor_contact_distance[label] = (
                    group_df["donor_contact_distance"].dropna().astype(float).tolist()
                    if "donor_contact_distance" in group_df
                    else []
                )
                label_angles: List[float] = []
                label_torsions: List[float] = []
                for field in ("angle_a", "angle_b"):
                    if field in group_df:
                        label_angles.extend(group_df[field].dropna().astype(float).tolist())
                for field in ("torsion_a", "torsion_b"):
                    if field in group_df:
                        label_torsions.extend(group_df[field].dropna().astype(float).tolist())
                grouped_angle[label] = label_angles
                grouped_torsion[label] = label_torsions

                safe_label = self._safe_label(label)
                label_specs = [
                    (
                        by_label_dir / f"distance_{safe_label}.png",
                        grouped_distance[label],
                        f"Anchor-Contact Distance Density: {label}",
                        "Anchor...contact distance, Angstrom",
                    ),
                    (
                        by_label_dir / f"donor_contact_distance_{safe_label}.png",
                        grouped_donor_contact_distance[label],
                        f"Donor-Contact Distance Density: {label}",
                        "Donor...contact distance, Angstrom",
                    ),
                    (
                        by_label_dir / f"angle_{safe_label}.png",
                        grouped_angle[label],
                        f"Angle Density: {label}",
                        "Angle, degrees",
                    ),
                    (
                        by_label_dir / f"torsion_{safe_label}.png",
                        grouped_torsion[label],
                        f"Torsion Density: {label}",
                        "Torsion angle, degrees",
                    ),
                ]
                for output_path, values, title, xlabel in label_specs:
                    try:
                        self._plot_density(values, title, xlabel, output_path, bins=bins, dpi=dpi)
                    except Exception as exc:
                        print(f"⚠️ Plot generation error {output_path.name}: {exc}")
                        continue
                    if output_path.exists():
                        generated.append(str(output_path))

            overlay_specs = [
                (
                    plots_dir / "distance_by_contact_label.png",
                    grouped_distance,
                    "Anchor-Contact Distance Density by Contact Type",
                    "Anchor...contact distance, Angstrom",
                ),
                (
                    plots_dir / "donor_contact_distance_by_contact_label.png",
                    grouped_donor_contact_distance,
                    "Donor-Contact Distance Density by Contact Type",
                    "Donor...contact distance, Angstrom",
                ),
                (
                    plots_dir / "angle_by_contact_label.png",
                    grouped_angle,
                    "Angle Density by Contact Type",
                    "Angle, degrees",
                ),
                (
                    plots_dir / "torsion_by_contact_label.png",
                    grouped_torsion,
                    "Torsion Density by Contact Type",
                    "Torsion angle, degrees",
                ),
            ]
            for output_path, grouped_values, title, xlabel in overlay_specs:
                try:
                    self._plot_overlay_density(grouped_values, title, xlabel, output_path, bins=bins, dpi=dpi)
                except Exception as exc:
                    print(f"⚠️ Plot generation error {output_path.name}: {exc}")
                    continue
                if output_path.exists():
                    generated.append(str(output_path))
        return generated

    def _clean_generated_output(self, output_dir: Path, basename: str) -> None:
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

    def export_results(self) -> Dict[str, Any]:
        if self.payload is None:
            raise RuntimeError("No query results to export")
        output_settings = self.query_manager.get_output_settings()
        output_dir = Path(output_settings["output_dir"]).expanduser().resolve()
        basename = str(output_settings.get("basename", "mol2_contact_results"))
        output_dir.mkdir(parents=True, exist_ok=True)
        if output_settings.get("clean_output", False):
            self._clean_generated_output(output_dir, basename)
        written_files: List[str] = []

        exported_structures: Dict[int, Dict[str, str]] = {}
        if output_settings.get("export_structures", True):
            exported_structures = self._export_hit_structures(output_dir)

        formats = [str(fmt).lower() for fmt in output_settings.get("formats", ["json"])]
        if "json" in formats:
            json_path = output_dir / f"{basename}.json"
            self._write_json(json_path)
            written_files.append(str(json_path))
        if "csv" in formats:
            csv_path = output_dir / f"{basename}.csv"
            self._write_csv(csv_path)
            written_files.append(str(csv_path))

        plots_cfg = output_settings.get("plots", {})
        plot_files: List[str] = []
        if plots_cfg.get("enabled", True):
            plot_files = self._build_plots(
                output_dir=output_dir,
                basename=basename,
                bins=int(plots_cfg.get("bins", 50)),
                dpi=int(plots_cfg.get("dpi", 160)),
            )
            written_files.extend(plot_files)

        summary = dict(self.payload.get("summary", {}))
        summary["statistics"] = self._result_statistics()
        summary["exported_structures"] = len(exported_structures)
        if self.query_manager.get_logging_settings().get("save_summary", True):
            summary_path = output_dir / f"{basename}_summary.json"
            written_files.append(str(summary_path))
            summary["summary_file"] = str(summary_path)
        summary["written_files"] = written_files
        summary["plot_files"] = plot_files
        if output_settings.get("html_report", True):
            html_path = output_dir / f"{basename}_report.html"
            self._write_html_report(html_path, summary, plot_files)
            written_files.append(str(html_path))
        if self.query_manager.get_logging_settings().get("save_summary", True):
            with summary_path.open("w", encoding="utf-8") as fh:
                json.dump(summary, fh, indent=2, ensure_ascii=False, default=str)
        return summary

    def print_summary(self, export_summary: Dict[str, Any]):
        if not self.query_manager.get_logging_settings().get("verbose", True):
            return
        print("\n" + "=" * 60)
        print("Graph query summary")
        print("=" * 60)
        for key in (
            "mode",
            "search_backend",
            "scanned_structures",
            "structures_with_hits",
            "contacts_found",
            "radius_max",
            "fragment_mol2",
            "fragment_a",
            "fragment_b",
            "exported_structures",
        ):
            if key in export_summary:
                print(f"{key}: {export_summary[key]}")
        if export_summary.get("written_files"):
            print("\nWritten files:")
            for path in export_summary["written_files"]:
                print(f"   • {path}")
        print("=" * 60)

    def close(self):
        if self.db:
            self.db.close()
        if self.query_manager.get_logging_settings().get("verbose", True):
            print("Connection was closed")

    def run(self):
        try:
            self.connect()
            payload = self.run_query()
            if len(payload.get("results", [])) == 0:
                print("No graph contacts found for the query")
                export_summary = self.export_results()
                self.print_summary(export_summary)
                return
            export_summary = self.export_results()
            self.print_summary(export_summary)
        except FileNotFoundError as exc:
            print(f"Error: {exc}")
        except Exception as exc:
            print(f"Critical error: {exc}")
            import traceback

            traceback.print_exc()
        finally:
            self.close()


class DatabaseExporter:
    def __init__(self, config_path: str = "config.json"):
        self.config_manager = ConfigManager(config_path)
        self.db = None
        self.stats = {"success": 0, "failed": 0, "errors": [], "formats": {}}

    def connect(self):
        db_path = self.config_manager.config.get("database", {}).get("path", "uspex_db_v2")
        if not os.path.exists(db_path):
            raise FileNotFoundError(f"Database not found, check path to db: {db_path}")
        self.db = DirectoryStructureDB(db_path)
        if self.config_manager.config.get("logging", {}).get("verbose", True):
            print(f"Connection completed: {db_path}")
            print(f"Total structures in db: {self.db.count()}")

    def search(self) -> List[int]:
        filters = self.config_manager.get_search_filters()
        if not filters:
            print("No filters are specified or specified incorrectly, check the config-file")
        indices = self.db.search(**filters)
        if self.config_manager.config.get("logging", {}).get("verbose", True):
            print(f"\nNumber of found structures: {len(indices)}")
            if filters:
                print("Applied filters:")
                for key, value in filters.items():
                    print(f"   • {key}: {value}")
        return indices

    def export(self, indices: List[int]) -> Dict:
        export_settings = self.config_manager.get_export_settings()
        output_dir = export_settings["output_dir"]
        limit = export_settings["limit"]
        formats = export_settings["formats"]

        if self.config_manager.config.get("logging", {}).get("verbose", True):
            print(f"\nStarting export to {output_dir}")

        for fmt_name, fmt_config in formats.items():
            subdir = fmt_config.get("output_subdir", f"{fmt_name}_files")
            fmt_output_dir = os.path.join(output_dir, subdir)

            if self.config_manager.config.get("logging", {}).get("verbose", True):
                print(f"\nExport format: {fmt_name.upper()}...")

            format_kwargs = self._get_format_kwargs(fmt_name, fmt_config)
            format_stats = self.db.export_structures(
                indices=indices,
                output_dir=fmt_output_dir,
                format=fmt_name,
                limit=limit,
                **format_kwargs,
            )
            self.stats["formats"][fmt_name] = format_stats
            self.stats["success"] += format_stats.get("success", 0)
            self.stats["failed"] += format_stats.get("failed", 0)
            self.stats["errors"].extend(format_stats.get("errors", []))
        return self.stats

    def _get_format_kwargs(self, fmt_name: str, fmt_config: Dict) -> Dict:
        kwargs = {}
        if fmt_name == "cif":
            kwargs["include_symmetry"] = fmt_config.get("include_symmetry", True)
        elif fmt_name == "poscar":
            kwargs["direct"] = fmt_config.get("direct", True)
            kwargs["sort"] = fmt_config.get("sort", False)
        elif fmt_name == "json":
            kwargs["include_structure"] = fmt_config.get("include_structure", True)
            kwargs["indent"] = fmt_config.get("indent", 2)
        return kwargs

    def save_stats(self):
        logging_cfg = self.config_manager.get_logging_settings()
        if not logging_cfg.get("save_stats", True):
            return
        stats_file = logging_cfg.get("stats_file", "export_stats.json")
        with open(stats_file, "w", encoding="utf-8") as fh:
            json.dump(self.stats, fh, indent=2, ensure_ascii=False, default=str)
        if logging_cfg.get("verbose", True):
            print(f"\nSaved statistics: {stats_file}")

    def print_summary(self):
        logging_cfg = self.config_manager.get_logging_settings()
        if not logging_cfg.get("verbose", True):
            return

        print("\n" + "=" * 60)
        print("Export statistics")
        print("=" * 60)
        print(f"Successfully: {self.stats['success']}")
        print(f"Errors: {self.stats['failed']}")

        if self.stats["formats"]:
            print("\nStatistics by format:")
            for fmt_name, fmt_stats in self.stats["formats"].items():
                print(
                    f"   • {fmt_name.upper()}: {fmt_stats.get('success', 0)} successfully, "
                    f"{fmt_stats.get('failed', 0)} errors"
                )

        if self.stats["errors"] and len(self.stats["errors"]) <= 10:
            print("\nErrors:")
            for error in self.stats["errors"][:10]:
                print(f"   • {error}")

        print("=" * 60)

    def close(self):
        if self.db:
            self.db.close()
        if self.config_manager.config.get("logging", {}).get("verbose", True):
            print("Connection was closed")

    def run(self):
        try:
            self.connect()
            indices = self.search()
            if len(indices) == 0:
                print("No structures found with these filters. Try to change some filters")
                return
            self.export(indices)
            self.save_stats()
            self.print_summary()
        except FileNotFoundError as exc:
            print(f"Error: {exc}")
        except Exception as exc:
            print(f"Critical error: {exc}")
            import traceback

            traceback.print_exc()
        finally:
            self.close()


def main(argv: Optional[Sequence[str]] = None) -> int:
    print("=" * 60)
    print("USPEX db-v2 Export Tool")
    print("=" * 60)

    parser = argparse.ArgumentParser(
        description="Search and export structures from a db-v2 directory."
    )
    parser.add_argument(
        "input_path",
        nargs="?",
        default=None,
        help="Path to config.json or query.json",
    )
    parser.add_argument(
        "--type",
        choices=("auto", "config", "query"),
        default="auto",
        help="Input file type. Default: auto",
    )
    parser.add_argument(
        "--init-config",
        dest="init_config",
        help="Write a template config.json to this path and exit",
    )
    parser.add_argument(
        "--init-query",
        dest="init_query",
        help="Write a template query.json to this path and exit",
    )
    parser.add_argument(
        "--db",
        dest="db_path",
        help="Database directory for --init-config / --init-query",
    )
    parser.add_argument(
        "--build-graph-cache",
        action="store_true",
        help="Build graph cache for the database and exit",
    )
    parser.add_argument(
        "--graph-cache-path",
        dest="graph_cache_path",
        default="indexes/graph_cache",
        help="Graph cache directory relative to the database root",
    )
    parser.add_argument(
        "--cache-max-structures",
        dest="cache_max_structures",
        default="100",
        help="Maximum structures to process while building graph cache. Use 'all' for no limit.",
    )
    parser.add_argument(
        "--cache-max-atoms",
        dest="cache_max_atoms",
        type=int,
        default=1000,
        help="Skip structures with more atoms while building graph cache",
    )
    parser.add_argument(
        "--cache-min-nonbonded-distance",
        dest="cache_min_nonbonded_distance",
        type=float,
        default=0.6,
        help="Skip structures with any interatomic distance below this Angstrom threshold",
    )
    parser.add_argument(
        "--rebuild-graph-cache",
        action="store_true",
        help="Overwrite an existing graph cache",
    )
    parser.add_argument(
        "--component-filter-backend",
        default="pymatgen",
        help="Backend for 0D component filtering: pymatgen, geometry, none",
    )
    parser.add_argument(
        "--bond-order-backend",
        default="rdkit",
        help="Backend for organic bond-order perception: rdkit, openbabel, auto, geometry, none",
    )
    parser.add_argument(
        "--cache-progress-every",
        dest="cache_progress_every",
        type=int,
        default=100,
        help="Print graph-cache build progress every N structures",
    )
    parser.add_argument(
        "--cache-flush-every",
        dest="cache_flush_every",
        type=int,
        default=250,
        help="Write graph-cache parquet row groups every N processed structures; use 0 to flush only at the end",
    )
    parser.add_argument(
        "--cache-workers",
        dest="cache_workers",
        type=int,
        default=1,
        help="Number of worker processes for graph-cache construction",
    )
    parser.add_argument(
        "--cache-worker-chunk-size",
        dest="cache_worker_chunk_size",
        type=int,
        default=10,
        help="Number of structure tasks batched per worker dispatch",
    )
    args = parser.parse_args(argv)

    script_path = Path(__file__).resolve()

    if args.build_graph_cache:
        database_path = args.db_path
        if not database_path and args.input_path:
            candidate = Path(args.input_path).expanduser().resolve()
            if candidate.is_dir() and (candidate / "manifest.json").exists():
                database_path = str(candidate)
            elif candidate.exists() and candidate.is_file():
                with candidate.open("r", encoding="utf-8") as fh:
                    payload = json.load(fh)
                database_path = payload.get("database", {}).get("path")
        if not database_path:
            raise ValueError("--build-graph-cache needs --db or a config/query file with database.path")
        with DirectoryStructureDB(database_path) as db:
            stats = db.build_graph_cache(
                graph_cache_path=args.graph_cache_path,
                max_structures=_optional_int_arg(args.cache_max_structures),
                max_atoms=args.cache_max_atoms,
                min_nonbonded_distance=args.cache_min_nonbonded_distance,
                component_filter_backend=args.component_filter_backend,
                bond_order_backend=args.bond_order_backend,
                overwrite=args.rebuild_graph_cache,
                progress_every=args.cache_progress_every,
                flush_every=args.cache_flush_every,
                workers=args.cache_workers,
                worker_chunk_size=args.cache_worker_chunk_size,
            )
        print(json.dumps(stats, indent=2, ensure_ascii=False))
        return 0

    if args.init_config:
        database_path = args.db_path or "uspex_db_v2"
        config_path = create_config_template(args.init_config, database_path)
        print(f"Template config created: {config_path}")
        print(f"Database path in config: {database_path}")
        print(f"Run next: python {script_path} {config_path}")
        return 0

    if args.init_query:
        database_path = args.db_path or "uspex_db_v2"
        query_path = create_query_template(args.init_query, database_path)
        print(f"Template query created: {query_path}")
        print(f"Database path in query: {database_path}")
        print(f"Run next: python {script_path} --type query {query_path}")
        return 0

    default_input = "query.json" if args.type == "query" else "config.json"
    input_path = Path(args.input_path or default_input).expanduser().resolve()
    if input_path.exists() and input_path.is_dir():
        manifest_path = input_path / "manifest.json"
        if manifest_path.exists():
            print(f"Detected database directory: {input_path}")
            print("This CLI expects a path to config.json or query.json as its positional argument.")
            print("")
            print("Use one of these commands:")
            print(f"  python {script_path} --type config {input_path.parent / 'config.json'}")
            print(f"  python {script_path} --type query {input_path.parent / 'query.json'}")
            print(
                f"  python {script_path} --init-config {input_path.parent / 'config_v2.json'} "
                f"--db {input_path}"
            )
            print(
                f"  python {script_path} --type query --init-query {input_path.parent / 'query_v2.json'} "
                f"--db {input_path}"
            )
            return 1
        print(f"Directory passed, but manifest.json was not found: {input_path}")
        return 1

    input_type = args.type
    if input_type == "auto":
        if input_path.name.lower().startswith("query"):
            input_type = "query"
        elif input_path.exists():
            try:
                with input_path.open("r", encoding="utf-8") as fh:
                    payload = json.load(fh)
                if isinstance(payload, dict) and "query" in payload:
                    input_type = "query"
                else:
                    input_type = "config"
            except Exception:
                input_type = "config"
        else:
            input_type = "config"

    if input_type == "query":
        runner = QueryRunner(str(input_path))
        runner.run()
    else:
        exporter = DatabaseExporter(str(input_path))
        exporter.run()
    return 0


if __name__ == "__main__":
    sys.exit(main())
