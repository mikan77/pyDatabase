from __future__ import annotations

from typing import Dict, Tuple

from ase.data import atomic_numbers


HYBRIDIZATION_CODES: Dict[str, int] = {
    "": 0,
    "unknown": 0,
    "nan": 0,
    "sp": 1,
    "sp2": 2,
    "sp3": 3,
    "aromatic": 4,
    "metal": 5,
}

BOND_ORDER_CODES: Dict[str, int] = {
    "unknown": 0,
    "single": 1,
    "double": 2,
    "triple": 3,
    "aromatic": 4,
    "amide": 5,
    "coordination": 6,
    "not_connected": 7,
    "dummy": 8,
    "any": 15,
}

CODE_TO_BOND_ORDER = {value: key for key, value in BOND_ORDER_CODES.items()}


def normalize_symbol(symbol: object) -> str:
    text = str(symbol or "").strip()
    if not text:
        return ""
    if len(text) == 1:
        return text.upper()
    return text[0].upper() + text[1:].lower()


def atomic_number_from_symbol(symbol: object) -> int:
    normalized = normalize_symbol(symbol)
    if normalized in {"*", "Du", "DU"}:
        return 0
    return int(atomic_numbers.get(normalized, 0))


def hybridization_code(value: object) -> int:
    return int(HYBRIDIZATION_CODES.get(str(value or "").strip().lower(), 0))


def bond_order_code(value: object) -> int:
    return int(BOND_ORDER_CODES.get(str(value or "").strip().lower(), 0))


def generic_edge_code(number_a: int, number_b: int) -> int:
    left, right = sorted((int(number_a), int(number_b)))
    return int(left * 128 + right)


def strict_edge_code(number_a: int, number_b: int, order_code: int) -> int:
    return int((generic_edge_code(number_a, number_b) << 4) | int(order_code))


def parse_edge_key(edge_key: str) -> Tuple[int, int, int]:
    pair_text, _, order_text = str(edge_key).partition(":")
    left_text, _, right_text = pair_text.partition("-")
    left = atomic_number_from_symbol(left_text)
    right = atomic_number_from_symbol(right_text)
    order = bond_order_code(order_text or "unknown")
    return left, right, order
