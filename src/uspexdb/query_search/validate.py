from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterable, Tuple


CONTACT_KEY_FIELDS = (
    "structure_id",
    "anchor_atom_zero_based",
    "contact_atom_zero_based",
    "contact_offset_x",
    "contact_offset_y",
    "contact_offset_z",
    "contact_label",
)


def contact_key(result: Dict[str, Any]) -> Tuple[Any, ...]:
    matched = tuple(int(value) for value in result.get("matched_atoms_zero_based", []))
    return tuple(result.get(field) for field in CONTACT_KEY_FIELDS) + (matched,)


def compare_payloads(candidate_payload: Dict[str, Any], reference_payload: Dict[str, Any]) -> Dict[str, Any]:
    candidate_by_key = {contact_key(item): item for item in candidate_payload.get("results", [])}
    reference_by_key = {contact_key(item): item for item in reference_payload.get("results", [])}
    candidate_keys = set(candidate_by_key)
    reference_keys = set(reference_by_key)
    missing = sorted(reference_keys - candidate_keys)
    extra = sorted(candidate_keys - reference_keys)

    numeric_fields = ("distance", "anchor_contact_distance", "donor_contact_distance", "angle_a", "torsion_a")
    max_abs_error = {field: 0.0 for field in numeric_fields}
    compared = 0
    for key in sorted(candidate_keys & reference_keys):
        compared += 1
        candidate = candidate_by_key[key]
        reference = reference_by_key[key]
        for field in numeric_fields:
            left = candidate.get(field)
            right = reference.get(field)
            if left is None or right is None:
                continue
            error = abs(float(left) - float(right))
            if error > max_abs_error[field]:
                max_abs_error[field] = error

    return {
        "candidate_contacts": len(candidate_keys),
        "reference_contacts": len(reference_keys),
        "compared_contacts": compared,
        "missing_contacts": len(missing),
        "extra_contacts": len(extra),
        "missing_examples": [list(value) for value in missing[:10]],
        "extra_examples": [list(value) for value in extra[:10]],
        "max_abs_error": max_abs_error,
        "exact_key_match": not missing and not extra,
    }


def compare_with_reference_file(candidate_payload: Dict[str, Any], reference_json: Path) -> Dict[str, Any]:
    reference_path = Path(reference_json).expanduser().resolve()
    with reference_path.open("r", encoding="utf-8") as fh:
        reference_payload = json.load(fh)
    return compare_payloads(candidate_payload, reference_payload)
