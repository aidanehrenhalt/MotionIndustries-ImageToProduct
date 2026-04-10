"""
PGC1 Label Mapping — Single source of truth for the 8-class product label space.

The trained CNN classifies images into 8 PGC1 (Product Group Code level 1)
categories.  PGC1 is always the first digit of the 4-digit PGC4 code that
appears in Motion Industries product catalogs.

Model class indices are 0-indexed:  class_index = PGC1 - 1
"""

# PGC1 value (1-8) → human-readable description.
PGC1_CLASSES: dict[int, str] = {
    1: "BEARINGS",
    2: "SEALS AND ACCESSORIES",
    3: "POWER TRANSMISSION",
    4: "ELECTRICAL & MAT'L HAND'G",
    5: "HOSE AND FITTINGS",
    6: "FLUID POWER",
    7: "PROCESS PUMPS AND EQUIPMENT",
    8: "INDUSTRIAL SUPPLIES",
}

# PGC1=9 is MISCELLANEOUS — excluded from the 8-class model.
EXCLUDED_PGC1: set[int] = {9}

N_CLASSES: int = len(PGC1_CLASSES)  # 8


def pgc_code_to_label(pgc_code) -> int | None:
    """Convert a PGC4 code (e.g. ``"1110"``) or PGC1 code (e.g. ``"1"``)
    to a 0-indexed class label (0-7).

    Returns ``None`` if the code maps to an excluded class (PGC1=9) or is
    invalid / empty.
    """
    s = str(pgc_code).strip() if pgc_code is not None else ""
    if not s:
        return None
    try:
        first_digit = int(s[0])
    except ValueError:
        return None
    if first_digit in EXCLUDED_PGC1 or first_digit not in PGC1_CLASSES:
        return None
    return first_digit - 1  # 0-indexed


def class_index_to_description(class_index: int) -> str:
    """Convert a 0-indexed model class (0-7) to its PGC1 description.

    Returns ``"UNKNOWN"`` for out-of-range values.
    """
    return PGC1_CLASSES.get(class_index + 1, "UNKNOWN")


def get_expected_classes(product: dict) -> set[int]:
    """Return the set of 0-indexed class indices valid for this product's PGC.

    Returns an empty set if the PGC is absent or unmapped — callers must not
    penalize candidates when the set is empty.
    """
    pgc_raw = product.get("pgc", "")
    if pgc_raw:
        result = pgc_code_to_label(pgc_raw)
        if result is not None:
            return {result}
    return set()
