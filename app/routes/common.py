from __future__ import annotations


def mask_name(name: str) -> str:
    """'Aniket Kumar' -> 'Aniket K.' (staff never need the full name to rank donors)."""
    parts = name.split()
    if not parts:
        return "Donor"
    return parts[0] if len(parts) == 1 else f"{parts[0]} {parts[-1][0].upper()}."
