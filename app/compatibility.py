"""ABO / Rh red-cell compatibility rules."""
from __future__ import annotations

from .config import BLOOD_GROUPS


def _split(group: str) -> tuple[str, str]:
    group = group.strip().upper()
    if group not in BLOOD_GROUPS:
        raise ValueError(f"Unknown blood group: {group!r}")
    return group[:-1], group[-1]


def is_compatible(donor_group: str, recipient_group: str) -> bool:
    """True if red cells from ``donor_group`` can be given to ``recipient_group``."""
    d_abo, d_rh = _split(donor_group)
    r_abo, r_rh = _split(recipient_group)
    abo_ok = d_abo == "O" or d_abo == r_abo or r_abo == "AB"
    rh_ok = d_rh == "-" or r_rh == "+"
    return abo_ok and rh_ok


def compatible_donor_groups(recipient_group: str) -> list[str]:
    return [g for g in BLOOD_GROUPS if is_compatible(g, recipient_group)]


def compatibility_score(donor_group: str, recipient_group: str) -> float:
    """Soft compatibility score in [0, 1].

    * identical group  -> 1.0
    * compatible       -> 0.75
    * O- for a non-O-  -> 0.5 (universal-donor units are scarce; prefer
      matching donors when other things are equal)
    * incompatible     -> 0.0
    """
    if not is_compatible(donor_group, recipient_group):
        return 0.0
    if donor_group == recipient_group:
        return 1.0
    if donor_group == "O-":
        return 0.5
    return 0.75
