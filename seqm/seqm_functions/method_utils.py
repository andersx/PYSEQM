"""Method name helpers for SEQM variants."""

from __future__ import annotations

# Canonical internal names for methods that use an spd (up to 9 AO/atom) layout.
D_ORBITAL_METHODS = frozenset({"PM6", "MNDO_d"})

METHOD_ALIASES = {
    "MNDO/d": "MNDO_d",
    "MNDO/D": "MNDO_d",
    "MNDOd": "MNDO_d",
    "MNDOD": "MNDO_d",
}

# Thiel MNDO/d elements that carry valence d orbitals (DORBS=.TRUE. when |UDD|>0).
MNDOD_D_ELEMENTS = frozenset({13, 14, 15, 16, 17, 35, 53})  # Al–Cl, Br, I

# OM2+d infrastructure: S/Cl get dummy valence d (not production chemistry).
OM2_D_ELEMENTS = frozenset({16, 17})  # S, Cl


def normalize_method(method: str) -> str:
    return METHOD_ALIASES.get(method, method)


def has_d_orbitals(method: str) -> bool:
    return normalize_method(method) in D_ORBITAL_METHODS


def is_mndo_d(method: str) -> bool:
    return normalize_method(method) == "MNDO_d"


def is_om2_d_element(z: int) -> bool:
    return int(z) in OM2_D_ELEMENTS


def molecule_uses_d_basis(molecule) -> bool:
    """True when the SCF / hcore layout is the 9-AO (spd) block packing."""
    method = normalize_method(getattr(molecule, "method", ""))
    if has_d_orbitals(method):
        return True
    if method == "OM2":
        nsh = getattr(molecule, "nSuperHeavy", None)
        if nsh is not None and bool((nsh > 0).any().item()):
            return True
    return False
