"""Geometry and reaction metrics."""

from __future__ import annotations

import numpy as np


def bond_length(coords, i, j):
    return float(np.linalg.norm(coords[i] - coords[j]))


def bond_angle_deg(coords, i, j, k):
    """Angle i-j-k in degrees."""
    a = coords[i] - coords[j]
    b = coords[k] - coords[j]
    cos = np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
    cos = float(np.clip(cos, -1.0, 1.0))
    return float(np.degrees(np.arccos(cos)))


def geometry_errors(coords, exp):
    """Return lists of (label, value, ref, abs_err) for bonds (Å) and angles (deg)."""
    rows = []
    for i, j, ref in exp.get("bonds", []):
        val = bond_length(coords, i, j)
        rows.append((f"r({i}-{j})", val, ref, abs(val - ref)))
    for i, j, k, ref in exp.get("angles", []):
        val = bond_angle_deg(coords, i, j, k)
        rows.append((f"a({i}-{j}-{k})", val, ref, abs(val - ref)))
    return rows


def reaction_energy(energies: dict, reaction: dict) -> float:
    """ΔE = Σ E(products) - Σ E(reactants) in eV."""
    e = 0.0
    for name in reaction["products"]:
        e += energies[name]
    for name in reaction["reactants"]:
        e -= energies[name]
    return float(e)
