"""Small closed-shell set where second-row d orbitals matter.

Geometry targets are experimental gas-phase values (CCCBDB / textbooks).
The reaction is all-singlet: H2S + Cl2 -> SCl2 + H2.
DFT reference reaction energy is filled by ``build_dft_references.py``.
"""

from __future__ import annotations

import json
from pathlib import Path

# Atomic numbers
H, O, F, S, Cl = 1, 8, 9, 16, 17

# Initial Cartesian guesses (Å). Optimized by each method during the benchmark.
MOLECULES = {
    "H2S": {
        "atoms": [S, H, H],
        "coords": [
            [0.000000, 0.000000, 0.103000],
            [0.000000, 0.961600, -0.823000],
            [0.000000, -0.961600, -0.823000],
        ],
        "exp": {"bonds": [(0, 1, 1.3356), (0, 2, 1.3356)], "angles": [(1, 0, 2, 92.11)]},
    },
    "SO2": {
        "atoms": [S, O, O],
        "coords": [
            [0.000000, 0.000000, 0.000000],
            [0.000000, 1.246300, 0.722700],
            [0.000000, -1.246300, 0.722700],
        ],
        "exp": {"bonds": [(0, 1, 1.4308), (0, 2, 1.4308)], "angles": [(1, 0, 2, 119.33)]},
    },
    "Cl2": {
        "atoms": [Cl, Cl],
        "coords": [[0.0, 0.0, 0.0], [0.0, 0.0, 1.988]],
        "exp": {"bonds": [(0, 1, 1.9879)], "angles": []},
    },
    "SCl2": {
        # PYSEQM requires non-increasing Z along each molecule row.
        "atoms": [Cl, Cl, S],
        "coords": [
            [0.000000, 1.570000, 1.250000],
            [0.000000, -1.570000, 1.250000],
            [0.000000, 0.000000, 0.000000],
        ],
        "exp": {"bonds": [(2, 0, 2.011), (2, 1, 2.011)], "angles": [(0, 2, 1, 102.7)]},
    },
    "SF2": {
        "atoms": [S, F, F],
        "coords": [
            [0.000000, 0.000000, 0.000000],
            [0.000000, 1.200000, 1.050000],
            [0.000000, -1.200000, 1.050000],
        ],
        "exp": {"bonds": [(0, 1, 1.589), (0, 2, 1.589)], "angles": [(1, 0, 2, 98.0)]},
    },
    "H2": {
        "atoms": [H, H],
        "coords": [[0.0, 0.0, 0.0], [0.0, 0.0, 0.7414]],
        "exp": {"bonds": [(0, 1, 0.7414)], "angles": []},
    },
}

# Closed-shell reaction used for thermochemistry / fitting
REACTION = {
    "name": "H2S + Cl2 -> SCl2 + H2",
    "reactants": ["H2S", "Cl2"],
    "products": ["SCl2", "H2"],
    # Placeholder; overwritten by DFT (B3LYP/def2-SVP) at DFT-optimized geometries.
    "dft_de_ev": None,
}

DATA_DIR = Path(__file__).resolve().parent / "data"


def save_dataset(path: Path | None = None) -> Path:
    path = path or (DATA_DIR / "testset.json")
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"molecules": MOLECULES, "reaction": REACTION}
    path.write_text(json.dumps(payload, indent=2) + "\n")
    return path


def load_dataset(path: Path | None = None) -> dict:
    path = path or (DATA_DIR / "testset.json")
    if not path.exists():
        save_dataset(path)
    return json.loads(path.read_text())
