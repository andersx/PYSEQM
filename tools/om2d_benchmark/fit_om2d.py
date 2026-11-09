"""
Initial OM2+d d-channel parameters for S/Cl.

Geometry/reaction least-squares fitting against optimized OM2+d surfaces is
**not** viable until OM2 two-electron d integrals exist (the 9-AO OM2 path is
not geometry-safe). Until then we adopt Thiel MNDO/d (U_dd, ζ_d, β_d) as the
initial seed — the same values already validated in PYSEQM ``MNDO_d``.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from .dataset import DATA_DIR

Z_S, Z_Cl = 16, 17
CSV = Path(__file__).resolve().parents[2] / "seqm/params/parameters_OM2.csv"


def adopt_thiel_mndod_seed() -> dict:
    """Copy MNDO/d d-params into OM2 CSV (idempotent) and write JSON sidecar."""
    mndod = pd.read_csv(Path(__file__).resolve().parents[2] / "seqm/params/parameters_MNDO_d_MOPAC.csv")
    om2 = pd.read_csv(CSV)
    d_params = {"U_dd": {}, "zeta_d": {}, "beta_d": {}}
    for z in (Z_S, Z_Cl):
        src = mndod[mndod.N == z].iloc[0]
        m = om2.N == z
        for col in ("U_dd", "zeta_d", "beta_d"):
            val = float(src[col])
            om2.loc[m, col] = val
            d_params[col][str(z)] = val
    om2.to_csv(CSV, index=False)

    out = {
        "source": "Thiel MNDO/d (parameters_MNDO_d_MOPAC.csv)",
        "d_params": d_params,
        "note": (
            "INITIAL OM2+d d-params by literature transfer from MNDO/d. "
            "Not re-fit to DFT/exp geometries: OM2 9-AO path lacks two-electron "
            "d integrals and is not geometry-safe. Ort/ECP d-factors remain 0. "
            "Enable with seqm_parameters['om2_use_d']=True when integrating OM2 TETCI-d."
        ),
    }
    path = DATA_DIR / "fitted_om2d_params.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out, indent=2) + "\n")
    print(f"Wrote {path}")
    print(f"Updated {CSV}")
    return out


def fit(*_args, **_kwargs):
    return adopt_thiel_mndod_seed()


def write_csv(fitted_path: Path | None = None):
    adopt_thiel_mndod_seed()


if __name__ == "__main__":
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from tools.om2d_benchmark.fit_om2d import adopt_thiel_mndod_seed as _main

    _main()
