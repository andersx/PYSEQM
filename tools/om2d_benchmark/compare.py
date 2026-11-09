"""Optimize the test set with MNDO / MNDO_d / OM2(sp) / OM2+d and print a report."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .dataset import DATA_DIR, load_dataset
from .metrics import geometry_errors, reaction_energy
from .pyseqm_driver import optimize_geometry

METHODS = [
    ("MNDO", "MNDO", True),
    ("MNDO_d", "MNDO_d", True),
    ("OM2_sp", "OM2", False),
    ("OM2_d", "OM2", True),
]


def run(d_params=None, out_path: Path | None = None) -> dict:
    ds = load_dataset()
    mols = ds["molecules"]
    reaction = ds["reaction"]
    out_path = out_path or (DATA_DIR / "compare_results.json")

    report = {"methods": {}, "reaction": {}}
    summary_rows = []

    for label, method, use_d in METHODS:
        print(f"\n=== {label} (method={method}, om2_use_d={use_d}) ===", flush=True)
        block = {"molecules": {}, "geom_mae": {}}
        energies = {}
        bond_errs, angle_errs = [], []

        for name, spec in mols.items():
            if name == "H2" and method.startswith("MNDO"):
                # still fine for MNDO
                pass
            coords0 = spec.get("dft_coords", spec["coords"])
            print(f"  opt {name} ...", flush=True)
            try:
                opt = optimize_geometry(
                    spec["atoms"],
                    coords0,
                    method=method,
                    om2_use_d=use_d,
                    d_params=d_params if (method == "OM2" and use_d) else None,
                )
            except Exception as exc:
                print(f"    FAILED: {exc}", flush=True)
                block["molecules"][name] = {"error": str(exc)}
                continue

            energies[name] = opt["energy_ev"]
            errs = geometry_errors(opt["coords"], spec["exp"])
            for lab, val, ref, ae in errs:
                if lab.startswith("r"):
                    bond_errs.append(ae)
                else:
                    angle_errs.append(ae)
            block["molecules"][name] = {
                "energy_ev": opt["energy_ev"],
                "force_max": opt["force_max"],
                "success": opt["success"],
                "coords": opt["coords"].tolist(),
                "errors": [
                    {"label": lab, "value": val, "ref": ref, "abs_err": ae} for lab, val, ref, ae in errs
                ],
            }
            pretty = ", ".join(f"{lab}={val:.4f}(ref {ref:.4f}, |e|={ae:.4f})" for lab, val, ref, ae in errs)
            print(f"    E={opt['energy_ev']:.6f} eV  |F|max={opt['force_max']:.3e}  {pretty}", flush=True)

        block["geom_mae"]["bond_A"] = float(np.mean(bond_errs)) if bond_errs else None
        block["geom_mae"]["angle_deg"] = float(np.mean(angle_errs)) if angle_errs else None

        # reaction among available species
        needed = set(reaction["reactants"] + reaction["products"])
        if needed.issubset(energies):
            de = reaction_energy(energies, reaction)
            block["reaction_de_ev"] = de
            dft_de = reaction.get("dft_de_ev")
            block["reaction_err_vs_dft"] = None if dft_de is None else de - dft_de
            print(
                f"  ΔE({reaction['name']}) = {de:.4f} eV"
                + (f"  (DFT {dft_de:.4f}, err {de - dft_de:+.4f})" if dft_de is not None else ""),
                flush=True,
            )
        report["methods"][label] = block
        summary_rows.append(
            (
                label,
                block["geom_mae"]["bond_A"],
                block["geom_mae"]["angle_deg"],
                block.get("reaction_de_ev"),
                block.get("reaction_err_vs_dft"),
            )
        )

    print("\n=== Summary (MAE bonds Å / angles deg, reaction ΔE eV) ===")
    for row in summary_rows:
        label, b, a, de, derr = row
        print(
            f"  {label:8s}  bond_MAE={b and f'{b:.4f}' or 'n/a':>8s}  "
            f"angle_MAE={a and f'{a:.3f}' or 'n/a':>8s}  "
            f"ΔE={de and f'{de:+.4f}' or 'n/a':>9s}  "
            f"ΔE-DFT={derr and f'{derr:+.4f}' or 'n/a':>9s}"
        )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2) + "\n")
    print(f"\nWrote {out_path}")
    return report


if __name__ == "__main__":
    run()
