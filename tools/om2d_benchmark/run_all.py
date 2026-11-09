#!/usr/bin/env python3
"""Build DFT refs → compare methods → fit OM2+d d-params → re-compare → write CSV."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Allow `python tools/om2d_benchmark/run_all.py` from repo root
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from tools.om2d_benchmark.dataset import DATA_DIR, save_dataset  # noqa: E402
from tools.om2d_benchmark import compare, dft_references, fit_om2d  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-dft", action="store_true", help="Reuse existing dft_refs.json")
    ap.add_argument("--skip-fit", action="store_true")
    ap.add_argument("--fit-maxiter", type=int, default=20)
    ap.add_argument("--compare-only", action="store_true")
    args = ap.parse_args()

    save_dataset()
    dft_path = DATA_DIR / "dft_refs.json"

    if args.compare_only:
        compare.run()
        return

    if not args.skip_dft or not dft_path.exists():
        dft_references.build()
    else:
        print(f"Reusing {dft_path}")

    print("\n######## BEFORE FIT ########")
    compare.run(out_path=DATA_DIR / "compare_before.json")

    fitted = None
    if not args.skip_fit:
        fitted = fit_om2d.fit(maxiter=args.fit_maxiter)
        fit_om2d.write_csv()
        d_params = {
            "U_dd": {int(k): v for k, v in fitted["d_params"]["U_dd"].items()},
            "zeta_d": {int(k): v for k, v in fitted["d_params"]["zeta_d"].items()},
            "beta_d": {int(k): v for k, v in fitted["d_params"]["beta_d"].items()},
        }
        print("\n######## AFTER FIT ########")
        compare.run(d_params=d_params, out_path=DATA_DIR / "compare_after.json")

    # Markdown report
    report = DATA_DIR / "REPORT.md"
    before = json.loads((DATA_DIR / "compare_before.json").read_text())
    after = None
    if (DATA_DIR / "compare_after.json").exists():
        after = json.loads((DATA_DIR / "compare_after.json").read_text())
    lines = [
        "# OM2+d small-set benchmark",
        "",
        "Closed-shell molecules where S/Cl d orbitals matter, plus",
        "`H2S + Cl2 -> SCl2 + H2`. Geometry refs: experiment. Reaction ref: B3LYP/def2-SVP.",
        "",
        "## Method comparison (before fit)",
        "",
        "| Method | bond MAE (Å) | angle MAE (°) | ΔE (eV) | ΔE−DFT |",
        "|---|---:|---:|---:|---:|",
    ]
    for label, block in before["methods"].items():
        g = block.get("geom_mae", {})
        lines.append(
            f"| {label} | {g.get('bond_A')} | {g.get('angle_deg')} | "
            f"{block.get('reaction_de_ev')} | {block.get('reaction_err_vs_dft')} |"
        )
    if after:
        lines += ["", "## After OM2+d d-param fit", ""]
        lines += [
            "| Method | bond MAE (Å) | angle MAE (°) | ΔE (eV) | ΔE−DFT |",
            "|---|---:|---:|---:|---:|",
        ]
        for label, block in after["methods"].items():
            g = block.get("geom_mae", {})
            lines.append(
                f"| {label} | {g.get('bond_A')} | {g.get('angle_deg')} | "
                f"{block.get('reaction_de_ev')} | {block.get('reaction_err_vs_dft')} |"
            )
    if fitted:
        lines += ["", "## Fitted d-params (S, Cl)", "", "```json", json.dumps(fitted["d_params"], indent=2), "```"]
    report.write_text("\n".join(lines) + "\n")
    print(f"Wrote {report}")


if __name__ == "__main__":
    main()
