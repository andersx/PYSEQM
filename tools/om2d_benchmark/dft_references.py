"""B3LYP/def2-SVP optimized geometries + reaction energy (pyscf)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from pyscf import dft, gto
from scipy.optimize import minimize

from .dataset import DATA_DIR, MOLECULES, REACTION, save_dataset

try:
    from pyscf.geomopt.geometric_solver import optimize as geometric_optimize
except Exception:  # geometric optional
    geometric_optimize = None

HARTREE2EV = 27.211386245988


def _mol(atoms, coords, charge=0):
    atom = [[int(z), c] for z, c in zip(atoms, coords)]
    return gto.M(atom=atom, basis="def2-svp", charge=charge, verbose=0)


def _energy(atoms, coords):
    mol = _mol(atoms, coords)
    mf = dft.RKS(mol)
    mf.xc = "b3lyp"
    mf.conv_tol = 1e-8
    return float(mf.kernel())


def _optimize_scipy(atoms, coords0, maxiter=80):
    """Fallback Cartesian BFGS with pyscf analytic gradients."""
    nat = len(atoms)
    x0 = np.asarray(coords0, dtype=np.float64).ravel()

    def fun(x):
        mol = _mol(atoms, x.reshape(nat, 3))
        mf = dft.RKS(mol)
        mf.xc = "b3lyp"
        mf.conv_tol = 1e-7
        e = mf.kernel()
        grad = mf.nuc_grad_method().kernel()  # Ha/Bohr
        # convert to Ha/Angstrom for Cartesian Å coordinates
        bohr = 0.529177210903
        return float(e), (grad / bohr).ravel()

    res = minimize(
        lambda x: fun(x)[0],
        x0,
        jac=lambda x: fun(x)[1],
        method="L-BFGS-B",
        options={"maxiter": maxiter, "ftol": 1e-8, "gtol": 1e-4},
    )
    coords = res.x.reshape(nat, 3)
    e = _energy(atoms, coords)
    return coords, e, bool(res.success)


def _optimize(atoms, coords0):
    if geometric_optimize is not None:
        try:
            mol = _mol(atoms, coords0)
            mf = dft.RKS(mol)
            mf.xc = "b3lyp"
            mf.conv_tol = 1e-8
            mf.kernel()
            mol_eq = geometric_optimize(mf, maxsteps=50)
            coords = np.asarray(mol_eq.atom_coords(unit="Angstrom"), dtype=np.float64)
            e = _energy(atoms, coords)
            return coords, e, True
        except Exception:
            pass
    return _optimize_scipy(atoms, coords0)


def build(path: Path | None = None) -> dict:
    path = path or (DATA_DIR / "dft_refs.json")
    path.parent.mkdir(parents=True, exist_ok=True)
    results = {"molecules": {}, "reaction": dict(REACTION)}
    energies = {}
    for name, spec in MOLECULES.items():
        print(f"[DFT] optimizing {name} ...", flush=True)
        coords, e, ok = _optimize(spec["atoms"], spec["coords"])
        energies[name] = e * HARTREE2EV
        results["molecules"][name] = {
            "coords": coords.tolist(),
            "energy_ha": e,
            "energy_ev": energies[name],
            "success": ok,
        }
        print(f"  E={e:.8f} Ha  ok={ok}", flush=True)

    de = 0.0
    for name in REACTION["products"]:
        de += energies[name]
    for name in REACTION["reactants"]:
        de -= energies[name]
    results["reaction"]["dft_de_ev"] = de
    print(f"[DFT] reaction ΔE = {de:.6f} eV", flush=True)

    path.write_text(json.dumps(results, indent=2) + "\n")

    # Also stamp dft_de_ev into the dataset JSON for the fitter.
    ds_path = save_dataset()
    ds = json.loads(ds_path.read_text())
    ds["reaction"]["dft_de_ev"] = de
    # store DFT opts as alternate starting geometries
    for name, block in results["molecules"].items():
        ds["molecules"][name]["dft_coords"] = block["coords"]
    ds_path.write_text(json.dumps(ds, indent=2) + "\n")
    return results


if __name__ == "__main__":
    build()
