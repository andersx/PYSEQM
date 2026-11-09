"""Thin PYSEQM energy/force + BFGS geometry optimization helpers."""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
from scipy.optimize import minimize

from seqm.ElectronicStructure import Electronic_Structure
from seqm.Molecule import Molecule
from seqm.seqm_functions.constants import Constants
from seqm.seqm_functions.omx_utils import build_beta_tensor

D_PARAM_NAMES = ("U_dd", "zeta_d", "beta_d")


def _base_params(method: str, om2_use_d: bool = True, fit_d: bool = False) -> dict[str, Any]:
    p = {
        "method": method,
        "scf_eps": 1.0e-7,
        "scf_converger": [1],
        "sp2": [False, 1.0e-5],
        "pair_outer_cutoff": 1.0e10,
        "eig": True,
        "Hf_flag": True,
        "scf_backward": 0,
        "UHF": False,
    }
    if method == "OM2":
        p["om2_use_d"] = om2_use_d
        if fit_d and om2_use_d:
            # Keep these out of the CSV merge so learned overrides stick.
            p["learned"] = list(D_PARAM_NAMES)
    return p


def _d_learned_for_atoms(atoms, d_params: dict | None, device, dtype):
    """Map {Z: value} dicts -> per-real-atom tensors (one molecule, no padding)."""
    if not d_params:
        return {}
    z = torch.as_tensor(list(atoms), dtype=torch.int64, device=device)
    out = {}
    for name in D_PARAM_NAMES:
        mp = d_params.get(name)
        if mp is None:
            continue
        t = torch.zeros(z.numel(), dtype=dtype, device=device)
        for zi, v in mp.items():
            t[z == int(zi)] = float(v)
        out[name] = t
    return out


def eval_energy_force(
    atoms,
    coords,
    method: str,
    om2_use_d: bool = True,
    d_params: dict | None = None,
):
    """
    d_params: optional {"U_dd": {16:..,17:..}, "zeta_d": {...}, "beta_d": {...}}
    """
    torch.set_default_dtype(torch.float64)
    device = torch.device("cpu")
    dtype = torch.float64
    species = torch.as_tensor([list(atoms)], dtype=torch.int64, device=device)
    coordinates = torch.as_tensor(np.asarray(coords, dtype=np.float64)[None, ...], dtype=dtype, device=device)
    const = Constants().to(device)
    fit_d = bool(d_params) and method == "OM2" and om2_use_d
    params = _base_params(method, om2_use_d=om2_use_d, fit_d=fit_d)
    params["elements"] = [0] + sorted(set(int(z) for z in atoms))
    learned = _d_learned_for_atoms(atoms, d_params, device, dtype) if fit_d else {}
    molecule = Molecule(const, params, coordinates, species, learned_parameters=learned).to(device)
    # Re-pass learned into the energy driver — Force/Energy re-packs parameters each call.
    Electronic_Structure(params).to(device)(molecule, learned_parameters=learned)
    e = float(molecule.Etot.item())
    f = molecule.force[0].detach().cpu().numpy()
    return e, f, molecule


def optimize_geometry(
    atoms,
    coords0,
    method: str,
    om2_use_d: bool = True,
    d_params: dict | None = None,
    maxiter: int = 100,
):
    """Cartesian L-BFGS-B using PYSEQM forces (energy eV, force eV/Å)."""
    x0 = np.asarray(coords0, dtype=np.float64).ravel()
    nat = len(atoms)
    cache = {}

    def _eval(x):
        key = x.tobytes()
        if key not in cache:
            e, f, _ = eval_energy_force(atoms, x.reshape(nat, 3), method, om2_use_d, d_params)
            cache[key] = (e, (-f).ravel())
        return cache[key]

    res = minimize(
        lambda x: _eval(x)[0],
        x0,
        jac=lambda x: _eval(x)[1],
        method="L-BFGS-B",
        options={"maxiter": maxiter, "ftol": 1e-8, "gtol": 1e-5},
    )
    coords = res.x.reshape(nat, 3)
    e, f, _ = eval_energy_force(atoms, coords, method, om2_use_d, d_params)
    return {
        "coords": coords,
        "energy_ev": e,
        "force_max": float(np.abs(f).max()),
        "success": bool(res.success),
        "nfev": int(res.nfev),
        "message": str(res.message),
    }
