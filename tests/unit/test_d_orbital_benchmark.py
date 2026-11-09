"""MNDO vs MNDO/d on SO2: d orbitals improve the O-S-O angle vs experiment."""

from __future__ import annotations

import torch

from seqm.ElectronicStructure import Electronic_Structure
from seqm.Molecule import Molecule
from seqm.seqm_functions.constants import Constants


def _angle_deg(coords, i, j, k):
    a = coords[i] - coords[j]
    b = coords[k] - coords[j]
    cos = (a * b).sum() / (a.norm() * b.norm())
    return float(torch.rad2deg(torch.arccos(cos.clamp(-1, 1))).item())


def _optimize(method, maxiter=60):
    # SO2, Z non-increasing
    atoms = [16, 8, 8]
    coords0 = torch.tensor(
        [[[0.0, 0.0, 0.0], [0.0, 1.2463, 0.7227], [0.0, -1.2463, 0.7227]]],
        dtype=torch.float64,
    )
    species = torch.as_tensor([atoms], dtype=torch.int64)
    const = Constants()
    params = {
        "method": method,
        "scf_eps": 1.0e-7,
        "scf_converger": [1],
        "sp2": [False, 1.0e-5],
        "pair_outer_cutoff": 1.0e10,
        "eig": True,
        "Hf_flag": True,
        "scf_backward": 0,
        "UHF": False,
        "elements": [0, 8, 16],
    }
    # Lightweight Cartesian L-BFGS via repeated force steps
    from scipy.optimize import minimize
    import numpy as np

    def pack_eval(x):
        xyz = torch.as_tensor(x.reshape(1, 3, 3), dtype=torch.float64)
        mol = Molecule(const, dict(params), xyz, species)
        Electronic_Structure(params)(mol)
        return float(mol.Etot.item()), (-mol.force[0].detach().cpu().numpy()).ravel()

    x0 = coords0.numpy().ravel()
    res = minimize(
        lambda x: pack_eval(x)[0],
        x0,
        jac=lambda x: pack_eval(x)[1],
        method="L-BFGS-B",
        options={"maxiter": maxiter, "ftol": 1e-8, "gtol": 1e-5},
    )
    return res.x.reshape(3, 3)


def test_mndo_d_improves_so2_angle_vs_mndo():
    torch.set_default_dtype(torch.float64)
    exp = 119.33
    xyz_mndo = _optimize("MNDO")
    xyz_d = _optimize("MNDO_d")
    err_mndo = abs(_angle_deg(torch.as_tensor(xyz_mndo), 1, 0, 2) - exp)
    err_d = abs(_angle_deg(torch.as_tensor(xyz_d), 1, 0, 2) - exp)
    assert err_d < err_mndo
    assert err_d < 5.0  # MNDO/d should be within ~5° of experiment on this set
