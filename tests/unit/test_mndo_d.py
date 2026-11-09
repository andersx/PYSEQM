"""Closed-shell MNDO/d energy (+ force FD) checks vs MNDO99 references."""

from __future__ import annotations

import torch

from seqm.ElectronicStructure import Electronic_Structure
from seqm.Molecule import Molecule
from seqm.seqm_functions.constants import Constants

# MNDO99 (iop=-10) SCF TOTAL ENERGY in eV for the geometries below.
# Water: MNDO/d ≡ MNDO for HCNOF.
MNDO99_ETOT_EV = {
    "water": -351.40984,
    "H2S": -216.84580,
    "Cl2": -529.19415,
    "SO2": -831.23559,
}

GEOMS = {
    "water": (
        [8, 1, 1],
        [
            [0.0, 0.0, 0.1173],
            [0.0, 0.7572, -0.4692],
            [0.0, -0.7572, -0.4692],
        ],
    ),
    "H2S": (
        [16, 1, 1],
        [
            [0.0, 0.0, 0.103],
            [0.0, 0.9616, -0.823],
            [0.0, -0.9616, -0.823],
        ],
    ),
    "Cl2": (
        [17, 17],
        [
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 1.988],
        ],
    ),
    "SO2": (
        [16, 8, 8],
        [
            [0.0, 0.0, 0.0],
            [0.0, 1.2463, 0.7227],
            [0.0, -1.2463, 0.7227],
        ],
    ),
}


def _seqm_params(method="MNDO_d"):
    return {
        "method": method,
        "scf_eps": 1.0e-8,
        "scf_converger": [1],
        "sp2": [False, 1.0e-5],
        "pair_outer_cutoff": 1.0e10,
        "eig": True,
        "Hf_flag": True,
        "scf_backward": 0,
        "UHF": False,
    }


def _run(atoms, coords, method="MNDO_d", device="cpu"):
    torch.set_default_dtype(torch.float64)
    device = torch.device(device)
    species = torch.as_tensor([atoms], dtype=torch.int64, device=device)
    coordinates = torch.tensor([coords], dtype=torch.float64, device=device)
    const = Constants().to(device)
    params = _seqm_params(method)
    molecule = Molecule(const, params, coordinates, species).to(device)
    Electronic_Structure(params).to(device)(molecule)
    return molecule


def test_mndo_d_alias_normalizes():
    mol = _run(*GEOMS["water"], method="MNDO/d")
    assert mol.method == "MNDO_d"


def test_water_mndo_d_matches_mndo_and_mndo99():
    """HCNOF: MNDO/d reduces to MNDO."""
    mndo = _run(*GEOMS["water"], method="MNDO")
    mndod = _run(*GEOMS["water"], method="MNDO_d")
    assert abs(mndo.Etot.item() - mndod.Etot.item()) < 1e-8
    assert abs(mndod.Etot.item() - MNDO99_ETOT_EV["water"]) < 1e-4


def test_h2s_mndo_d_energy_vs_mndo99():
    mol = _run(*GEOMS["H2S"], method="MNDO_d")
    assert abs(mol.Etot.item() - MNDO99_ETOT_EV["H2S"]) < 1e-4
    # With d on S, energy must differ from plain MNDO.
    mndo = _run(*GEOMS["H2S"], method="MNDO")
    assert abs(mol.Etot.item() - mndo.Etot.item()) > 1.0


def test_cl2_mndo_d_energy_vs_mndo99():
    mol = _run(*GEOMS["Cl2"], method="MNDO_d")
    assert abs(mol.Etot.item() - MNDO99_ETOT_EV["Cl2"]) < 1e-4


def test_so2_mndo_d_energy_vs_mndo99():
    """S–O with d on S only: nuclear matches tightly; electronic ~1–2 meV residual."""
    mol = _run(*GEOMS["SO2"], method="MNDO_d")
    assert abs(mol.Etot.item() - MNDO99_ETOT_EV["SO2"]) < 2e-3


def test_h2s_mndo_d_forces_match_finite_difference():
    atoms, coords0 = GEOMS["H2S"]
    mol = _run(atoms, coords0, method="MNDO_d")
    force = mol.force[0].detach().clone()

    eps = 1.0e-4
    coords = torch.tensor([coords0], dtype=torch.float64)
    species = torch.as_tensor([atoms], dtype=torch.int64)
    const = Constants()
    params = _seqm_params("MNDO_d")
    ffd = torch.zeros_like(force)
    for a in range(coords.shape[1]):
        for c in range(3):
            xp = coords.clone()
            xm = coords.clone()
            xp[0, a, c] += eps
            xm[0, a, c] -= eps
            mp = Molecule(const, params, xp, species)
            mm = Molecule(const, params, xm, species)
            Electronic_Structure(params)(mp)
            Electronic_Structure(params)(mm)
            ffd[a, c] = -(mp.Etot.item() - mm.Etot.item()) / (2.0 * eps)

    assert torch.allclose(force, ffd, rtol=1e-4, atol=1e-5)
