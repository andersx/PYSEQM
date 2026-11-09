"""OM2+d infrastructure: HCNOF regression + S/Cl dummy-d smoke tests."""

from __future__ import annotations

import torch

from seqm.ElectronicStructure import Electronic_Structure
from seqm.Molecule import Molecule
from seqm.seqm_functions.constants import Constants

WATER = (
    [8, 1, 1],
    [
        [0.0, 0.0, 0.1173],
        [0.0, 0.7572, -0.4692],
        [0.0, -0.7572, -0.4692],
    ],
)

H2S = (
    [16, 1, 1],
    [
        [0.0, 0.0, 0.103],
        [0.0, 0.9616, -0.823],
        [0.0, -0.9616, -0.823],
    ],
)

# Locked OM2 water energy on this geometry (HCNOF must stay bit-stable).
OM2_WATER_ETOT = -361.58824223810245


def _params():
    return {
        "method": "OM2",
        "scf_eps": 1.0e-8,
        "scf_converger": [1],
        "sp2": [False, 1.0e-5],
        "pair_outer_cutoff": 1.0e10,
        "eig": True,
        "Hf_flag": True,
        "scf_backward": 0,
        "UHF": False,
    }


def _run(atoms, coords):
    torch.set_default_dtype(torch.float64)
    const = Constants()
    params = _params()
    molecule = Molecule(
        const,
        params,
        torch.tensor([coords], dtype=torch.float64),
        torch.as_tensor([atoms], dtype=torch.int64),
    )
    Electronic_Structure(params)(molecule)
    return molecule


def test_om2_water_hcno_regression():
    mol = _run(*WATER)
    assert mol.nSuperHeavy.item() == 0
    assert mol.norb.item() == 6
    assert abs(mol.Etot.item() - OM2_WATER_ETOT) < 1e-8


def test_om2_h2s_dummy_d_converges_with_finite_forces():
    """S uses 9 AO with dummy MNDO/d-inspired d-params — smoke only, not MNDO99."""
    mol = _run(*H2S)
    assert mol.nSuperHeavy.item() == 1
    assert mol.norb.item() == 11
    assert torch.isfinite(mol.Etot).all()
    assert torch.isfinite(mol.force).all()
    assert mol.force.abs().max().item() > 0.0
