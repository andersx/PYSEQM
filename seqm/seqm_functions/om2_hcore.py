import torch

from .method_utils import molecule_uses_d_basis
from .om1_overlap import (
    diatom_overlap_matrix_OM1,
    diatom_resonance_matrix_OM1,
    om1_local_overlap_terms,
    om1_local_resonance_terms,
)
from .om1_pair_backend import omx_pair_hcore_terms
from .omx_basis import select_om1_basis_payload
from .omx_utils import omx_pair_params
from .two_elec_two_center_int import rotate_with_quaternion

_OMX_HCORE_CONFIG = {
    "OM1": {"use_cor": False, "direct_resonance": True},
    "OM2": {"use_cor": True, "direct_resonance": False},
    "OM3": {"use_cor": False, "direct_resonance": False},
}


def _pad_sp_blocks_to_nbf(blocks_4, nbf):
    """Embed (..., 4, 4) sp blocks into (..., nbf, nbf); d channels left zero."""
    if nbf == 4:
        return blocks_4
    out = blocks_4.new_zeros(blocks_4.shape[:-2] + (nbf, nbf))
    out[..., :4, :4] = blocks_4
    return out


def _embed_w_sp_to_spd(w_sp, nbf):
    """
    Embed OM2/OM1 packed sp two-electron integrals (10×10) into spd (45×45).
    Indices among the first 4 AOs map onto the matching tril pairs of the 9-AO set.
    """
    if nbf == 4:
        return w_sp
    device, dtype = w_sp.device, w_sp.dtype
    tril4 = torch.tril_indices(4, 4, offset=0, device=device)
    tril9 = torch.tril_indices(nbf, nbf, offset=0, device=device)
    # map each 4-AO tril pair to its index in the nbf tril list
    key9 = tril9[0] * nbf + tril9[1]
    key4 = tril4[0] * nbf + tril4[1]
    # searchsorted requires sorted keys; tril_indices is row-major sorted for square
    idx = torch.searchsorted(key9, key4)
    npairs = w_sp.shape[0]
    npack = tril9.shape[1]
    w = w_sp.new_zeros((npairs, npack, npack))
    w[:, idx[:, None], idx[None, :]] = w_sp
    return w


def _build_cor_table(molecule, pair_core_semi, pi, pj, nbf):
    nmol, molsize = molecule.species.shape
    dtype = molecule.coordinates.dtype
    device = molecule.coordinates.device

    cor = torch.zeros((nmol, molsize * nbf, molsize), dtype=dtype, device=device)

    mol = molecule.pair_molid
    atom_i = molecule.maskd[molecule.idxi] % molsize
    atom_j = molecule.maskd[molecule.idxj] % molsize
    row_i = atom_i * nbf
    row_j = atom_j * nbf

    cor[mol, row_i + 0, atom_j] = pi["U_ss"] + pair_core_semi[:, 0, 0]
    cor[mol, row_j + 0, atom_i] = pj["U_ss"] + pair_core_semi[:, 0, 1]

    heavy_i = molecule.ni > 1
    if heavy_i.any():
        temp_i = (
            pi["U_pp"][heavy_i] + (pair_core_semi[heavy_i, 2, 0] + 2.0 * pair_core_semi[heavy_i, 3, 0]) / 3.0
        )
        for o in (1, 2, 3):
            cor[mol[heavy_i], row_i[heavy_i] + o, atom_j[heavy_i]] = temp_i

    heavy_j = molecule.nj > 1
    if heavy_j.any():
        temp_j = (
            pj["U_pp"][heavy_j] + (pair_core_semi[heavy_j, 2, 1] + 2.0 * pair_core_semi[heavy_j, 3, 1]) / 3.0
        )
        for o in (1, 2, 3):
            cor[mol[heavy_j], row_j[heavy_j] + o, atom_i[heavy_j]] = temp_j

    # Dummy d Ort/COR channels: leave zero (documented DUMMY — not for production).
    return cor


def build_omx_pair_context(molecule, method, idxi, idxj, ni, nj, xij, rij):
    pi, pj = omx_pair_params(molecule.parameters, idxi, idxj)
    basis_tables = molecule.parameters.get("_omx_basis")
    basis_data = molecule.parameters.get("_omx_basis_data")

    basis_i = select_om1_basis_payload(basis_data, idxi)
    basis_j = select_om1_basis_payload(basis_data, idxj)

    rot = rotate_with_quaternion(xij)
    rot_t = rot.transpose(1, 2)
    direction = rot_t[:, :, 0]

    s_local = om1_local_overlap_terms(rij, basis_i, basis_j)
    t_local = om1_local_resonance_terms(ni, nj, rij, pi, pj)
    pair = omx_pair_hcore_terms(
        method,
        ni,
        nj,
        rij,
        pi,
        pj,
        molecule.const.tore,
        s_local,
        t_local,
        rot,
        rot_t,
        basis_tables,
        basis_i,
        basis_j,
    )

    pair_resonance = diatom_resonance_matrix_OM1(xij, t_local, direction)
    context = {"pair": pair, "pair_resonance": pair_resonance, "pair_overlap": None}
    if method in {"OM2", "OM3"}:
        context["pair_overlap"] = diatom_overlap_matrix_OM1(xij, rij, direction, basis_i, basis_j, s_local)
    return context


def _apply_omx_orthogonalization(molecule, H_blocks, S_blocks, B_blocks, pi, pj, pair_core_semi, use_cor, nbf):
    S5 = S_blocks.reshape(molecule.nmol, molecule.molsize, molecule.molsize, nbf, nbf)
    B5 = B_blocks.reshape(molecule.nmol, molecule.molsize, molecule.molsize, nbf, nbf)
    COR = None
    gval2 = None
    if use_cor:
        gval2 = (pi["gval2"], pj["gval2"])
        COR = _build_cor_table(molecule, pair_core_semi, pi, pj, nbf)
        molecule.om2_COR = COR
    _add_betor_cor_dense_(H_blocks, S5, B5, molecule, (pi["gval1"], pj["gval1"]), COR, gval2, nbf)


def _shell_to_flat(X, nbf):
    """X: [M, N, N, nbf, nbf] -> [M, nbf*N, nbf*N]."""
    M, N = X.shape[:2]
    return X.permute(0, 1, 3, 2, 4).reshape(M, nbf * N, nbf * N)


def _flat_to_blocks(T, N, nbf):
    """T: [M, nbf*N, nbf*N] -> [M, N, N, nbf, nbf]."""
    M = T.shape[0]
    return T.reshape(M, N, nbf, N, nbf).permute(0, 1, 3, 2, 4)


def _add_betor_cor_dense_(out_blocks, S5, B5, molecule, gval1, COR, gval2, nbf):
    M, N = S5.shape[:2]
    dtype = S5.dtype

    mask = molecule.mask
    mol = molecule.pair_molid
    ai = (mask // N) % N
    aj = mask % N

    S2 = _shell_to_flat(S5, nbf)
    B2 = _shell_to_flat(B5, nbf)

    T1_flat = torch.bmm(S2, B2.transpose(1, 2))
    T1_flat += torch.bmm(B2, S2.transpose(1, 2))
    T1 = _flat_to_blocks(T1_flat, N, nbf)

    ft1 = 0.25 * (gval1[0] + gval1[1])
    hsrc = -ft1[:, None, None] * T1[mol, ai, aj]

    if COR is None:
        out_blocks.index_add_(0, molecule.mask, hsrc)
        return

    CORs = COR[:, 0::nbf, :]
    CORp = COR[:, 1::nbf, :]

    heavy = (molecule.species > 1).to(dtype)

    Hrow = S5.new_zeros((M, N, N, nbf))
    Hrow[..., 0] = CORs
    Hrow[..., 1:4] = CORp[..., None] * heavy[:, :, None, None]
    # d Ort factors left at 0 (dummy)

    Ccol = S5.new_zeros((M, N, N, nbf))
    Ccol[..., 0] = CORs.transpose(1, 2)
    Ccol[..., 1:4] = CORp.transpose(1, 2)[..., None] * heavy[:, None, :, None]

    D = S5 * (Hrow.unsqueeze(-1) - Ccol.unsqueeze(-2))
    D2 = _shell_to_flat(D, nbf)

    T2_flat = torch.bmm(D2, S2.transpose(1, 2))
    T2_flat += torch.bmm(S2, D2.transpose(1, 2))
    T2 = _flat_to_blocks(T2_flat, N, nbf)

    ft2 = 0.0625 * (gval2[0] + gval2[1])
    hsrc = hsrc + ft2[:, None, None] * T2[mol, ai, aj]

    out_blocks.index_add_(0, mask, hsrc)


def _fill_dummy_d_channels(molecule, H_blocks, S_blocks, B_blocks, pair_overlap, pair_resonance, nbf):
    """
    Placeholder for OM2+d off-diagonal d channels.

    Until OM2 two-electron d integrals exist, do **not** inject β_d·S_d
    resonance: that attraction without matching repulsion collapses bonds
    (e.g. linear H2S, Cl–Cl ~0.7 Å). Diagonal U_dd is already set in
    ``build_omx_hcore``. ζ_d / β_d remain available for a later TETCI stage.
    """
    return


def build_omx_hcore(molecule):
    method = molecule.method
    cfg = _OMX_HCORE_CONFIG[method]
    use_cor = cfg["use_cor"]
    direct_resonance = cfg["direct_resonance"]

    use_d = molecule_uses_d_basis(molecule)
    nbf = 9 if use_d else 4
    device = molecule.coordinates.device
    dtype = molecule.coordinates.dtype
    nblocks = molecule.nmol * molecule.molsize * molecule.molsize
    H_blocks = torch.zeros((nblocks, nbf, nbf), dtype=dtype, device=device)

    context = build_omx_pair_context(
        molecule, method, molecule.idxi, molecule.idxj, molecule.ni, molecule.nj, molecule.xij, molecule.rij
    )
    pi, pj = omx_pair_params(molecule.parameters, molecule.idxi, molecule.idxj)
    pair_data = context["pair"]

    # One-center U (sp always; dummy U_dd when present)
    H_blocks[molecule.maskd, 0, 0] = molecule.parameters["U_ss"].to(dtype=dtype, device=device)
    for orb in (1, 2, 3):
        H_blocks[molecule.maskd, orb, orb] = molecule.parameters["U_pp"].to(dtype=dtype, device=device)
    if use_d and "U_dd" in molecule.parameters:
        for orb in range(4, 9):
            H_blocks[molecule.maskd, orb, orb] = molecule.parameters["U_dd"].to(dtype=dtype, device=device)

    e1b = _pad_sp_blocks_to_nbf(pair_data["e1b"], nbf)
    e2a = _pad_sp_blocks_to_nbf(pair_data["e2a"], nbf)
    H_blocks.index_add_(0, molecule.maskd[molecule.idxi], e1b)
    H_blocks.index_add_(0, molecule.maskd[molecule.idxj], e2a)

    pair_resonance = context["pair_resonance"]
    H_blocks[molecule.mask, :4, :4] = pair_resonance

    w = _embed_w_sp_to_spd(pair_data["w"], nbf)

    if direct_resonance or molecule.molsize < 3:
        if use_d:
            _fill_dummy_d_channels(molecule, H_blocks, None, None, None, pair_resonance, nbf)
        return H_blocks, w, pair_data["fko"], None, None, None

    pair_overlap = context["pair_overlap"]
    S_blocks = torch.zeros_like(H_blocks)
    B_blocks = torch.zeros_like(H_blocks)
    S_blocks[molecule.mask, :4, :4] = pair_overlap
    S_blocks[molecule.mask_l, :4, :4] = pair_overlap.transpose(-1, -2)
    B_blocks[molecule.mask, :4, :4] = pair_resonance
    B_blocks[molecule.mask_l, :4, :4] = pair_resonance.transpose(-1, -2)

    if use_d:
        _fill_dummy_d_channels(
            molecule, H_blocks, S_blocks, B_blocks, pair_overlap, pair_resonance, nbf
        )

    _apply_omx_orthogonalization(
        molecule, H_blocks, S_blocks, B_blocks, pi, pj, pair_data.get("core_semi"), use_cor, nbf
    )
    return H_blocks, w, pair_data["fko"], None, None, None
