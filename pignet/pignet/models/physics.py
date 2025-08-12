from typing import Union

import torch

# For device-agnostic typing
FloatTensor = Union[torch.FloatTensor, torch.cuda.FloatTensor]
LongTensor = Union[torch.LongTensor, torch.cuda.LongTensor]
BoolTensor = Union[torch.BoolTensor, torch.cuda.BoolTensor]


def interaction_edges(
    ligand_mask: BoolTensor,
    batch: LongTensor,
) -> LongTensor:
    """\
    Args:
        ligand_mask: (nodes,)
        batch: (nodes,)

    Return: (2, node_pairs)
        Uni-directional, fully-connected ligand-protein edges.
    """
    device = ligand_mask.device
    nodes = torch.arange(ligand_mask.numel(), device=device)
    edges = torch.tensor([], dtype=torch.long, device=device)

    for i in range(batch.max() + 1):
        batch_mask = batch == i
        ligand_nodes = nodes[ligand_mask & batch_mask]
        target_nodes = nodes[~ligand_mask & batch_mask]
        edges_per_graph = torch.cartesian_prod(ligand_nodes, target_nodes)
        edges = torch.cat((edges, edges_per_graph))
        # -> (node_pairs, 2)

    return edges.t().contiguous()


def distances(
    pos: FloatTensor,
    edge_index: LongTensor,
) -> FloatTensor:
    """\
    Args:
        pos: (nodes, 3)
        edge_index: (2, pairs)

    Return: (pairs,)
    """
    pos1 = pos[edge_index[0]]
    pos2 = pos[edge_index[1]]
    # -> (edges, 3)
    D = torch.sqrt(torch.pow(pos1 - pos2, 2).sum(-1) + 1e-10)
    return D


def lennard_jones_potential(
    D: FloatTensor,
    R: FloatTensor,
    E: Union[float, FloatTensor],
    n_short: int,
    n_long: int,
) -> FloatTensor:
    """\
    Args:
        D: (pairs,)
            Pairwise distances.
        R: (pairs,)
            Pairwise sum of (corrected) vdW radii.
        E: () or (pairs,)
            Pairwise energy minima.
        n_sort: short-range base order of L-J potential.
        n_long: long-range base order of L-J potential.
            The form of L-J potential is divided into a short-range form
            and a long-range form.
            The original order of 12-6 becomes "2*n_short-n_sort" for short
            range and "2*n_long-n_long" for long range.
            The two potentials are ultimately mixed by comparing `D` and `R`.

    Return: (pairs,)
    """

    def eqn(D, R, N):
        """L-J potential calculator."""
        term1 = torch.pow(R / D, N * 2)
        term2 = -2 * torch.pow(R / D, N)
        return term1 + term2

    energy_short = eqn(D, R, n_short)
    energy_long = eqn(D, R, n_long)

    energy = torch.where(D > R, energy_long, energy_short)
    energy = energy.clamp(max=100.0)
    energy = energy * E
    return energy


def morse_potential(
    D: FloatTensor,
    R: FloatTensor,
    E: Union[float, FloatTensor],
    A: Union[float, FloatTensor],
    short_range_A: float = None,
) -> FloatTensor:
    """\
    Args:
        D: (pairs,)
            Pairwise distances.
        R: (pairs,)
            Pairwise sum of (corrected) vdW radii.
        E: () or (pairs,)
            Pairwise energy minima.
        A: () or (pairs,)
            Pairwise energy width.

    Return: (pairs,)
    """
    energy = (1 - torch.exp(-A * (D - R))) ** 2 - 1
    if short_range_A is not None:
        energy2 = (1 - torch.exp(-short_range_A * (D - R))) ** 2 - 1
        energy = torch.where(D > R, energy, energy2)

    energy = energy.clamp(max=100.0)
    energy = energy * E
    return energy


def linear_potential(
    D: FloatTensor,
    R: FloatTensor,
    E: Union[float, FloatTensor],
    c1: float,
    c2: float,
) -> FloatTensor:
    """\
    Args:
        D: (pairs,)
            Pairwise distances.
        R: (pairs,)
            Pairwise sum of (corrected) vdW radii.
        E: () or (pairs,)
            Pairwise energy minima.
        c1, c2: float
            Cutoff distances to interpolate `E` to 0.

    Return: (pairs,)
    """
    energy = (D - R - c2) / (c1 - c2)
    energy = energy.clamp(min=0.0, max=1.0)
    energy = energy * E
    return energy


def interaction_masks(
    metal_mask: BoolTensor,
    h_donor_mask: BoolTensor,
    h_acceptor_mask: BoolTensor,
    hydrophobic_mask: BoolTensor,
    edge_index: LongTensor,
    include_ionic: bool = False,
) -> BoolTensor:
    def combine(m1, m2=None):
        nonlocal edge_index
        if m2 is None:
            m = m1[edge_index[0]] & m1[edge_index[1]]
        else:
            m = m1[edge_index[0]] & m2[edge_index[1]]
            m = m | (m2[edge_index[0]] & m1[edge_index[1]])
        return m

    masks = torch.stack(
        (
            # vdw
            combine(~metal_mask),
            # Hydrogen bond, igonre metal atom to hbonding
            combine(h_donor_mask & (~metal_mask), h_acceptor_mask & (~metal_mask)),
            # Metal-ligand
            combine(metal_mask, h_acceptor_mask & (~metal_mask)),
            # Hydrophobic
            combine(hydrophobic_mask),
        )
    )
    if include_ionic:
        # Atomic charges will zero out vanishing ionic interactions,
        # so the mask is just set as all True.
        mask_ionic = torch.ones(edge_index.size()[1], dtype=torch.bool)
        mask_ionic = mask_ionic.to(edge_index.device)
        masks = torch.cat((masks, mask_ionic.unsqueeze(0)))
    return masks
