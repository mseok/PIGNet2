import os
import pickle
from typing import Any, Dict, List, Optional, Tuple

import torch
from rdkit import Chem
from rdkit.Chem import rdMolDescriptors, rdmolops
from torch_geometric.data import Data, Dataset
from torch_geometric.utils import dense_to_sparse

from . import chem


def one_hot_encode(
    x: Any,
    kinds: List[Any],
    handle_unknown: str = "error",
) -> List[bool]:
    """
    Make a one-hot vector.

    Args:
        handle_unknown: 'error' | 'ignore' | 'last'
            If `x` not in `kinds`:
              'error' -> raise ValueError
              'ignore' -> return zero vector
              'last' -> use the last kind.
    """
    onehot = [False] * len(kinds)
    try:
        onehot[kinds.index(x)] = True

    except ValueError:
        if handle_unknown == "error":
            msg = f"input {x} not in the allowed set {kinds}"
            raise ValueError(msg)
        elif handle_unknown == "ignore":
            pass
        elif handle_unknown == "last":
            onehot[-1] = True
        else:
            raise NotImplementedError

    return onehot


def get_period_group(atom: Chem.Atom) -> List[bool]:
    period, group = chem.PERIODIC_TABLE[atom.GetSymbol().upper()]
    period_vec = one_hot_encode(period, chem.PERIODS)
    group_vec = one_hot_encode(group, chem.GROUPS)
    total_vec = period_vec + group_vec
    return total_vec


def get_vdw_radius(atom: Chem.Atom) -> float:
    atomic_number = atom.GetAtomicNum()
    try:
        radius = chem.VDW_RADII[atomic_number]
    except KeyError:
        radius = Chem.GetPeriodicTable().GetRvdw(atomic_number)
    return radius


def get_atom_charges(mol: Chem.Mol) -> List[float]:
    charges = [atom.GetFormalCharge() for atom in mol.GetAtoms()]
    return charges


def get_metals(mol: Chem.Mol) -> List[bool]:
    mask = [atom.GetSymbol() in chem.METALS for atom in mol.GetAtoms()]
    return mask


def get_smarts_matches(mol: Chem.Mol, smarts: str) -> List[bool]:
    # Get the matching atom indices.
    pattern = Chem.MolFromSmarts(smarts)
    matches = {idx for match in mol.GetSubstructMatches(pattern) for idx in match}

    # Convert to a mask vector.
    mask = [idx in matches for idx in range(mol.GetNumAtoms())]
    return mask


def get_hydrophobes(mol: Chem.Mol) -> List[bool]:
    mask = []

    for atom in mol.GetAtoms():
        symbol = atom.GetSymbol().upper()
        if symbol in chem.HYDROPHOBES:
            # Check if all neighbors are also in `hyd_atoms`.
            neighbor_symbols = {
                neighbor.GetSymbol().upper() for neighbor in atom.GetNeighbors()
            }
            neighbors_not_hyd = neighbor_symbols - chem.HYDROPHOBES
            mask.append(len(neighbors_not_hyd) == 0)
        else:
            mask.append(False)

    return mask


def atom_to_features(atom: Chem.Atom) -> List[bool]:
    # Total 47, currently.
    features = [
        # Symbol (10)
        one_hot_encode(atom.GetSymbol(), chem.ATOM_SYMBOLS, "last"),
        # Degree (6)
        one_hot_encode(atom.GetDegree(), chem.ATOM_DEGREES, "last"),
        # Hybridization (7)
        one_hot_encode(atom.GetHybridization(), chem.HYBRIDIZATIONS, "last"),
        # Period & group (23)
        get_period_group(atom),
        # Aromaticity (1)
        [atom.GetIsAromatic()],
    ]

    # Flatten
    features = [value for row in features for value in row]
    return features


def mol_to_data(
    mol: Chem.Mol,
    remove_hydrogens: bool = True,
    pos_noise_std: float = 0.0,
    pos_noise_max: float = 0.0,
) -> Data:
    """Convert a RDKit mol to PyG data.
    Every numerical attributes are converted to torch.tensor.
    Note that label `y` is not set here.

    Data attributes:
        x: (num_atoms, num_atom_features), float
        edge_index: (2, num_bonds), long
        pos: (num_atoms, 3), float
        vdw_radii: (num_atoms,), float
        is_metal: (num_atoms,), bool
        is_h_donor: (num_atoms,), bool
        is_h_acceptor: (num_atoms,), bool
        is_hydrophobic: (num_atoms,), bool
    """
    if remove_hydrogens:
        mol = Chem.RemoveAllHs(mol)

    # Node features
    x = torch.tensor(
        [atom_to_features(atom) for atom in mol.GetAtoms()], dtype=torch.float
    )
    # Adjacency matrix
    # Self-loops will be added in GNNs only when necessary.
    adj = torch.tensor(rdmolops.GetAdjacencyMatrix(mol))
    # Convert to the sparse, long-type form.
    edge_index, edge_attr = dense_to_sparse(adj)

    data = Data()
    data.x = x
    data.edge_index = edge_index

    # Cartesian coordinates
    try:
        pos = mol.GetConformers()[0].GetPositions()
    except IndexError:
        msg = "No position in the `Chem.Mol` data!"
        raise RuntimeError(msg)
    data.pos = torch.tensor(pos, dtype=torch.float)

    noise = torch.zeros_like(data.pos)
    if pos_noise_std and pos_noise_max:
        noise += torch.normal(0, pos_noise_std, size=noise.shape)
        noise.clamp_(-pos_noise_max, pos_noise_max)
    elif pos_noise_std:
        noise += torch.normal(0, pos_noise_std, size=noise.shape)
    elif pos_noise_max:
        noise += (pos_noise_max * 2) * torch.rand(noise.shape) - pos_noise_max
    data.pos += noise

    # VdW radii
    vdw_radii = [get_vdw_radius(atom) for atom in mol.GetAtoms()]
    data.vdw_radii = torch.tensor(vdw_radii, dtype=torch.float)

    # atomic charge
    atom_charges = get_atom_charges(mol)
    data.atom_charges = torch.tensor(atom_charges)

    # Masks
    metals = get_metals(mol)
    h_donors = get_smarts_matches(mol, chem.H_DONOR_SMARTS)
    h_acceptors = get_smarts_matches(mol, chem.H_ACCEPTOR_SMARTS)
    hydrophobes = get_hydrophobes(mol)
    # Expect bool tensors, but the exact dtype won't be important.
    data.is_metal = torch.tensor(metals)
    data.is_h_donor = torch.tensor(h_donors)
    data.is_h_acceptor = torch.tensor(h_acceptors)
    data.is_hydrophobic = torch.tensor(hydrophobes)

    return data


def get_complex_edges(
    pos1: torch.Tensor,
    pos2: torch.Tensor,
    min_distance: float,
    max_distance: float,
) -> torch.LongTensor:
    """\
    Args:
        pos1: (num_atoms1, 3)
        pos2: (num_atoms2, 3)
        min_distance, max_distance:
            Atoms a_i and a_j are deemed connected if:
                min_distance <= d_ij <= max_distance
    """
    # Distance matrix
    D = torch.sqrt(
        torch.pow(pos1.view(-1, 1, 3) - pos2.view(1, -1, 3), 2).sum(-1) + 1e-10
    )
    # -> (num_atoms1, num_atoms2)

    # Rectangular adjacency matrix
    A = torch.zeros_like(D)
    A[(min_distance <= D) & (D <= max_distance)] = 1.0

    # Convert to a sparse edge-index tensor.
    edge_index = []
    for i in range(A.size(0)):
        for j in torch.nonzero(A[i]).view(-1):
            j_shifted = j.item() + pos1.size(0)
            edge_index.append([i, j_shifted])
            edge_index.append([j_shifted, i])
    edge_index = torch.tensor(edge_index).t().contiguous()

    # Some complexes can have no intermolecular edge.
    if not edge_index.numel():
        edge_index = edge_index.view(2, -1).long()

    return edge_index


def complex_to_data(
    mol_ligand: Chem.Mol,
    mol_target: Chem.Mol,
    label: Optional[float] = None,
    key: Optional[str] = None,
    conv_range: Tuple[float, float] = None,
    remove_hydrogens: bool = True,
    pos_noise_std: float = 0.0,
    pos_noise_max: float = 0.0,
) -> Data:
    """\
    Data attributs (additional to `mol_to_data`):
        y: (1, 1), float
        key: str
        rotor: (1, 1), float
        is_ligand: (num_ligand_atoms + num_target_atoms,), bool
        edge_index_c: (2, num_edges), long
            Intermolecular edges for graph convolution.
        mol_ligand: Chem.Mol
            Ligand Mol object used for docking.
        mol_target: Chem.Mol
            Target Mol object used for docking.
    """
    ligand = mol_to_data(mol_ligand, remove_hydrogens, pos_noise_std, pos_noise_max)
    target = mol_to_data(mol_target, remove_hydrogens, pos_noise_std, pos_noise_max)
    data = Data()

    if remove_hydrogens:
        mol_ligand = Chem.RemoveAllHs(mol_ligand)
        mol_target = Chem.RemoveAllHs(mol_target)

    # Combine the values.
    assert set(ligand.keys) == set(target.keys)
    for attr in ligand.keys:
        ligand_value = ligand[attr]
        target_value = target[attr]

        # Shift atom indices for some attributes.
        if attr in ("edge_index",):
            target_value = target_value + ligand.num_nodes

        # Dimension to concatenate over.
        cat_dim = ligand.__cat_dim__(attr, None)
        value = torch.cat((ligand_value, target_value), cat_dim)
        data[attr] = value

    if label is not None:
        data.y = torch.tensor(label, dtype=torch.float).view(1, 1)

    if key is not None:
        data.key = key

    rotor = rdMolDescriptors.CalcNumRotatableBonds(mol_ligand)
    data.rotor = torch.tensor(rotor, dtype=torch.float).view(1, 1)

    # Ligand mask
    is_ligand = [True] * ligand.num_nodes + [False] * target.num_nodes
    data.is_ligand = torch.tensor(is_ligand)

    # Intermolecular edges
    if conv_range is not None:
        data.edge_index_c = get_complex_edges(ligand.pos, target.pos, *conv_range)

    # Save the Mol objects; used for docking.
    data.mol_ligand = mol_ligand
    data.mol_target = mol_target

    return data


class ComplexDataset(Dataset):
    def __init__(
        self,
        keys: List[str],
        data_dir: Optional[str] = None,
        id_to_y: Optional[Dict[str, float]] = None,
        conv_range: Optional[Tuple[float, float]] = None,
        processed_data_dir: Optional[str] = None,
        pos_noise_std: float = 0.0,
        pos_noise_max: float = 0.0,
    ):
        assert data_dir is not None or processed_data_dir is not None

        super().__init__()
        self.keys = keys
        self.data_dir = data_dir
        self.id_to_y = id_to_y
        self.conv_range = conv_range
        self.processed_data_dir = processed_data_dir
        self.pos_noise_std = pos_noise_std
        self.pos_noise_max = pos_noise_max

    def len(self) -> int:
        return len(self.keys)

    def get(self, idx) -> Data:
        key = self.keys[idx]

        # Setting 'processed_data_dir' takes priority than 'data_dir'.
        if self.processed_data_dir is not None:
            data_path = os.path.join(self.processed_data_dir, key + ".pt")

            with open(data_path, "rb") as f:
                data = torch.load(f)

        elif self.data_dir is not None:
            # pK_d -> kcal/mol
            label = self.id_to_y[key] * -1.36
            data_path = os.path.join(self.data_dir, key)

            # Unpickle the `Chem.Mol` data.
            with open(data_path, "rb") as f:
                data = pickle.load(f)
                try:
                    mol_ligand, _, mol_target, _ = data
                except ValueError:
                    mol_ligand, mol_target = data

            data = complex_to_data(
                mol_ligand,
                mol_target,
                label,
                key,
                self.conv_range,
                pos_noise_std=self.pos_noise_std,
                pos_noise_max=self.pos_noise_max,
            )

        return data
