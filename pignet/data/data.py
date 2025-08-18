import os
import pickle
from typing import Any

import torch
from rdkit import Chem
from rdkit.Chem import BRICS, rdMolDescriptors, rdmolops
from torch_geometric.data import Data, Dataset
from torch_geometric.utils import dense_to_sparse

from . import chem


def one_hot_encode(
    x: Any,
    kinds: list[Any],
    handle_unknown: str = "error",
) -> list[bool]:
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


def get_period_group(atom: Chem.Atom) -> list[bool]:
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


def get_atom_charges(mol: Chem.Mol) -> list[float]:
    charges = [atom.GetFormalCharge() for atom in mol.GetAtoms()]
    return charges


def get_metals(mol: Chem.Mol) -> list[bool]:
    mask = [atom.GetSymbol() in chem.METALS for atom in mol.GetAtoms()]
    return mask


def get_smarts_matches(mol: Chem.Mol, smarts: str) -> list[bool]:
    # Get the matching atom indices.
    pattern = Chem.MolFromSmarts(smarts)
    matches = {idx for match in mol.GetSubstructMatches(pattern) for idx in match}

    # Convert to a mask vector.
    mask = [idx in matches for idx in range(mol.GetNumAtoms())]
    return mask


def get_hydrophobes(mol: Chem.Mol) -> list[bool]:
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


def atom_to_features(atom: Chem.Atom) -> list[bool]:
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
    fragmentation: bool = False,
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
        fragment_map: (num_atoms,), long (optional)
        fragment_smiles: List[str] (optional)
        residue_map: (num_atoms,), long (optional)
        residue_names: List[str] (optional)
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

    # BRICS fragmentation for ligands
    if fragmentation:
        try:
            frags = []
            broken_mol = BRICS.BreakBRICSBonds(mol)
            frag_mols = list(Chem.GetMolFrags(broken_mol, asMols=True, frags=frags))

            # Get atom to fragment mapping
            if frags and len(frags) > 0:
                atom_frag_indices = frags[: mol.GetNumAtoms()]
                data.fragment_map = torch.tensor(atom_frag_indices, dtype=torch.long)

                # Store fragment SMILES for explanation
                fragment_smiles = []
                for frag_mol in frag_mols:
                    try:
                        smiles = Chem.MolToSmiles(frag_mol)
                        fragment_smiles.append(smiles)
                    except:
                        fragment_smiles.append("Unknown")
                data.fragment_smiles = fragment_smiles
            else:
                # If fragmentation fails, treat entire molecule as one fragment
                data.fragment_map = torch.zeros(mol.GetNumAtoms(), dtype=torch.long)
                data.fragment_smiles = [Chem.MolToSmiles(mol)]
        except:
            # If BRICS fails, treat entire molecule as one fragment
            data.fragment_map = torch.zeros(mol.GetNumAtoms(), dtype=torch.long)
            try:
                data.fragment_smiles = [Chem.MolToSmiles(mol)]
            except:
                data.fragment_smiles = ["Unknown"]

    # Add residue mapping for proteins
    if not fragmentation and mol.GetNumAtoms() > 0:
        # Check if this is a protein by looking for PDB residue info
        has_residue_info = False
        residue_map = []
        residue_names = set()
        residue_name_list = []

        for atom in mol.GetAtoms():
            res_info = atom.GetPDBResidueInfo()
            if res_info:
                has_residue_info = True
                res_name = res_info.GetResidueName()
                res_num = res_info.GetResidueNumber()
                res_chain = res_info.GetChainId()
                res_id = f"{res_name}{res_num}{res_chain}"

                if res_id not in residue_names:
                    residue_names.add(res_id)
                    residue_name_list.append(res_id)

                residue_map.append(
                    residue_name_list.index(res_id) + 1
                )  # +1 so 0 means no residue
            else:
                residue_map.append(0)  # 0 means no residue (ligand atoms)

        if has_residue_info:
            data.residue_map = torch.tensor(residue_map, dtype=torch.long)
            data.residue_names = residue_name_list

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
    label: float | None = None,
    key: str | None = None,
    conv_range: tuple[float, float] = None,
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
        fragment_map: (num_ligand_atoms,), long
            Fragment assignment for ligand atoms
        residue_map: (num_all_atoms,), long
            Residue assignment (0 for ligand, >0 for protein residues)
    """
    ligand = mol_to_data(
        mol_ligand, remove_hydrogens, pos_noise_std, pos_noise_max, fragmentation=True
    )
    target = mol_to_data(
        mol_target, remove_hydrogens, pos_noise_std, pos_noise_max, fragmentation=False
    )
    data = Data()

    if remove_hydrogens:
        mol_ligand = Chem.RemoveAllHs(mol_ligand)
        mol_target = Chem.RemoveAllHs(mol_target)

    # Combine the values - get all unique keys from both ligand and target
    all_keys = set(ligand.keys()) | set(target.keys())

    for attr in all_keys:
        # Handle ligand-only attributes
        if attr in ("fragment_map", "fragment_smiles"):
            if attr in ligand.keys():
                data[attr] = ligand[attr]
            continue

        # Handle residue-specific attributes from target
        if attr == "residue_map":
            if attr in target.keys():
                # Ligand atoms get residue ID 0 (no residue)
                ligand_residue_map = torch.zeros(ligand.num_nodes, dtype=torch.long)
                target_residue_map = target[attr]
                # Combine them
                value = torch.cat((ligand_residue_map, target_residue_map), 0)
                data[attr] = value
            continue

        if attr == "residue_names":
            if attr in target.keys():
                # Store residue names from target only
                data[attr] = target[attr]
            continue

        # Handle attributes present in both
        if attr in ligand.keys() and attr in target.keys():
            ligand_value = ligand[attr]
            target_value = target[attr]

            # Shift atom indices for some attributes.
            if attr in ("edge_index",):
                target_value = target_value + ligand.num_nodes

            # Dimension to concatenate over.
            cat_dim = ligand.__cat_dim__(attr, None)
            value = torch.cat((ligand_value, target_value), cat_dim)
            data[attr] = value
        elif attr in ligand.keys():
            # Only in ligand
            data[attr] = ligand[attr]
        elif attr in target.keys():
            # Only in target (shouldn't reach here due to special cases above)
            data[attr] = target[attr]

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
        keys: list[str],
        data_dir: str | None = None,
        id_to_y: dict[str, float] | None = None,
        conv_range: tuple[float, float] | None = None,
        processed_data_dir: str | None = None,
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
                try:
                    data = torch.load(f)
                except Exception as _:
                    data = torch.load(f, weights_only=False)

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
                except ValueError:
                    mol_ligand, mol_target = data.mol_ligand, data.mol_target

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
