from collections import defaultdict

import torch
import torch.nn.functional as F
from omegaconf import DictConfig
from torch.nn import Dropout, Module, ModuleList, Parameter, ReLU, Sigmoid, Tanh
from torch.nn.parameter import UninitializedParameter
from torch.optim import Adam
from torch_geometric.data import Batch
from torch_geometric.nn import Linear, Sequential
from torch_geometric.utils import scatter

from . import physics
from .layers import GatedGAT, InteractionNet


class PIGNet(Module):
    def __init__(
        self,
        config: DictConfig,
        in_features: int = -1,
        **kwargs,
    ):
        super().__init__()
        self.reset_log()
        self.config = config
        n_gnn = config.model.n_gnn
        dim_gnn = config.model.dim_gnn
        dim_mlp = config.model.dim_mlp
        dropout_rate = config.run.dropout_rate

        self.embed = Linear(in_features, dim_gnn, bias=False)

        self.intraconv = ModuleList()
        for _ in range(n_gnn):
            self.intraconv.append(
                Sequential(
                    "x, edge_index",
                    [
                        (GatedGAT(dim_gnn, dim_gnn), "x, edge_index -> x"),
                        (Dropout(dropout_rate), "x -> x"),
                    ],
                )
            )

        self.interconv = ModuleList()
        if config.model.interconv:
            for _ in range(n_gnn):
                self.interconv.append(
                    Sequential(
                        "x, edge_index",
                        [
                            (InteractionNet(dim_gnn), "x, edge_index -> x"),
                            (Dropout(dropout_rate), "x -> x"),
                        ],
                    )
                )

        self.nn_vdw_epsilon = Sequential(
            "x",
            [
                (Linear(dim_gnn * 2, dim_mlp), "x -> x"),
                ReLU(),
                Linear(dim_mlp, 1),
                Sigmoid(),
            ],
        )

        self.nn_dvdw = Sequential(
            "x",
            [
                (Linear(dim_gnn * 2, dim_mlp), "x -> x"),
                ReLU(),
                Linear(dim_mlp, 1),
                Tanh(),
            ],
        )

        self.hbond_coeff = Parameter(torch.tensor([1.0]))
        self.hydrophobic_coeff = Parameter(torch.tensor([0.5]))
        self.rotor_coeff = Parameter(torch.tensor([0.5]))
        if config.model.get("include_ionic", False):
            self.ionic_coeff = Parameter(torch.tensor([1.0]))

    @property
    def size(self) -> tuple[int, int]:
        """Get the number of all learnable parameters.

        Returns: (num_parameters, num_uninitialized_parameters)
        """
        num_params = 0
        num_uninitialized = 0

        for param in self.parameters():
            if isinstance(param, UninitializedParameter):
                num_uninitialized += 1
            elif param.requires_grad:
                num_params += param.numel()

        return num_params, num_uninitialized

    @property
    def in_features(self) -> int:
        """Get the number of input features."""
        try:
            return self.embed.in_channels
        except AttributeError:
            return self.embed.in_features

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device

    def conv(self, x, edge_index_1, edge_index_2):
        for conv in self.intraconv:
            x = conv(x, edge_index_1)

        for conv in self.interconv:
            x = conv(x, edge_index_2)
        return x

    def forward(self, sample: Batch):
        cfg = self.config.model

        # Initial embedding
        x = self.embed(sample.x)

        # Graph convolutions
        x = self.conv(x, sample.edge_index, sample.edge_index_c)

        # Ligand-to-target uni-directional edges
        # to compute pairwise interactions: (2, pairs)
        edge_index_i = physics.interaction_edges(sample.is_ligand, sample.batch)

        # Pairwise distances: (pairs,)
        D = physics.distances(sample.pos, edge_index_i)

        # Limit the interaction distance.
        _mask = (cfg.interaction_range[0] <= D) & (D <= cfg.interaction_range[1])
        edge_index_i = edge_index_i[:, _mask]
        D = D[_mask]

        # Pairwise node features: (pairs, 2*features)
        x_cat = torch.cat((x[edge_index_i[0]], x[edge_index_i[1]]), -1)

        # Pairwise vdW-radii deviations: (pairs,)
        dvdw_radii = self.nn_dvdw(x_cat).view(-1)
        dvdw_radii = dvdw_radii * cfg.dev_vdw_radii_coeff

        # Pairwise vdW radii: (pairs,)
        R = (
            sample.vdw_radii[edge_index_i[0]]
            + sample.vdw_radii[edge_index_i[1]]
            + dvdw_radii
        )

        # Prepare a pair-energies contrainer: (energy_types, pairs)
        if cfg.get("include_ionic", False):
            energies_pairs = torch.empty(5, D.numel()).to(self.device)
        else:
            energies_pairs = torch.empty(4, D.numel()).to(self.device)

        # vdW energy minima (well depths): (pairs,)
        vdw_epsilon = self.nn_vdw_epsilon(x_cat).view(-1)
        # Scale the minima as done in AutoDock Vina.
        vdw_epsilon = (
            vdw_epsilon * (cfg.vdw_epsilon_scale[1] - cfg.vdw_epsilon_scale[0])
            + cfg.vdw_epsilon_scale[0]
        )
        # vdW interaction
        energies_pairs[0] = physics.lennard_jones_potential(
            D, R, vdw_epsilon, cfg.vdw_N_short, cfg.vdw_N_long
        )

        # Hydrogen-bond, metal-ligand, hydrophobic interactions
        minima_hbond = -(self.hbond_coeff**2)
        minima_hydrophobic = -(self.hydrophobic_coeff**2)
        energies_pairs[1] = physics.linear_potential(
            D, R, minima_hbond, *cfg.hydrogen_bond_cutoffs
        )
        energies_pairs[2] = physics.linear_potential(
            D, R, minima_hbond, *cfg.metal_ligand_cutoffs
        )
        energies_pairs[3] = physics.linear_potential(
            D, R, minima_hydrophobic, *cfg.hydrophobic_cutoffs
        )
        # Include the ionic interaction if required.
        if cfg.get("include_ionic", False):
            # Note the sign of `minima_ionic`
            minima_ionic = self.ionic_coeff**2 * (
                sample.atom_charges[edge_index_i[0]]
                * sample.atom_charges[edge_index_i[1]]
            )
            energies_pairs[4] = physics.linear_potential(
                D, R, minima_ionic, *cfg.ionic_cutoffs
            )

        # Interaction masks according to atom types: (energy_types, pairs)
        masks = physics.interaction_masks(
            sample.is_metal,
            sample.is_h_donor,
            sample.is_h_acceptor,
            sample.is_hydrophobic,
            edge_index_i,
            include_ionic=cfg.get("include_ionic", False),
        )
        energies_pairs = energies_pairs * masks

        # Per-graph sum -> (energy_types, batch)
        energies = scatter(energies_pairs, sample.batch[edge_index_i[0]], dim=-1)
        # Reshape -> (batch, energy_types)
        energies = energies.t().contiguous()

        # Rotor penalty
        if cfg.rotor_penalty:
            penalty = 1 + self.rotor_coeff**2 * sample.rotor
            # -> (batch, 1)
            energies = energies / penalty

        return energies, dvdw_radii

    def loss_dvdw(self, dvdw_radii: torch.Tensor):
        loss = dvdw_radii.pow(2).mean()
        return loss

    def loss_regression(
        self,
        energies: torch.Tensor,
        true: torch.Tensor,
    ):
        return F.mse_loss(energies.sum(-1, True), true)

    def loss_augment(
        self,
        energies: torch.Tensor,
        true: torch.Tensor,
        min: float | None = None,
        max: float | None = None,
    ):
        """Loss functions for docking, random & cross screening.

        Args:
            sample
            task: 'docking' | 'random' | 'cross'
        """
        loss_energy = true - energies.sum(-1, True)
        loss_energy = loss_energy.clamp(min, max)
        loss_energy = loss_energy.mean()
        return loss_energy

    def training_step(self, batch: dict[str, Batch]):
        loss_total = torch.tensor(0.0, device=self.device)

        for task, sample in batch.items():
            task_config = self.config.data[task]

            energies, dvdw_radii = self(sample)
            loss_dvdw = self.loss_dvdw(dvdw_radii)
            if task_config.objective == "regression":
                loss_energy = self.loss_regression(energies, sample.y)
            elif task_config.objective == "augment":
                loss_energy = self.loss_augment(
                    energies, sample.y, *task_config.loss_range
                )
            else:
                raise NotImplementedError(
                    "Current loss functions only support regression and augment."
                )

            loss_total += loss_energy * task_config.loss_ratio
            loss_total += loss_dvdw * self.config.run.loss_dvdw_ratio

            # Update log
            self.losses["energy"][task].append(loss_energy.item())
            self.losses["dvdw"][task].append(loss_dvdw.item())
            for key, pred, true in zip(sample.key, energies, sample.y):
                self.predictions[task][key] = pred.tolist()
                self.labels[task][key] = true.item()

        return loss_total

    def validation_step(self, batch: dict[str, Batch]):
        return self.training_step(batch)

    def test_step(self, batch: Batch):
        sample = batch
        task = next(iter(self.config.data))

        energies, dvdw_radii = self(sample)
        loss_energy = self.loss_regression(energies, sample.y)
        loss_dvdw = self.loss_dvdw(dvdw_radii)

        # Update log
        self.losses["energy"][task].append(loss_energy.item())
        self.losses["dvdw"][task].append(loss_dvdw.item())
        for key, pred, true in zip(sample.key, energies, sample.y):
            self.predictions[task][key] = pred.tolist()
            self.labels[task][key] = true.item()

    def predict_step(self, batch: Batch):
        sample = batch
        task = next(iter(self.config.data))
        energies, dvdw_radii = self(sample)
        for key, pred in zip(sample.key, energies):
            self.predictions[task][key] = pred.tolist()

    def configure_optimizers(self):
        return Adam(
            self.parameters(),
            lr=self.config.run.lr,
            weight_decay=self.config.run.weight_decay,
        )

    def predict_atom_interactions(self, sample: Batch):
        """Predict atom-atom interaction energies.

        Returns:
            Dict with keys for each molecule containing:
            - 'ligand_indices': ligand atom indices
            - 'pocket_indices': pocket atom indices
            - 'energies': (n_ligand, n_pocket, n_energy_types) interaction matrix
        """
        # Run forward pass to get molecular energies
        mol_energies, _ = self(sample)

        # Get interaction network outputs if available
        results = {}

        # Handle both single and batched predictions
        if hasattr(sample, "key"):
            # For batched data, key is a list
            if isinstance(sample.key, list):
                keys = sample.key
            else:
                # For single data, wrap in list
                keys = [sample.key]
        else:
            # Fallback if no key attribute
            keys = [
                f"mol_{i}"
                for i in range(
                    sample.num_graphs if hasattr(sample, "num_graphs") else 1
                )
            ]

        for i, key in enumerate(keys):
            # Get molecule mask
            mol_mask = sample.batch == i
            mol_ligand_mask = mol_mask & sample.is_ligand
            mol_pocket_mask = mol_mask & ~sample.is_ligand

            # Get ligand and pocket atom indices
            ligand_indices = torch.arange(len(sample.batch), device=self.device)[
                mol_ligand_mask
            ]
            pocket_indices = torch.arange(len(sample.batch), device=self.device)[
                mol_pocket_mask
            ]

            n_ligand = ligand_indices.shape[0]
            n_pocket = pocket_indices.shape[0]

            # Create interaction matrix (simplified - would need actual edge energies)
            # For now, create a placeholder that will be refined
            interaction_matrix = torch.zeros(
                n_ligand, n_pocket, mol_energies.shape[1], device=self.device
            )

            # Get edges between ligand and pocket
            if hasattr(sample, "edge_index_c"):
                edge_index = sample.edge_index_c

                # Count intermolecular edges for normalization
                intermol_edges = 0
                for edge_idx in range(edge_index.shape[1]):
                    src, dst = edge_index[:, edge_idx]
                    if src in ligand_indices and dst in pocket_indices:
                        intermol_edges += 1

                # Find edges connecting ligand to pocket
                for edge_idx in range(edge_index.shape[1]):
                    src, dst = edge_index[:, edge_idx]

                    # Check if this edge connects ligand to pocket
                    if src in ligand_indices and dst in pocket_indices:
                        i_lig = (ligand_indices == src).nonzero().item()
                        i_pock = (pocket_indices == dst).nonzero().item()
                        # Distribute total energy across intermolecular edges
                        # Each edge gets a fraction of the total molecular energy
                        interaction_matrix[i_lig, i_pock, :] = mol_energies[i, :] / max(
                            intermol_edges, 1
                        )

            results[keys[i]] = {
                "ligand_indices": ligand_indices.cpu().numpy(),
                "pocket_indices": pocket_indices.cpu().numpy(),
                "energies": interaction_matrix.cpu().numpy(),
            }

        return results

    def predict_residue_fragment_interactions(self, sample: Batch):
        """Predict residue-fragment interaction energies.

        Returns:
            Dict with keys for each molecule containing:
            - 'fragment_energies': (n_fragments, n_residues, n_energy_types)
            - 'fragment_ids': fragment IDs
            - 'residue_ids': residue IDs
        """
        # First get atom-level interactions
        atom_interactions = self.predict_atom_interactions(sample)

        results = {}

        # Handle both single and batched predictions
        if hasattr(sample, "key"):
            # For batched data, key is a list
            if isinstance(sample.key, list):
                keys = sample.key
            else:
                # For single data, wrap in list
                keys = [sample.key]
        else:
            # Fallback if no key attribute
            keys = [
                f"mol_{i}"
                for i in range(
                    sample.num_graphs if hasattr(sample, "num_graphs") else 1
                )
            ]

        for i, key in enumerate(keys):
            # Skip if no residue information
            if not hasattr(sample, "residue_map") or sample.residue_map is None:
                results[key] = {
                    "fragment_energies": None,
                    "fragment_ids": None,
                    "residue_ids": None,
                    "error": "No residue information available",
                }
                continue

            # Get molecule mask
            mol_mask = sample.batch == i
            mol_ligand_mask = mol_mask & sample.is_ligand
            mol_pocket_mask = mol_mask & ~sample.is_ligand

            # Get fragment and residue maps for this molecule
            ligand_indices = torch.arange(len(sample.batch), device=self.device)[
                mol_ligand_mask
            ]
            pocket_indices = torch.arange(len(sample.batch), device=self.device)[
                mol_pocket_mask
            ]

            # Get fragment IDs for ligand atoms
            if hasattr(sample, "fragment_map") and sample.fragment_map is not None:
                fragment_ids = sample.fragment_map[ligand_indices]
                n_fragments = (
                    fragment_ids.max().item() + 1 if fragment_ids.numel() > 0 else 0
                )
            else:
                # Treat entire ligand as one fragment
                n_fragments = 1 if ligand_indices.numel() > 0 else 0
                fragment_ids = torch.zeros_like(ligand_indices)

            # Get residue IDs for pocket atoms
            residue_ids = sample.residue_map[pocket_indices]
            n_residues = residue_ids.max().item() + 1 if residue_ids.numel() > 0 else 0

            if n_fragments == 0 or n_residues == 0:
                results[key] = {
                    "fragment_energies": torch.zeros(
                        0, 0, atom_interactions[key]["energies"].shape[2]
                    ),
                    "fragment_ids": torch.tensor([]),
                    "residue_ids": torch.tensor([]),
                }
                continue

            # Get atom interaction matrix
            atom_energies = torch.tensor(
                atom_interactions[key]["energies"], device=self.device
            )

            # Aggregate to fragment-residue level
            n_energy_types = atom_energies.shape[2]
            fragment_residue_energies = torch.zeros(
                n_fragments, n_residues, n_energy_types, device=self.device
            )

            # Sum energies by fragment and residue
            for lig_idx, frag_id in enumerate(fragment_ids):
                for pock_idx, res_id in enumerate(residue_ids):
                    fragment_residue_energies[frag_id, res_id] += atom_energies[
                        lig_idx, pock_idx
                    ]

            results[key] = {
                "fragment_energies": fragment_residue_energies.cpu().numpy(),
                "fragment_ids": torch.arange(n_fragments).cpu().numpy(),
                "residue_ids": torch.arange(n_residues).cpu().numpy(),
            }

        return results

    def reset_log(self):
        """Reset logs. Intended to be called every epoch.

        Attributes:
            losses: Dict[str, Dict[str, List[float]]]
                losses[loss_type][task] -> loss_values
                where
                    loss_type: 'energy' | 'dvdw'
                    task: 'scoring' | 'docking' | 'random' | 'cross' | ...
                    loss_values: List[float] of shape (batches,)

            predictions: Dict[str, Dict[str, Tuple[float, ...]]]
                predictions[task][key] -> energies
                where
                    energies: List[float] of shape (4,)

            labels: Dict[str, Dict[str, float]]
                labels[task][key] -> energy (float)
        """
        self.losses = defaultdict(lambda: defaultdict(list))
        self.predictions = defaultdict(dict)
        self.labels = defaultdict(dict)
