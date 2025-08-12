import torch
from omegaconf import DictConfig
from torch.nn import Parameter, ReLU, Sigmoid
from torch_geometric.data import Batch
from torch_geometric.nn import Linear, Sequential
from torch_scatter import scatter

from . import physics
from .pignet import PIGNet


class PIGNetMorse(PIGNet):
    def __init__(
        self,
        config: DictConfig,
        in_features: int = -1,
        **kwargs,
    ):
        super().__init__(config=config)
        self.reset_log()
        self.config = config
        dim_gnn = config.model.dim_gnn
        dim_mlp = config.model.dim_mlp

        self.embed = Linear(in_features, dim_gnn, bias=False)

        self.nn_vdw_epsilon = Sequential(
            "x",
            [
                (Linear(dim_gnn * 2, dim_mlp), "x -> x"),
                ReLU(),
                Linear(dim_mlp, 1),
                Sigmoid(),
            ],
        )
        self.nn_vdw_width = Sequential(
            "x",
            [
                (Linear(dim_gnn * 2, dim_mlp), "x -> x"),
                ReLU(),
                Linear(dim_mlp, 1),
                Sigmoid(),
            ],
        )
        self.nn_vdw_radius = Sequential(
            "x",
            [
                (Linear(dim_gnn * 2, dim_mlp), "x -> x"),
                ReLU(),
                Linear(dim_mlp, 1),
                ReLU(),
            ],
        )

        self.hbond_coeff = Parameter(torch.tensor([0.714]))
        self.metal_ligand_coeff = Parameter(torch.tensor([1.0]))
        self.hydrophobic_coeff = Parameter(torch.tensor([0.216]))
        self.rotor_coeff = Parameter(torch.tensor([0.102]))
        self.ionic_coeff = Parameter(torch.tensor([1.0]))  # NOT USED

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
        energies_pairs = torch.zeros(5, D.numel()).to(self.device)

        # vdW energy minima (well depths): (pairs,)
        vdw_epsilon = self.nn_vdw_epsilon(x_cat).squeeze(-1)

        # Scale the minima as done in AutoDock Vina.
        vdw_epsilon = (
            vdw_epsilon * (cfg.vdw_epsilon_scale[1] - cfg.vdw_epsilon_scale[0])
            + cfg.vdw_epsilon_scale[0]
        )

        vdw_width = self.nn_vdw_width(x_cat).squeeze(-1)
        vdw_width = (
            vdw_width * (cfg.vdw_width_scale[1] - cfg.vdw_width_scale[0])
            + cfg.vdw_width_scale[0]
        )
        energies_pairs[0] = physics.morse_potential(
            D,
            R,
            vdw_epsilon,
            vdw_width,
            cfg.short_range_A,
        )

        minima_hbond = -(self.hbond_coeff**2)
        minima_metal_ligand = -(self.metal_ligand_coeff**2)
        minima_hydrophobic = -(self.hydrophobic_coeff**2)
        energies_pairs[1] = physics.linear_potential(
            D, R, minima_hbond, *cfg.hydrogen_bond_cutoffs
        )
        energies_pairs[2] = physics.linear_potential(
            D, R, minima_metal_ligand, *cfg.metal_ligand_cutoffs
        )
        energies_pairs[3] = physics.linear_potential(
            D, R, minima_hydrophobic, *cfg.hydrophobic_cutoffs
        )

        # Interaction masks according to atom types: (energy_types, pairs)
        masks = physics.interaction_masks(
            sample.is_metal,
            sample.is_h_donor,
            sample.is_h_acceptor,
            sample.is_hydrophobic,
            edge_index_i,
            True,
        )

        # ionic interaction
        energies_pairs[4] = torch.zeros_like(energies_pairs[4])
        if cfg.get("include_ionic", False):
            # Note the sign of `minima_ionic`
            minima_ionic = self.ionic_coeff**2 * (
                sample.atom_charges[edge_index_i[0]]
                * sample.atom_charges[edge_index_i[1]]
            )
            energies_pairs[4] = physics.linear_potential(
                D, R, minima_ionic, *cfg.ionic_cutoffs
            )

        energies_pairs = energies_pairs * masks
        # Per-graph sum -> (energy_types, batch)
        energies = scatter(energies_pairs, sample.batch[edge_index_i[0]])
        # Reshape -> (batch, energy_types)
        energies = energies.t().contiguous()

        # Rotor penalty
        if cfg.rotor_penalty:
            penalty = 1 + self.rotor_coeff**2 * sample.rotor
            # -> (batch, 1)
            energies = energies / penalty

        return energies, dvdw_radii
