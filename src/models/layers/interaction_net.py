import torch
import torch.nn.functional as F
from torch.nn import GRUCell
from torch_geometric.nn import Linear, MessagePassing
from torch_geometric.utils import add_self_loops


class InteractionNet(MessagePassing):
    def __init__(
        self,
        node_features: int,
        add_self_loops: bool = False,
        aggr: str = "max",
        **kwargs
    ):
        super().__init__(**kwargs)
        self.W1 = Linear(node_features, node_features)
        self.W2 = Linear(node_features, node_features)
        # GRUCell(input_size, hidden_size) -> hidden_size
        self.rnn = GRUCell(node_features, node_features)
        self.add_self_loops = add_self_loops
        self.aggr = aggr

    def forward(self, x, edge_index):
        # Need to pass num_nodes to handle isolated nodes (in proteins).
        num_nodes = x.size(0)

        if self.add_self_loops:
            edge_index, _ = add_self_loops(edge_index, num_nodes=num_nodes)

        x_updated = self.propagate(edge_index, x=x, size=(num_nodes, num_nodes))
        return x_updated

    def message(self, x_j):
        return self.W2(x_j)

    def update(self, inputs, x):
        x_prime = F.relu(self.W1(x) + inputs)
        return self.rnn(x_prime, x)

    def compare(self, x, sample):
        """\
        Temporary function to compare the `forward` implementation.
        """

        def get_neighbors(sample, idx: int, order: int):
            edges = sample.edge_index_c
            if self.add_self_loops:
                edges, _ = add_self_loops(edges, num_nodes=sample.x.size(0))

            ligand_size = sample.is_ligand.sum().item()

            # If `idx` is of ligand,
            if order == 0:
                srcs = edges[0, edges[1] == idx]
                srcs = srcs - ligand_size
            # If `idx` is of target,
            elif order == 1:
                srcs = edges[0, edges[1] == idx + ligand_size]
            return srcs

        x1 = x[sample.is_ligand]
        x2 = x[~sample.is_ligand]

        # ligand <- protein
        M = self.W2(x2)
        A = torch.zeros(x1.size(0), M.size(1))
        for i in range(x1.size(0)):
            srcs = get_neighbors(sample, i, 0)
            # If isolated,
            if not srcs.numel():
                continue
            A[i] = torch.max(M[srcs], 0).values
        x_prime = F.relu(self.W1(x1) + A)
        x1_updated = self.rnn(x_prime, x1)

        # protein <- ligand
        M = self.W2(x1)
        A = torch.zeros(x2.size(0), M.size(1))
        for i in range(x2.size(0)):
            srcs = get_neighbors(sample, i, 1)
            # If isolated,
            if not srcs.numel():
                continue
            A[i] = torch.max(M[srcs], 0).values
        x_prime = F.relu(self.W1(x2) + A)
        x2_updated = self.rnn(x_prime, x2)

        return torch.cat((x1_updated, x2_updated), 0)
