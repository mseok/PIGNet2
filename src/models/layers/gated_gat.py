import torch
import torch.nn.functional as F
from torch.nn import Parameter
from torch.nn.init import xavier_uniform_
from torch_geometric.nn import Linear, MessagePassing
from torch_geometric.utils import add_self_loops, softmax


class GatedGAT(MessagePassing):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        add_self_loops: bool = True,
        aggr: str = "sum",
        **kwargs
    ):
        super().__init__(**kwargs)
        self.W1 = Linear(in_features, out_features)
        self.W2 = Parameter(torch.zeros(out_features, out_features))
        if in_features == -1:
            self.gate = Linear(-1, 1)
        else:
            self.gate = Linear(in_features + out_features, 1)
        self.add_self_loops = add_self_loops
        self.aggr = aggr

    def forward(self, x, edge_index):
        # Need to pass num_nodes to handle isolated nodes (in proteins).
        num_nodes = x.size(0)

        if self.add_self_loops:
            edge_index, _ = add_self_loops(edge_index, num_nodes=num_nodes)

        # To save computation.
        Wx = self.W1(x)

        x_updated = self.propagate(edge_index, x=x, Wx=Wx, size=(num_nodes, num_nodes))
        return x_updated

    def message(self, Wx_i, Wx_j, index, size_i):
        """\
        Args:
            x_i: x[edge_index[1]] (target) (edges, features)
            x_j: x[edge_index[0]] (source) (edges, features)
            index: edge_index[1]  (target) (edges,)
            size_i: size[1]. Same as `size_j` (size[0]) in our case.
        """
        E = torch.einsum("ei,ij,ej->e", Wx_i, self.W2, Wx_j)
        # -> (edges,)
        # Symmetrize
        E_T = torch.einsum("ei,ij,ej->e", Wx_j, self.W2, Wx_i)
        E = E + E_T
        # Normalized attention coefficients
        A = softmax(E, index, dim=0, num_nodes=size_i)
        return torch.einsum("e,ef->ef", A, Wx_j)

    def update(self, inputs, x):
        x_prime = F.relu(inputs)
        # -> (atoms, out_features)
        z = torch.sigmoid(self.gate(torch.cat([x, x_prime], -1)))
        # -> (atoms, 1)
        return z * x + (1 - z) * x_prime

    def compare(self, x, edge_index):
        """\
        Temporary function to compare the `forward` implementation.
        """

        def get_neighbors(edge_index, idx: int):
            return edge_index[0, edge_index[1] == idx]

        if self.add_self_loops:
            edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))

        E = torch.einsum("ai,ij,bj->ab", self.W1(x), self.W2, self.W1(x))
        E = E + E.t()
        for i in range(x.size(0)):
            srcs = get_neighbors(edge_index, i)
            tmp = torch.softmax(E[i, srcs], 0)
            for j, val in zip(srcs, tmp):
                E[i, j] = val
            for j in range(x.size(0)):
                if j not in srcs:
                    E[i, j] = 0.0
        A = E

        x_prime = F.relu(A @ x)
        z = torch.sigmoid(self.gate(torch.cat([x, x_prime], -1)))
        # -> (atoms, 1)
        return z * x + (1 - z) * x_prime
