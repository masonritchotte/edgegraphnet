
import torch
import torch.nn as nn
from torch_scatter import scatter
from torch_geometric.utils import add_self_loops, softmax

class NodeAttentionLayer(nn.Module):
    """
    Node-level attention layer for NENN.
    Implements:
      (a) includes self in node-based neighbors via self-loops (ONLY for node->node attention);
      (b) applies a post-aggregation nonlinearity σ to each branch before concatenation.
    """
    def __init__(self, in_node_dim: int, in_edge_dim: int, out_dim: int, negative_slope: float = 0.05):
        super().__init__()
        # Projections
        self.node_linear = nn.Linear(in_node_dim, out_dim, bias=False)  # W_n
        self.edge_linear = nn.Linear(in_edge_dim, out_dim, bias=False)  # W_e

        # Attention scorers
        self.node_attention = nn.Linear(2 * out_dim, 1, bias=False)     # a_n^T [h_i || h_j]
        self.edge_attention = nn.Linear(2 * out_dim, 1, bias=False)     # a_e^T [h_i || g_k]

        self.leaky_relu = nn.LeakyReLU(negative_slope)
        # (b) Post-aggregation σ for each branch
        self.post_node = nn.LeakyReLU(negative_slope)
        self.post_edge = nn.LeakyReLU(negative_slope)

    def forward(self, x, edge_attr, edge_index):
        """
        Args:
            x:           [N, F_n] node features
            edge_attr:   [E, F_e] edge features
            edge_index:  [2, E]   COO edge index (can be directed; undirected should be stored both ways)
        Returns:
            out:         [N, 2 * out_dim] concatenated node embedding from node- and edge-neighbor streams
        """
        N = x.size(0)
        E = edge_index.size(1)
        device = x.device

        # Linear projections
        h = self.node_linear(x)             # [N, d]
        g = self.edge_linear(edge_attr)     # [E, d]

        # -------- Node->Node attention (with self-loops) --------
        # (a) add self-loops ONLY for node-neighbor stream
        edge_index_n, _ = add_self_loops(edge_index, num_nodes=N)  # [2, E + N]
        i_idx, j_idx = edge_index_n  # source/target indices for attention group-by center node i

        # Attention logits and normalized coefficients across j in N(i)
        e_n = self.leaky_relu(self.node_attention(torch.cat([h[i_idx], h[j_idx]], dim=-1))).squeeze(-1)  # [E+N]
        alpha_n = softmax(e_n, i_idx)  # softmax per i over its neighbors (incl. self)

        # Weighted MEAN aggregation into center nodes
        # (scatter with reduce='mean' reproduces MEAN in the paper)
        x_from_nodes = scatter(alpha_n.unsqueeze(-1) * h[j_idx], i_idx, dim=0, dim_size=N, reduce='mean')  # [N, d]

        # -------- Node->Edge attention over incident edges (NO self-loops here) --------
        # Build mapping "node i -> incident edges k"
        row, col = edge_index  # [E], [E]
        # repeat each edge twice to associate with both endpoints (i=row and i=col)
        i_idx_e = torch.cat([row, col], dim=0)                           # [2E]
        k_idx_e = torch.cat([torch.arange(E, device=device)] * 2, dim=0) # [2E]

        e_e = self.leaky_relu(self.edge_attention(torch.cat([h[i_idx_e], g[k_idx_e]], dim=-1))).squeeze(-1)  # [2E]
        alpha_e = softmax(e_e, i_idx_e)  # normalize across incident edges of each node
        x_from_edges = scatter(alpha_e.unsqueeze(-1) * g[k_idx_e], i_idx_e, dim=0, dim_size=N, reduce='mean')  # [N, d]

        # (b) Post-aggregation activations before concat
        x_from_nodes = self.post_node(x_from_nodes)
        x_from_edges = self.post_edge(x_from_edges)

        return torch.cat([x_from_nodes, x_from_edges], dim=-1)  # [N, 2d]
