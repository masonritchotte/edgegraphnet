# edge_attention_layer.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_scatter import scatter
from torch_geometric.utils import softmax
from torch_sparse import SparseTensor

class EdgeAttentionLayer(nn.Module):
    """
    Edge-level attention per NENN:
      - Node-based neighbors of an edge (its two endpoints):  β^n via q_n^T [We e_i || Wn x_j]
      - Edge-based neighbors of an edge (edges sharing a node): β^e via q_e^T [We e_i || We e_k]
      - Each branch aggregates with MEAN of weighted neighbors, then ReLU, and we concat -> [E, 2H]
    """
    def __init__(self, in_node_dim, in_edge_dim, out_dim, negative_slope=0.2):
        super().__init__()
        self.out_dim = out_dim
        # Linear projections (We, Wn)
        self.lin_node = nn.Linear(in_node_dim, out_dim, bias=False)  # Wn
        self.lin_edge = nn.Linear(in_edge_dim, out_dim, bias=False)  # We
        # Attention score MLPs implementing q_n^T [We e_i || Wn x_j] and q_e^T [We e_i || We e_k]
        self.att_n = nn.Linear(2 * out_dim, 1, bias=False)  # for β^n
        self.att_e = nn.Linear(2 * out_dim, 1, bias=False)  # for β^e
        self.leaky_relu = nn.LeakyReLU(negative_slope)

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.lin_node.weight)
        nn.init.xavier_uniform_(self.lin_edge.weight)
        nn.init.xavier_uniform_(self.att_n.weight)
        nn.init.xavier_uniform_(self.att_e.weight)

    @torch.no_grad()
    def _build_edge_edge_index_fast(self, edge_index: torch.Tensor, num_nodes: int):
        """
        Vectorized construction via incidence B: A_e = B^T B (E x E) with diag cleared.
        Returns [2, E2E] (row=e_src, col=e_nbr).
        """
        device = edge_index.device
        E = edge_index.size(1)
        edge_ids = torch.arange(E, device=device)
        row = torch.cat([edge_index[0], edge_index[1]], dim=0)   # node ids
        col = torch.cat([edge_ids,        edge_ids       ], dim=0)   # edge ids
        B = SparseTensor(row=row, col=col, sparse_sizes=(num_nodes, E)).coalesce()
        A_e = B.t() @ B
        A_e = A_e.set_diag(0)
        r, c, _ = A_e.coo()
        return torch.stack([r, c], dim=0)

    def forward(self, x_node, edge_attr, edge_index, edge_edge_index=None):
        """
        x_node: [N, F_n]
        edge_attr: [E, F_e]
        edge_index: [2, E]
        edge_edge_index: optional [2, E2E] precomputed/tiled for the batch
        returns: [E, 2*out_dim]
        """
        device = x_node.device
        N = int(edge_index.max()) + 1
        E = edge_attr.size(0)

        # Linear projections
        h = self.lin_node(x_node)    # [N, H] = Wn x
        g = self.lin_edge(edge_attr) # [E, H] = We e

        # ---------- β^n over node-based neighbors (endpoints of each edge) ----------
        u, v = edge_index[0], edge_index[1]                      # [E], [E]
        e_rep = torch.cat([torch.arange(E, device=device),       # [2E] -> which edge each row belongs to
                           torch.arange(E, device=device)], 0)
        n_for_e = torch.cat([u, v], dim=0)                       # [2E] -> node index per row

        # Scores: q_n^T [We e_i || Wn x_j]
        cat_ne = torch.cat([g[e_rep], h[n_for_e]], dim=-1)       # [2E, 2H]
        score_ne = self.leaky_relu(self.att_n(cat_ne)).squeeze(-1)  # [2E]
        alpha_ne = softmax(score_ne, e_rep, num_nodes=E)         # normalize across the two endpoints per edge

        # Aggregate: e_Ni = σ( MEAN_j alpha_ne * (Wn x_j) )
        e_from_nodes = scatter(alpha_ne.unsqueeze(-1) * h[n_for_e],
                               e_rep, dim=0, dim_size=E, reduce='mean')  # [E, H]
        e_from_nodes = F.relu(e_from_nodes)

        # ---------- β^e over edge-based neighbors (edges sharing a node) ----------
        if edge_edge_index is None:
            edge_edge_index = self._build_edge_edge_index_fast(edge_index, N).to(device)

        e_src, e_nbr = edge_edge_index  # [E2E]
        if e_src.numel() == 0:
            e_from_edges = g
        else:
            # Scores: q_e^T [We e_i || We e_k]
            cat_ee = torch.cat([g[e_src], g[e_nbr]], dim=-1)     # [E2E, 2H]
            score_ee = self.leaky_relu(self.att_e(cat_ee)).squeeze(-1)  # [E2E]
            alpha_ee = softmax(score_ee, e_src, num_nodes=E)
            # Aggregate: e_Ei = σ( MEAN_k alpha_ee * (We e_k) )
            e_from_edges = scatter(alpha_ee.unsqueeze(-1) * g[e_nbr],
                                   e_src, dim=0, dim_size=E, reduce='mean')  # [E, H]
            e_from_edges = F.relu(e_from_edges)

        # Final concat per Eq. (13): e^{l+1}_i = CONCAT(e_Ni, e_Ei)
        return torch.cat([e_from_nodes, e_from_edges], dim=-1)   # [E, 2H]
