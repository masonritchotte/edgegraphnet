# wire_up_nenn.py
import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import LabelEncoder
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
import torch.nn as nn
from torch_geometric.nn import global_mean_pool
from torch import autocast as torch_autocast
from torch.cuda.amp import GradScaler
import matplotlib.pyplot as plt
from dataclasses import dataclass, field

from node_attention_layer import NodeAttentionLayer
from edge_attention_layer import EdgeAttentionLayer

from time import perf_counter
from tqdm import tqdm

#----------------- Helpers ----------------
@torch.no_grad()
def precompute_edge_edge_index_single(edge_index: torch.Tensor, num_nodes: int):
    """
    Build edge->edge index once for a SINGLE graph (no batch) via B^T B.
    Returns [2, E2E_single] on same device as edge_index.
    """
    from torch_sparse import SparseTensor  # import inside to avoid hard dep at module import
    E = edge_index.size(1)
    edge_ids = torch.arange(E, device=edge_index.device)
    row = torch.cat([edge_index[0], edge_index[1]], dim=0)
    col = torch.cat([edge_ids,        edge_ids       ], dim=0)
    B = SparseTensor(row=row, col=col, sparse_sizes=(num_nodes, E)).coalesce()
    A_e = B.t() @ B
    A_e = A_e.set_diag(0)
    r, c, _ = A_e.coo()
    return torch.stack([r, c], dim=0)  # [2, E2E_single]

def tile_edge_edge_index(single_e2e: torch.Tensor, num_graphs: int, edges_per_graph: int, device=None):
    """
    Expand a single-graph edge->edge index to a batch of `num_graphs` identical graphs
    by adding per-graph edge offsets.
    """
    if device is None:
        device = single_e2e.device
    if num_graphs == 1:
        return single_e2e.to(device)
    E2E = single_e2e.size(1)
    offsets = (torch.arange(num_graphs, device=device) * edges_per_graph).repeat_interleave(E2E)
    return torch.stack([
        single_e2e[0].repeat(num_graphs) + offsets,
        single_e2e[1].repeat(num_graphs) + offsets
    ], dim=0)  # [2, num_graphs * E2E]

# --- class weights from a torch.utils.data.Subset of ExpressionGraphDataset ---
def class_weights_from_subset(subset, num_classes: int):
    """
    subset: torch.utils.data.Subset wrapping ExpressionGraphDataset
    returns: FloatTensor [C] with inverse-frequency weights (normalized)
    """
    import torch
    y = torch.tensor([subset.dataset.y[i].item() for i in subset.indices], dtype=torch.long)
    counts = torch.bincount(y, minlength=num_classes).float().clamp(min=1)
    w = 1.0 / counts
    w = w * (counts.sum() / w.numel())  # normalize so mean weight ~ 1
    return w

# --- evaluation on a loader (graph-level metrics) ---
@torch.no_grad()
def eval_on_loader(loader, model, device):
    from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
    import torch
    model.eval()
    ys, preds = [], []
    for batch in loader:
        batch = batch.to(device)
        logits = model(batch)
        preds.append(logits.argmax(-1).cpu())
        ys.append(batch.y.cpu())
    y = torch.cat(ys).numpy(); p = torch.cat(preds).numpy()
    rep = classification_report(y, p, digits=3, zero_division=0)
    cm  = confusion_matrix(y, p)
    return {
        "acc": accuracy_score(y, p),
        "macro_f1": f1_score(y, p, average="macro", zero_division=0),
        "report": rep,
        "cm": cm
    }

# --- plot confusion matrix ---
def plot_confusion_matrix(cm, out_path="confusion_matrix.png", title="Confusion Matrix"):
    import numpy as np, matplotlib.pyplot as plt
    plt.figure(figsize=(7,6))
    im = plt.imshow(cm, interpolation='nearest')
    plt.title(title); plt.colorbar(im, fraction=0.046, pad=0.04)
    plt.xlabel("Predicted"); plt.ylabel("True")
    # light tick labels for large C to keep it readable
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight"); plt.close()
    
def dimreduce(X, n_components=2, seed=42):
    """
    Try UMAP; if not installed, fall back to t-SNE.
    X: np.ndarray [N, D]
    returns: (Z [N, n_components], method_name)
    """
    import numpy as np
    try:
        import umap
        reducer = umap.UMAP(n_components=n_components, n_neighbors=15, min_dist=0.1, random_state=seed)
        return reducer.fit_transform(X), "UMAP"
    except Exception:
        from sklearn.manifold import TSNE
        reducer = TSNE(n_components=n_components, perplexity=30, init="pca", learning_rate="auto", random_state=seed)
        return reducer.fit_transform(X), "t-SNE"

@torch.no_grad()
def get_single_graph_embeddings(ds, idx, model, device):
    """
    Compute node/edge/graph embeddings for ONE sample graph (idx).
    """
    from torch_geometric.loader import DataLoader
    model.eval()
    data = ds[idx]
    loader = DataLoader([data], batch_size=1, shuffle=False)
    batch = next(iter(loader)).to(device)
    x2, e1, g = model.embed(batch)
    return x2.detach().cpu().numpy(), e1.detach().cpu().numpy(), g.detach().cpu().numpy()  # [N,2H], [E,2H], [1,2H]

def plot_before_after_embeddings(before_X, after_X, out_path, title, n_components=2, seed=42):
    import numpy as np, matplotlib.pyplot as plt
    Z, method = dimreduce(np.vstack([before_X, after_X]), n_components=n_components, seed=seed)
    n = before_X.shape[0]
    Zb, Za = Z[:n], Z[n:]

    if n_components == 2:
        plt.figure(figsize=(6,5))
        plt.scatter(Zb[:,0], Zb[:,1], s=12, alpha=0.6, label="Before")
        plt.scatter(Za[:,0], Za[:,1], s=12, alpha=0.6, label="After")
        plt.legend(); plt.title(f"{title} ({method})"); plt.tight_layout()
        plt.savefig(out_path, dpi=150); plt.close()
    elif n_components == 3:
        from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
        fig = plt.figure(figsize=(7,6))
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(Zb[:,0], Zb[:,1], Zb[:,2], s=8, alpha=0.6, label="Before")
        ax.scatter(Za[:,0], Za[:,1], Za[:,2], s=8, alpha=0.6, label="After")
        ax.legend(); ax.set_title(f"{title} ({method})")
        plt.tight_layout(); plt.savefig(out_path, dpi=150); plt.close()
    else:
        raise ValueError("n_components must be 2 or 3")

@torch.no_grad()
def collect_graph_embeddings(ds, edge_edge_index_single, edges_per_graph, model_ctor, device, batch_size=32):
    from torch_geometric.loader import DataLoader
    # Build a lightweight encoder model that only needs embed(); reuse trained weights
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False)
    model = model_ctor().to(device)
    model.load_state_dict(torch.load("best.ckpt")["model"] if False else model.state_dict())  # no-op; keep current model if you didn't save
    model.eval()
    graphs, labels = [], []
    for batch in loader:
        batch = batch.to(device)
        _, _, g = model.embed(batch)
        graphs.append(g.cpu())
        labels.append(batch.y.cpu())
    import torch
    G = torch.cat(graphs).numpy()
    y = torch.cat(labels).numpy()
    return G, y

# Quick graph-level map with current (trained) model
def graph_map(model, ds, device, n_components=2, out_path=None, seed=42):
    from torch_geometric.loader import DataLoader
    import torch, matplotlib.pyplot as plt, numpy as np
    model.eval()
    loader = DataLoader(ds, batch_size=32, shuffle=False)
    Gs, Ys = [], []
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            _, _, g = model.embed(batch)
            Gs.append(g.cpu()); Ys.append(batch.y.cpu())
    G = torch.cat(Gs).numpy(); y = torch.cat(Ys).numpy()

    Z, method = dimreduce(G, n_components=n_components, seed=seed)
    if n_components == 2:
        plt.figure(figsize=(6,5))
        plt.scatter(Z[:,0], Z[:,1], s=10, c=y, cmap="tab20")
        plt.title(f"Graph embeddings ({method})"); plt.tight_layout()
        plt.savefig(out_path or "graph_embeddings_2d.png", dpi=150); plt.close()
    elif n_components == 3:
        from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
        fig = plt.figure(figsize=(7,6))
        ax = fig.add_subplot(111, projection='3d')
        p = ax.scatter(Z[:,0], Z[:,1], Z[:,2], s=10, c=y, cmap="tab20")
        fig.colorbar(p, ax=ax, fraction=0.046, pad=0.04)
        ax.set_title(f"Graph embeddings ({method})"); plt.tight_layout()
        plt.savefig(out_path or "graph_embeddings_3d.png", dpi=150); plt.close()
    else:
        raise ValueError("n_components must be 2 or 3")

# ---------------- Dataset ----------------
class ExpressionGraphDataset(torch.utils.data.Dataset):
    def __init__(self, expr_df: pd.DataFrame, labels_s: pd.Series, edge_index, edge_attr):
        # align labels to expression rows (by index)
        labels_s = labels_s.loc[expr_df.index]
        self.encoder = LabelEncoder()
        self.y = torch.tensor(self.encoder.fit_transform(labels_s.values), dtype=torch.long)

        self.X = torch.tensor(expr_df.values, dtype=torch.float32)   # [S, N]
        self.edge_index = edge_index
        self.edge_attr = edge_attr
        self.num_samples, self.num_genes = self.X.shape

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        x = self.X[idx].unsqueeze(-1)    # [N, 1]
        y = self.y[idx]
        return Data(x=x, edge_index=self.edge_index, edge_attr=self.edge_attr, y=y)

# --------------- Edges -------------------
def build_edges_from_npy(adj_np: np.ndarray, undirected: bool = True):
    """
    adj_np: array of [E, 3] where each row is [src_idx, dst_idx, embedding(1536)]
    returns: edge_index [2,E], edge_attr [E,1536] (float32)
    """
    src = adj_np[:, 0].astype(np.int64)
    dst = adj_np[:, 1].astype(np.int64)
    eattr = adj_np[:, 2]

    if undirected:
        src = np.concatenate([src, dst], axis=0)
        dst = np.concatenate([dst, src[:len(dst)]], axis=0)  # src copy from before concat
        eattr = np.concatenate([eattr, eattr], axis=0)

    edge_index = torch.tensor(np.stack([src, dst], axis=0), dtype=torch.long)
    eattr = np.stack([np.asarray(v, dtype=np.float32) for v in eattr], axis=0)  # [E,1536]
    edge_attr = torch.tensor(eattr, dtype=torch.float32)
    return edge_index, edge_attr

# -------------- Model --------------------
class NENNClassifier(nn.Module):
    def __init__(self, in_node_dim, in_edge_dim, hidden, num_classes, leak=0.2,
                 edge_edge_index_single=None, edges_per_graph=None):
        super().__init__()
        self.node1 = NodeAttentionLayer(in_node_dim, in_edge_dim, hidden, negative_slope=leak)
        self.edge1 = EdgeAttentionLayer(2*hidden, in_edge_dim, hidden, negative_slope=leak)
        self.node2 = NodeAttentionLayer(2*hidden, 2*hidden, hidden, negative_slope=leak)
        self.readout = nn.Linear(2*hidden, num_classes)

        self.edges_per_graph = edges_per_graph
        if edge_edge_index_single is not None:
            self.register_buffer("edge_edge_index_single", edge_edge_index_single)
        else:
            self.edge_edge_index_single = None  # type: ignore

        # NEW: cache of tiled batched E2E by batch size
        self._e2e_cache = {}  # maps int(num_graphs) -> Tensor on device

    def _get_batched_e2e(self, num_graphs: int, device: torch.device):
        if self.edge_edge_index_single is None or self.edges_per_graph is None:
            return None
        key = int(num_graphs)
        cached = self._e2e_cache.get(key, None)
        if cached is not None and cached.device == device:
            # print(f"[E2E cache] hit for B={num_graphs}")
            return cached
        # build once, then cache
        # print(f"[E2E cache] build for B={num_graphs}")
        e2e = tile_edge_edge_index(self.edge_edge_index_single, num_graphs, self.edges_per_graph, device=device)
        # keep gradients off and store as buffer-like (but not registered)
        e2e = e2e.detach()
        self._e2e_cache[key] = e2e
        return e2e
    
    def embed(self, data):
        """
        Returns final node embeddings (x2), final edge embeddings (e1), and pooled graph embeddings (g)
        for the given batch.
        """
        x, eattr, eidx, batch = data.x, data.edge_attr, data.edge_index, data.batch
        x1 = self.node1(x, eattr, eidx)
        # use the same cached/tiled E2E that forward() uses
        e2e = self._get_batched_e2e(data.num_graphs, x.device) if hasattr(self, "_get_batched_e2e") else None
        e1 = self.edge1(x1, eattr, eidx, edge_edge_index=e2e)  # [E, 2H]
        x2 = self.node2(x1, e1, eidx)                          # [N, 2H]
        g  = global_mean_pool(x2, batch)                       # [B, 2H]
        return x2, e1, g


    def forward(self, data):
        x, eattr, eidx, batch = data.x, data.edge_attr, data.edge_index, data.batch
        x1 = self.node1(x, eattr, eidx)

        # NEW: fetch cached tiled E2E (built only on first occurrence of a given batch size)
        e2e = self._get_batched_e2e(data.num_graphs, x.device)

        e1 = self.edge1(x1, eattr, eidx, edge_edge_index=e2e)
        x2 = self.node2(x1, e1, eidx)
        g  = global_mean_pool(x2, batch)
        return self.readout(g)


# -------------- Loaders ------------------
def make_loaders(expr_df, labels_s, edge_index, edge_attr, batch_size=16, train_per=0.8, test_per=0.1, seed=420, stratify=True):
    ds = ExpressionGraphDataset(expr_df, labels_s, edge_index, edge_attr)

    S = len(ds)
    idx = np.arange(S)

    if stratify:
        from sklearn.model_selection import StratifiedShuffleSplit
        y = ds.y.numpy()
        sss = StratifiedShuffleSplit(n_splits=1, test_size=(1 - train_per), random_state=seed)
        train_idx, temp_idx = next(sss.split(idx, y))
        # split temp into val/test stratified by the remaining ratio
        y_temp = y[temp_idx]
        from sklearn.model_selection import StratifiedShuffleSplit as SSS2
        ratio_test = test_per / (1 - train_per)
        sss2 = SSS2(n_splits=1, test_size=ratio_test, random_state=seed)
        val_rel, test_rel = next(sss2.split(temp_idx, y_temp))
        val_idx  = temp_idx[val_rel]
        test_idx = temp_idx[test_rel]
    else:
        rng = np.random.default_rng(seed)
        rng.shuffle(idx)
        train_end = int(train_per*S); test_end = train_end + int(test_per*S)
        train_idx, test_idx, val_idx = idx[:train_end], idx[train_end:test_end], idx[test_end:]

    subset = torch.utils.data.Subset
    train_ds, val_ds, test_ds = subset(ds, train_idx), subset(ds, val_idx), subset(ds, test_idx)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)
    test_loader  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)
    return train_loader, val_loader, test_loader, ds

# -------------- Logging + Train --------------------
@dataclass
class History:
    epoch: list = field(default_factory=list)
    train_loss: list = field(default_factory=list)
    val_loss: list = field(default_factory=list)
    val_acc: list = field(default_factory=list)

def run_training(train_loader, val_loader, model, device, epochs=30, lr=1e-3, weight_decay=1e-4,
                 log_csv="training_log.csv", plot_png="training_curves.png", log_every=10,
                 class_weights=None, use_amp=False):
    amp_enabled = (device.type == "cuda") and use_amp
    crit = nn.CrossEntropyLoss(weight=(class_weights.to(device) if class_weights is not None else None))
    opt  = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scaler = GradScaler(enabled=amp_enabled)

    hist = History()
    use_cuda = device.type == "cuda"
    if use_cuda:
        torch.backends.cudnn.benchmark = True

    print(f"Device: {device}, torch.cuda.is_available()={torch.cuda.is_available()}")
    
    for epoch in range(1, epochs+1):
        model.train()
        total = 0.0
        pbar = tqdm(enumerate(train_loader), total=len(train_loader), desc=f"Epoch {epoch:03d}", leave=False)
        avg_dt, avg_fw, avg_bw = 0.0, 0.0, 0.0

        t0_epoch = perf_counter()
        for i, batch in pbar:
            t0 = perf_counter()
            batch = batch.to(device, non_blocking=True)
            if use_cuda: torch.cuda.synchronize()
            t1 = perf_counter()

            with torch_autocast(device_type=device.type, enabled=amp_enabled):
                logits = model(batch)
                loss = crit(logits, batch.y)
            if use_cuda: torch.cuda.synchronize()
            t2 = perf_counter()

            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            if use_cuda: torch.cuda.synchronize()
            t3 = perf_counter()

            total += loss.item() * batch.num_graphs
            avg_dt = 0.9*avg_dt + 0.1*(t1 - t0)
            avg_fw = 0.9*avg_fw + 0.1*(t2 - t1)
            avg_bw = 0.9*avg_bw + 0.1*(t3 - t2)

            if (i + 1) % log_every == 0 or i == 0:
                pbar.set_postfix({
                    "data": f"{avg_dt*1000:.1f}ms",
                    "fwd":  f"{avg_fw*1000:.1f}ms",
                    "bwd":  f"{avg_bw*1000:.1f}ms",
                    "loss": f"{(total/((i+1)*batch.num_graphs)):.3f}"
                })

        # validation
        model.eval()
        with torch.no_grad():
            vloss = 0.0; correct = 0; total_g = 0
            for batch in val_loader:
                batch = batch.to(device, non_blocking=True)
                with torch_autocast(device_type=device.type, enabled=amp_enabled):
                    logits = model(batch)
                    loss = crit(logits, batch.y)
                vloss += loss.item() * batch.num_graphs
                pred = logits.argmax(-1)
                correct += (pred == batch.y).sum().item()
                total_g += batch.num_graphs

        train_loss = total / len(train_loader.dataset)
        val_loss   = vloss / max(1, len(val_loader.dataset))
        val_acc    = correct / max(1, total_g)
        epoch_time = perf_counter() - t0_epoch

        print(f"Epoch {epoch:03d} | {epoch_time:.1f}s | train_loss={train_loss:.4f} | "
              f"val_loss={val_loss:.4f} | val_acc={val_acc:.3f}")

        hist.epoch.append(epoch); hist.train_loss.append(train_loss)
        hist.val_loss.append(val_loss); hist.val_acc.append(val_acc)

    # save CSV + plot
    df = pd.DataFrame({"epoch": hist.epoch, "train_loss": hist.train_loss,
                       "val_loss": hist.val_loss, "val_acc": hist.val_acc})
    df.to_csv(log_csv, index=False)
    plt.figure()
    plt.plot(hist.epoch, hist.train_loss, label="Train Loss")
    plt.plot(hist.epoch, hist.val_loss, label="Val Loss")
    plt.xlabel("Epoch"); plt.ylabel("Loss"); plt.legend(); plt.title("Training/Validation Loss")
    plt.savefig(plot_png, bbox_inches="tight", dpi=150); plt.close()
    print(f"Saved training log to {log_csv} and loss curves to {plot_png}")
    return hist


# -------------- Main ---------------------
if __name__ == "__main__":
    # EDIT THESE PATHS:
    adj_file_path = '/home/mzr19001/edge graph net/data/embedding.npy'
    expression_csv = '/home/mzr19001/edge graph net/data/tcga/top500_expression_data.csv'
    labels_csv     = '/home/mzr19001/edge graph net/data/tcga/tcga_labels.csv'   # with column `icluster_cluster_assignment`

    # Load files (pandas/numpy as-is)
    adj_np   = np.load(adj_file_path, allow_pickle=True)
    expr_df  = pd.read_csv(expression_csv, index_col=0)
    labels_s = pd.read_csv(labels_csv, index_col=0)["icluster_cluster_assignment"]

    # Build edges
    edge_index, edge_attr = build_edges_from_npy(adj_np, undirected=False)

    # Sanity checks
    N = expr_df.shape[1]
    assert int(edge_index.max()) < N, f"Edge index exceeds number of genes ({N})."
    assert edge_attr.ndim == 2 and edge_attr.shape[1] == 1536, f"edge_attr must be [E,1536], got {edge_attr.shape}"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Precompute single-graph edge->edge index ONCE (uses GPU if available)
    edges_per_graph = edge_index.size(1)
    single_e2e = precompute_edge_edge_index_single(edge_index.to(device), N)  # [2, E2E_single]
    print(f"Single-graph E2E pairs: {single_e2e.size(1)}")

    # Loaders (you can also add num_workers/pin_memory for GPU)
    train_loader, val_loader, test_loader, ds = make_loaders(
        expr_df, labels_s, edge_index, edge_attr, batch_size=16, stratify=True
    )

    # Model
    in_node_dim = 1
    in_edge_dim = edge_attr.size(1)  # 1536
    hidden = 64
    num_classes = int(labels_s.nunique())
    train_weights = class_weights_from_subset(train_loader.dataset, num_classes)
    model = NENNClassifier(
        in_node_dim, in_edge_dim, hidden, num_classes, leak=0.2,
        edge_edge_index_single=single_e2e,           # <-- pass precomputed single-graph E2E
        edges_per_graph=edges_per_graph
    ).to(device)

    idx_vis = val_loader.dataset.indices[0] if hasattr(val_loader.dataset, "indices") else 0
    node_emb_before, edge_emb_before, graph_emb_before = get_single_graph_embeddings(ds, idx_vis, model, device)
    print(f"[Embeddings BEFORE] node={node_emb_before.shape}, edge={edge_emb_before.shape}, graph={graph_emb_before.shape}")

    hist = run_training(train_loader, val_loader, model, device,
                        epochs=100, lr=1e-3, weight_decay=1e-4,
                        class_weights=train_weights, use_amp=True)

    stats = eval_on_loader(test_loader, model, device)
    print(f"Test acc={stats['acc']:.3f}, macro-F1={stats['macro_f1']:.3f}")
    print(stats["report"])
    plot_confusion_matrix(stats["cm"], out_path="confusion_matrix.png", title="Confusion Matrix")

    node_emb_after, edge_emb_after, graph_emb_after = get_single_graph_embeddings(ds, idx_vis, model, device)
    print(f"[Embeddings AFTER] node={node_emb_after.shape}, edge={edge_emb_after.shape}, graph={graph_emb_after.shape}")

    # 2D versions
    plot_before_after_embeddings(node_emb_before, node_emb_after,
                                out_path="node_emb_before_after_2d.png", title="Node Embeddings", n_components=2)
    plot_before_after_embeddings(edge_emb_before, edge_emb_after,
                                out_path="edge_emb_before_after_2d.png", title="Edge Embeddings", n_components=2)
    graph_map(model, ds, device, n_components=2, out_path="graph_embeddings_2d.png")

    # 3D versions
    plot_before_after_embeddings(node_emb_before, node_emb_after,
                                out_path="node_emb_before_after_3d.png", title="Node Embeddings", n_components=3)
    plot_before_after_embeddings(edge_emb_before, edge_emb_after,
                                out_path="edge_emb_before_after_3d.png", title="Edge Embeddings", n_components=3)
    graph_map(model, ds, device, n_components=3, out_path="graph_embeddings_3d.png")
