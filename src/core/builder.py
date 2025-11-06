import torch
from src.models.mlp import MLP
from src.models.cnn import TinyCNN

try:
    from src.models.gnn import TinyDeepGCN
    from src.data import load_data
    HAS_GNN = True
except Exception:
    HAS_GNN = False


def build_model_and_inputs(args):
    arch = args.arch.lower()

    if arch == "mlp":
        model = MLP(in_dim=args.in_dim, hidden=args.hidden, out_dim=args.out_dim,
                    depth=args.depth, dropout=args.dropout)
        x = torch.randn(1, args.in_dim)
        return model, x, None

    if arch == "tinycnn":
        model = TinyCNN(in_ch=args.in_ch, num_classes=args.out_dim, width=args.cnn_width)
        x = torch.randn(1, args.in_ch, args.height, args.width)
        return model, x, None

    if arch == "deepgcn":
        if not HAS_GNN:
            raise RuntimeError("GNN dependencies not found. Ensure src/models/gnn.py and src/data.py exist.")
        data = load_data(args.dataset, normalize_feature=True, missing_rate=0, cuda=False)
        in_dim = data.x.size(1)
        out_dim = int(data.y.max()) + 1
        model = TinyDeepGCN(in_dim, args.hid, out_dim, n_layers=args.layers,
                            dropout=args.dropout, norm_mode=args.norm_mode, norm_scale=args.norm_scale)
        return model, data.x, data.adj

    raise ValueError(f"Unknown architecture: {args.arch}")
