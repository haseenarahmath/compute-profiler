import argparse

def build_argparser():
    p = argparse.ArgumentParser(description="Compute FLOPs & Params for MLP/CNN/GNN models")
    p.add_argument("--arch", choices=["mlp", "tinycnn", "deepgcn"], default="tinycnn")
    p.add_argument("--out_dim", type=int, default=10)

    # MLP args
    p.add_argument("--in_dim", type=int, default=128)
    p.add_argument("--hidden", type=int, default=256)
    p.add_argument("--depth", type=int, default=3)
    p.add_argument("--dropout", type=float, default=0.0)

    # CNN args
    p.add_argument("--in_ch", type=int, default=3)
    p.add_argument("--height", type=int, default=224)
    p.add_argument("--width", type=int, default=224)
    p.add_argument("--cnn_width", type=int, default=32)

    # GNN args
    p.add_argument("--dataset", choices=["cora", "citeseer", "pubmed"], default="cora")
    p.add_argument("--hid", type=int, default=128)
    p.add_argument("--layers", type=int, default=3)
    p.add_argument("--norm_mode", type=str, default="PN-SI")
    p.add_argument("--norm_scale", type=float, default=1.0)

    return p
