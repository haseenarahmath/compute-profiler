from src.flops_counter import get_model_complexity_info

def analyze_model(model, x, adj, args):
    flops, params = get_model_complexity_info(model, x, adj, print_per_layer_stat=True, as_strings=True, trim=True)

    print("\nSummary\n-------")
    print(f"Model : {args.arch}")
    if args.arch == "mlp":
        print(f"Input : [N=1, D={args.in_dim}]")
    elif args.arch == "tinycnn":
        print(f"Input : [N=1, C={args.in_ch}, H={args.height}, W={args.width}]")
    else:
        print(f"Graph : dataset={args.dataset}")
    print(f"FLOPs : {flops}")
    print(f"Params: {params}")
