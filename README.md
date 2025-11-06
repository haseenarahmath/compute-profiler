
![Python](https://img.shields.io/badge/Python-3.9%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-%F0%9F%A6%81-red)
![License: MIT](https://img.shields.io/badge/License-MIT-green)
![Status](https://img.shields.io/badge/Build-passing-success)


# Compute Profiler

A lightweight, modular toolkit for profiling **deep neural networks** — computing
**FLOPs**, **parameter counts**, and core **model complexity metrics** for 
MLP, CNN, and GNN architectures.  
Built with clean modularity, this repo is designed for both research reproducibility 
and industry-grade benchmarking.

---

## Features

✅ Supports multiple model families:
- **MLP** – dense feedforward networks  
- **CNN** – 2D convolutional architectures  
- **GNN** – adjacency-based graph networks (GCN-style)

✅ Modular structure  
Easily extend by adding new architectures under `src/models/`.

✅ Universal interface  
Works seamlessly for models with either `forward(x)` or `forward(x, adj)` signatures.

✅ Lightweight  
No external dependencies beyond PyTorch (and optional PyTorch-Geometric for GNN).

---

## Repository Structure

```

compute-profiler/
├── main.py                     # CLI entry point
├── src/
│   ├── flops_counter.py        # FLOPs counter core (from Sovrasov's MIT impl)
│   ├── core/
│   │   ├── parser.py           # Command-line argument parser
│   │   ├── builder.py          # Model + input constructor
│   │   └── analyzer.py         # FLOPs/Params calculation & reporting
│   └── models/
│       ├── mlp.py              # Feedforward MLP model
│       ├── cnn.py              # TinyCNN model
│       └── gnn.py              # TinyDeepGCN model (optional)
├── tests/                      # Optional validation tests
│   └── test_flops.py
├── README.md
├── requirements.txt
└── .gitignore

````

---

##  Quickstart

### 1️⃣ Install dependencies
```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
````

### 2️⃣ Run FLOPs/Params analysis

#### 🔹 Multi-Layer Perceptron (MLP)

```bash
python main.py --arch mlp --in_dim 128 --hidden 256 --depth 3
```

#### 🔹 Convolutional Neural Network (CNN)

```bash
python main.py --arch tinycnn --in_ch 3 --height 224 --width 224 --cnn_width 32
```

#### 🔹 Graph Neural Network (GNN)

> *(Requires `src/data.py` and `src/layers.py` for adjacency loading)*

```bash
python main.py --arch deepgcn --dataset cora --hid 128 --layers 3 --dropout 0.1
```

---

## 🧾 Example Output

```
Summary
-------
Model : tinycnn
Input : [N=1, C=3, H=224, W=224]
FLOPs : 225.41 MFLOPs
Params: 1.23 M
```

---

## Design Philosophy

* **Modular:** each component (parser, builder, analyzer, model) is isolated for clarity.
* **Extensible:** plug new models under `src/models/` with minimal change.
* **Transparent:** uses open FLOPs counting logic (Sovrasov’s MIT implementation).
* **Portable:** works in both research notebooks and CLI environments.

---

## Example Extension
To add a custom model (say `Transformer`):

1. Create `src/models/transformer.py`
2. Implement `class Transformer(nn.Module): ...`
3. Register it in `src/core/builder.py`
4. Run:

   ```bash
   python main.py --arch transformer --args ...
   ```

---

## 📜 License

MIT License — retains attribution for the FLOPs counter by Aleksandr Sovrasov (MIT License).
You’re free to use and modify this code for academic or commercial purposes.

---

