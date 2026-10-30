"""
Experiment B: add fake edges and measure when GCN stops beating tabular models.

Add fractions (relative to real undirected edges): 0%, 15%, 30%, 45%, 60%, 75%.
Models: MLP, XGBoost, GCN. Same Planetoid splits. Multiple seeds.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from copy import copy
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data import load_planetoid  # noqa: E402
from src.graph_noise import add_fake_edges  # noqa: E402
from src.models import GCNClassifier, MLPClassifier  # noqa: E402
from src.train_eval import train_torch_model, train_xgboost  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="Cora", choices=["Cora", "CiteSeer", "PubMed"])
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    parser.add_argument(
        "--add-fracs",
        type=float,
        nargs="+",
        default=[0.0, 0.15, 0.30, 0.45, 0.60, 0.75],
    )
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--hidden", type=int, default=64)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--weight-decay", type=float, default=5e-4)
    parser.add_argument("--out-dir", type=Path, default=ROOT.parent / "RESULTS")
    args = parser.parse_args()

    base = load_planetoid(args.dataset)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict] = []

    for add in args.add_fracs:
        for seed in args.seeds:
            ei = add_fake_edges(
                base.edge_index,
                add,
                seed=seed,
                num_nodes=base.num_nodes,
            )
            bundle = copy(base)
            bundle.edge_index = ei
            n_undirected = int(ei.size(1) // 2)
            n_fake = n_undirected - base.num_edges

            for model_name in ["mlp", "xgboost", "gcn"]:
                if model_name == "mlp":
                    model = MLPClassifier(base.num_features, args.hidden, base.num_classes)
                    metrics = train_torch_model(
                        model,
                        bundle,
                        epochs=args.epochs,
                        lr=args.lr,
                        weight_decay=args.weight_decay,
                        seed=seed,
                    )
                elif model_name == "gcn":
                    model = GCNClassifier(base.num_features, args.hidden, base.num_classes)
                    metrics = train_torch_model(
                        model,
                        bundle,
                        epochs=args.epochs,
                        lr=args.lr,
                        weight_decay=args.weight_decay,
                        seed=seed,
                    )
                else:
                    metrics = train_xgboost(bundle, seed=seed)

                row = {
                    "dataset": args.dataset,
                    "add_frac": add,
                    "edges_total": n_undirected,
                    "edges_fake": n_fake,
                    "model": model_name,
                    "seed": seed,
                    **metrics,
                }
                rows.append(row)
                print(
                    f"add={add:.0%}  {model_name:8s} seed={seed}  "
                    f"test={metrics['test_accuracy']:.4f}  "
                    f"edges={n_undirected} (+{n_fake} fake)"
                )

    summary: list[dict] = []
    for add in args.add_fracs:
        for model_name in ["mlp", "xgboost", "gcn"]:
            sub = [r for r in rows if r["add_frac"] == add and r["model"] == model_name]
            acc = np.array([r["test_accuracy"] for r in sub], dtype=float)
            f1 = np.array([r["test_macro_f1"] for r in sub], dtype=float)
            summary.append(
                {
                    "dataset": args.dataset,
                    "add_frac": add,
                    "model": model_name,
                    "n_seeds": len(sub),
                    "test_accuracy_mean": float(acc.mean()),
                    "test_accuracy_std": float(acc.std(ddof=1)) if len(acc) > 1 else 0.0,
                    "test_macro_f1_mean": float(f1.mean()),
                    "test_macro_f1_std": float(f1.std(ddof=1)) if len(f1) > 1 else 0.0,
                    "edges_total": sub[0]["edges_total"],
                    "edges_fake": sub[0]["edges_fake"],
                }
            )

    raw_path = args.out_dir / f"expB_{args.dataset.lower()}_raw.csv"
    sum_path = args.out_dir / f"expB_{args.dataset.lower()}_summary.csv"
    meta_path = args.out_dir / f"expB_{args.dataset.lower()}_meta.json"

    with raw_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    with sum_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(summary[0].keys()))
        w.writeheader()
        w.writerows(summary)

    meta_path.write_text(
        json.dumps(
            {
                "experiment": "B_fake_edges",
                "dataset": args.dataset,
                "add_fracs": args.add_fracs,
                "seeds": args.seeds,
                "epochs": args.epochs,
                "models": ["mlp", "xgboost", "gcn"],
                "note": "add_frac is relative to the number of real undirected edges",
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    print("\n=== Summary (test accuracy mean) ===")
    print(f"{'add':>6}  {'mlp':>8}  {'xgboost':>8}  {'gcn':>8}")
    for add in args.add_fracs:
        vals = {}
        for m in ["mlp", "xgboost", "gcn"]:
            s = next(x for x in summary if x["add_frac"] == add and x["model"] == m)
            vals[m] = s["test_accuracy_mean"]
        print(
            f"{add:6.0%}  {vals['mlp']:8.4f}  {vals['xgboost']:8.4f}  {vals['gcn']:8.4f}"
        )

    print(f"\nSaved:\n  {raw_path}\n  {sum_path}\n  {meta_path}")


if __name__ == "__main__":
    main()
