"""
Experiment A: delete real edges and measure when GCN stops beating tabular models.

Drop fractions: 0%, 20%, 40%, 60%, 80%, 100%.
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
from src.graph_noise import drop_edges  # noqa: E402
from src.models import GCNClassifier, MLPClassifier  # noqa: E402
from src.train_eval import train_torch_model, train_xgboost  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="Cora", choices=["Cora", "CiteSeer", "PubMed"])
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    parser.add_argument(
        "--drop-fracs",
        type=float,
        nargs="+",
        default=[0.0, 0.2, 0.4, 0.6, 0.8, 1.0],
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

    for drop in args.drop_fracs:
        for seed in args.seeds:
            ei = drop_edges(base.edge_index, drop, seed=seed)
            bundle = copy(base)
            bundle.edge_index = ei
            n_undirected = int(ei.size(1) // 2)

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
                    "drop_frac": drop,
                    "edges_remaining": n_undirected,
                    "model": model_name,
                    "seed": seed,
                    **metrics,
                }
                rows.append(row)
                print(
                    f"drop={drop:.0%}  {model_name:8s} seed={seed}  "
                    f"test={metrics['test_accuracy']:.4f}  edges={n_undirected}"
                )

    # summary mean/std over seeds
    summary: list[dict] = []
    for drop in args.drop_fracs:
        for model_name in ["mlp", "xgboost", "gcn"]:
            sub = [r for r in rows if r["drop_frac"] == drop and r["model"] == model_name]
            acc = np.array([r["test_accuracy"] for r in sub], dtype=float)
            f1 = np.array([r["test_macro_f1"] for r in sub], dtype=float)
            summary.append(
                {
                    "dataset": args.dataset,
                    "drop_frac": drop,
                    "model": model_name,
                    "n_seeds": len(sub),
                    "test_accuracy_mean": float(acc.mean()),
                    "test_accuracy_std": float(acc.std(ddof=1)) if len(acc) > 1 else 0.0,
                    "test_macro_f1_mean": float(f1.mean()),
                    "test_macro_f1_std": float(f1.std(ddof=1)) if len(f1) > 1 else 0.0,
                    "edges_remaining": sub[0]["edges_remaining"],
                }
            )

    raw_path = args.out_dir / f"expA_{args.dataset.lower()}_raw.csv"
    sum_path = args.out_dir / f"expA_{args.dataset.lower()}_summary.csv"
    meta_path = args.out_dir / f"expA_{args.dataset.lower()}_meta.json"

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
                "experiment": "A_edge_deletion",
                "dataset": args.dataset,
                "drop_fracs": args.drop_fracs,
                "seeds": args.seeds,
                "epochs": args.epochs,
                "models": ["mlp", "xgboost", "gcn"],
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    print("\n=== Summary (test accuracy mean) ===")
    header = f"{'drop':>6}  {'mlp':>8}  {'xgboost':>8}  {'gcn':>8}"
    print(header)
    for drop in args.drop_fracs:
        vals = {}
        for m in ["mlp", "xgboost", "gcn"]:
            s = next(x for x in summary if x["drop_frac"] == drop and x["model"] == m)
            vals[m] = s["test_accuracy_mean"]
        print(
            f"{drop:6.0%}  {vals['mlp']:8.4f}  {vals['xgboost']:8.4f}  {vals['gcn']:8.4f}"
        )

    print(f"\nSaved:\n  {raw_path}\n  {sum_path}\n  {meta_path}")


if __name__ == "__main__":
    main()
