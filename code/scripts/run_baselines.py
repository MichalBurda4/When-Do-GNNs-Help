"""
Class 4: first reproducible baseline comparison on a clean graph.

Models:
  - MLP (ignores edges)
  - XGBoost (ignores edges)
  - GCN (uses edges)

Same Planetoid splits for all models. Multiple seeds -> mean ± std.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data import load_planetoid  # noqa: E402
from src.models import GCNClassifier, MLPClassifier  # noqa: E402
from src.train_eval import train_torch_model, train_xgboost  # noqa: E402


def summarize(rows: list[dict], key: str) -> dict:
    vals = np.array([r[key] for r in rows], dtype=float)
    return {
        f"{key}_mean": float(vals.mean()),
        f"{key}_std": float(vals.std(ddof=1)) if len(vals) > 1 else 0.0,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="Cora", choices=["Cora", "CiteSeer", "PubMed"])
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--hidden", type=int, default=64)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--weight-decay", type=float, default=5e-4)
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=ROOT.parent / "RESULTS",
    )
    args = parser.parse_args()

    bundle = load_planetoid(args.dataset)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    all_rows: list[dict] = []
    summary_rows: list[dict] = []

    for model_name in ["mlp", "xgboost", "gcn"]:
        per_seed: list[dict] = []
        for seed in args.seeds:
            if model_name == "mlp":
                model = MLPClassifier(bundle.num_features, args.hidden, bundle.num_classes)
                metrics = train_torch_model(
                    model,
                    bundle,
                    epochs=args.epochs,
                    lr=args.lr,
                    weight_decay=args.weight_decay,
                    seed=seed,
                )
            elif model_name == "gcn":
                model = GCNClassifier(bundle.num_features, args.hidden, bundle.num_classes)
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
                "model": model_name,
                "seed": seed,
                **metrics,
            }
            per_seed.append(row)
            all_rows.append(row)
            print(
                f"{model_name:8s} seed={seed}  "
                f"test_acc={metrics['test_accuracy']:.4f}  "
                f"val_acc={metrics['val_accuracy']:.4f}"
            )

        summary = {
            "dataset": args.dataset,
            "model": model_name,
            "n_seeds": len(args.seeds),
            **summarize(per_seed, "test_accuracy"),
            **summarize(per_seed, "val_accuracy"),
            **summarize(per_seed, "test_macro_f1"),
        }
        summary_rows.append(summary)
        print(
            f"-> {model_name}: test_acc "
            f"{summary['test_accuracy_mean']:.4f} ± {summary['test_accuracy_std']:.4f}"
        )

    raw_path = args.out_dir / f"baseline_{args.dataset.lower()}_raw.csv"
    sum_path = args.out_dir / f"baseline_{args.dataset.lower()}_summary.csv"
    meta_path = args.out_dir / f"baseline_{args.dataset.lower()}_meta.json"

    with raw_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(all_rows[0].keys()))
        writer.writeheader()
        writer.writerows(all_rows)

    with sum_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
        writer.writeheader()
        writer.writerows(summary_rows)

    meta = {
        "dataset": args.dataset,
        "seeds": args.seeds,
        "epochs": args.epochs,
        "hidden": args.hidden,
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "models": ["mlp", "xgboost", "gcn"],
        "note": "Class 4 clean-graph baseline before edge stress tests.",
    }
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")

    print(f"\nSaved:\n  {raw_path}\n  {sum_path}\n  {meta_path}")


if __name__ == "__main__":
    main()
