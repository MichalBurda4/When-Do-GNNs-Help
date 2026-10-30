"""Class 4 data audit: print + save dataset stats for Cora (and optional PubMed)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data import audit_bundle, load_planetoid  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="Cora", choices=["Cora", "CiteSeer", "PubMed"])
    parser.add_argument(
        "--out",
        type=Path,
        default=ROOT.parent / "RESULTS" / "data_audit.json",
    )
    args = parser.parse_args()

    bundle = load_planetoid(args.dataset)
    report = audit_bundle(bundle)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(f"Dataset: {report['dataset']}")
    print(f"  nodes={report['num_nodes']}  undirected_edges={report['num_edges_undirected']}")
    print(f"  features={report['num_features']}  classes={report['num_classes']}")
    print(
        f"  split: train={report['train_nodes']}  "
        f"val={report['val_nodes']}  test={report['test_nodes']}"
    )
    print(f"  split_overlap_nodes={report['split_overlap_nodes']} (must be 0)")
    print(f"  class_counts={report['class_counts']}")
    print(f"Saved: {args.out}")


if __name__ == "__main__":
    main()
