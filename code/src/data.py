"""Dataset loading and fixed train/val/test masks."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch
from torch_geometric.datasets import Planetoid
from torch_geometric.transforms import NormalizeFeatures


DATA_ROOT = Path(__file__).resolve().parents[1] / "data"


@dataclass
class GraphBundle:
    name: str
    x: torch.Tensor
    y: torch.Tensor
    edge_index: torch.Tensor
    train_mask: torch.Tensor
    val_mask: torch.Tensor
    test_mask: torch.Tensor
    num_classes: int
    num_features: int
    num_nodes: int
    num_edges: int


def load_planetoid(name: str = "Cora") -> GraphBundle:
    """Load a Planetoid citation graph with feature normalization."""
    dataset = Planetoid(
        root=str(DATA_ROOT / "Planetoid"),
        name=name,
        transform=NormalizeFeatures(),
    )
    data = dataset[0]
    return GraphBundle(
        name=name,
        x=data.x,
        y=data.y,
        edge_index=data.edge_index,
        train_mask=data.train_mask,
        val_mask=data.val_mask,
        test_mask=data.test_mask,
        num_classes=int(dataset.num_classes),
        num_features=int(dataset.num_features),
        num_nodes=int(data.num_nodes),
        num_edges=int(data.edge_index.size(1) // 2),
    )


def audit_bundle(bundle: GraphBundle) -> dict:
    """Data audit numbers for Class 4 slides / README."""
    y = bundle.y
    class_counts = {
        int(c): int((y == c).sum()) for c in y.unique().tolist()
    }
    train_n = int(bundle.train_mask.sum())
    val_n = int(bundle.val_mask.sum())
    test_n = int(bundle.test_mask.sum())
    overlap = int(
        (bundle.train_mask & bundle.val_mask).sum()
        + (bundle.train_mask & bundle.test_mask).sum()
        + (bundle.val_mask & bundle.test_mask).sum()
    )
    return {
        "dataset": bundle.name,
        "num_nodes": bundle.num_nodes,
        "num_edges_undirected": bundle.num_edges,
        "num_features": bundle.num_features,
        "num_classes": bundle.num_classes,
        "train_nodes": train_n,
        "val_nodes": val_n,
        "test_nodes": test_n,
        "split_overlap_nodes": overlap,
        "class_counts": class_counts,
        "feature_min": float(bundle.x.min()),
        "feature_max": float(bundle.x.max()),
        "feature_mean": float(bundle.x.mean()),
    }
