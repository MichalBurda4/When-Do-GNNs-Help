"""Controlled graph corruption for Experiments A/B."""

from __future__ import annotations

import torch


def drop_edges(edge_index: torch.Tensor, drop_frac: float, seed: int) -> torch.Tensor:
    """
    Experiment A: randomly remove a fraction of undirected edges.

    Planetoid stores both directions; we drop pairs consistently by working on
    unique undirected edges (i < j), then rebuilding a bidirectional edge_index.
    """
    if drop_frac <= 0:
        return edge_index.clone()
    if drop_frac >= 1:
        return edge_index.new_zeros((2, 0))

    g = torch.Generator().manual_seed(seed)
    src, dst = edge_index
    # keep upper-triangle undirected unique edges
    mask = src < dst
    u, v = src[mask], dst[mask]
    n = u.numel()
    if n == 0:
        return edge_index.clone()

    keep = int(round(n * (1.0 - drop_frac)))
    keep = max(0, min(n, keep))
    perm = torch.randperm(n, generator=g)[:keep]
    u_keep, v_keep = u[perm], v[perm]

    # bidirectional
    ei = torch.stack(
        [torch.cat([u_keep, v_keep]), torch.cat([v_keep, u_keep])],
        dim=0,
    )
    return ei
