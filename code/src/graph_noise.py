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


def add_fake_edges(
    edge_index: torch.Tensor,
    add_frac: float,
    seed: int,
    num_nodes: int,
) -> torch.Tensor:
    """
    Experiment B: keep all real undirected edges; add random non-existing edges.

    ``add_frac`` is relative to the number of real undirected edges
    (e.g. 0.30 ≈ +30% fake edges). Self-loops are never added.
    """
    if add_frac <= 0:
        return edge_index.clone()

    g = torch.Generator().manual_seed(seed)
    src, dst = edge_index
    mask = src < dst
    u, v = src[mask], dst[mask]
    n_real = int(u.numel())
    n_fake = int(round(n_real * add_frac))
    if n_fake <= 0:
        return edge_index.clone()

    existing = {(int(a), int(b)) for a, b in zip(u.tolist(), v.tolist())}
    max_possible = num_nodes * (num_nodes - 1) // 2 - n_real
    n_fake = min(n_fake, max(0, max_possible))

    fake_pairs: list[tuple[int, int]] = []
    # rejection sampling is fine for Cora-scale graphs
    attempts = 0
    max_attempts = max(10_000, n_fake * 50)
    while len(fake_pairs) < n_fake and attempts < max_attempts:
        attempts += 1
        a = int(torch.randint(0, num_nodes, (1,), generator=g).item())
        b = int(torch.randint(0, num_nodes, (1,), generator=g).item())
        if a == b:
            continue
        if a > b:
            a, b = b, a
        if (a, b) in existing:
            continue
        existing.add((a, b))
        fake_pairs.append((a, b))

    if not fake_pairs:
        return edge_index.clone()

    fu = torch.tensor([p[0] for p in fake_pairs], dtype=edge_index.dtype)
    fv = torch.tensor([p[1] for p in fake_pairs], dtype=edge_index.dtype)
    u_all = torch.cat([u, fu])
    v_all = torch.cat([v, fv])
    ei = torch.stack(
        [torch.cat([u_all, v_all]), torch.cat([v_all, u_all])],
        dim=0,
    )
    return ei
