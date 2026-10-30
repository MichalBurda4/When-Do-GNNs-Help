# Code — Class 4 baseline

First reproducible evaluation on a **clean** citation graph (before edge deletion / fake edges).

## Setup

```bash
cd code
pip install -r requirements.txt
```

## 1) Data audit

```bash
python scripts/audit_data.py --dataset Cora
```

Writes `../RESULTS/data_audit.json`.

## 2) Baseline comparison (MLP vs XGBoost vs GCN)

```bash
python scripts/run_baselines.py --dataset Cora --seeds 0 1 2 3 4
```

Writes:

- `../RESULTS/baseline_cora_raw.csv`
- `../RESULTS/baseline_cora_summary.csv`
- `../RESULTS/baseline_cora_meta.json`

## 3) Experiment A — delete edges

```bash
python scripts/run_experiment_a.py --dataset Cora --seeds 0 1 2 3 4
```

Writes `../RESULTS/expA_cora_*.csv`.

## 4) Experiment B — add fake edges

```bash
# Class 5 partial (3 seeds):
python scripts/run_experiment_b.py --dataset Cora --seeds 0 1 2

# Full (5 seeds):
python scripts/run_experiment_b.py --dataset Cora --seeds 0 1 2 3 4
```

Writes `../RESULTS/expB_cora_*.csv`.
`add_frac` is relative to the number of real undirected edges
(0 / 15 / 30 / 45 / 60 / 75%).

## Fair protocol

- Same Planetoid train/val/test masks for every model
- MLP / XGBoost ignore edges; GCN uses `edge_index`
- Report mean ± std over seeds (5 for A; 3 so far for B)
- Experiment A: drop 0/20/40/60/80/100% of undirected edges
- Experiment B: keep real edges; add random fake edges
