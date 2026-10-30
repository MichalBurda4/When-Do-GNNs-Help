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

## Fair protocol (Class 4)

- Same Planetoid train/val/test masks for every model
- MLP / XGBoost ignore edges; GCN uses `edge_index`
- Report mean ± std over 5 seeds
- Experiment A: drop 0/20/40/60/80/100% of undirected edges
