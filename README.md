# When Do GNNs Help?

Computational Intelligence project (AGH) — Class 3 materials.

**Authors:** Michał Broś, Michał Burda  
**Course:** Computational Intelligence — Prof. Adrian Horzyk

## Research question

When do Graph Neural Networks beat simple tabular models (MLP / XGBoost), and how much edge noise can they survive before losing?

## Repository structure

- `papers/` — assigned research papers (PDFs)
- `LITERATURE.csv` — reading matrix
- `class3-presentation/` — Beamer (AGH theme) slides + PDF
- `code/` — baselines + Experiments A/B (MLP / XGBoost / GCN)
- `RESULTS/` — audit, baseline, Exp.\ A, partial Exp.\ B
- `class5-presentation/` — Class 5 Beamer slides (first research results)
- `CI2026-ProjectClasses_v3.pdf` — course project instructions

## Quick start

```bash
cd code
pip install -r requirements.txt
python scripts/audit_data.py --dataset Cora
python scripts/run_baselines.py --dataset Cora --seeds 0 1 2 3 4
python scripts/run_experiment_a.py --dataset Cora --seeds 0 1 2 3 4
python scripts/run_experiment_b.py --dataset Cora --seeds 0 1 2
```

## Presentation (Overleaf / local)

See `class3-presentation/README_OVERLEAF.md`. Compile `main.tex` with pdfLaTeX (MiKTeX / Overleaf).
Class 5 deck: `class5-presentation/main.tex`.
