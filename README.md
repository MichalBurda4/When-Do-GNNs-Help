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
- `code/` — Class 4 baselines (data audit + MLP / XGBoost / GCN)
- `RESULTS/` — saved audit + baseline / Experiment A metrics
- `CI2026-ProjectClasses_v3.pdf` — course project instructions

## Quick start (Class 4)

```bash
cd code
pip install -r requirements.txt
python scripts/audit_data.py --dataset Cora
python scripts/run_baselines.py --dataset Cora --seeds 0 1 2 3 4
```

## Presentation (Overleaf / local)

See `class3-presentation/README_OVERLEAF.md`. Compile `main.tex` with pdfLaTeX (MiKTeX / Overleaf).
