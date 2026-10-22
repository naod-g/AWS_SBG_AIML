# Week 1 — Small Python Data-Processing App

## What this folder contains

- `data/raw_students.csv` — raw dataset committed in the repo
- `src/load.py` — load CSV or an in-code list
- `src/clean.py` — named cleaning functions
- `src/process.py` — NumPy calculations
- `notebooks/week01_analysis.ipynb` — runnable notebook
- `requirements.txt` — includes `numpy`

## Create a venv

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Run the notebook

```bash
jupyter notebook notebooks/week01_analysis.ipynb
```

Run all cells from top to bottom. Saved cell outputs should already be present in the committed notebook.
