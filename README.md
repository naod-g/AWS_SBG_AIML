# AWS SBG AI-ML

Course work lives in weekly folders. Week 1 is a small Python data-processing app.

## Week 1

All assignment files are in [`week1/`](week1/).

### Open in Colab

The notebook is `week1/notebooks/week01_analysis.ipynb`:

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/naod-g/AWS_SBG_ALML/blob/main/week1/notebooks/week01_analysis.ipynb)

### Setup (venv)

From the repo root:

```bash
cd week1
python3 -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

`.venv/` is ignored by git (see `.gitignore`). Do not commit virtualenv, `__pycache__`, or `.ipynb_checkpoints` files.

### Run

```bash
cd week1
source .venv/bin/activate
jupyter notebook notebooks/week01_analysis.ipynb
```

Or open the same notebook in VS Code / Cursor and run all cells.

Colab: Runtime → Run all. The first cell installs `numpy` if needed and locates `week1/` next to the notebook.
