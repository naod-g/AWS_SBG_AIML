# Week 02: Mini Semantic Search

A small semantic search engine over a CSV of AWS FAQs. Instead of matching keywords, it matches **meaning**: a query like *"how can I save money on my cloud costs?"* finds the FAQ about reducing your AWS bill even though the words differ.

## Pipeline

| Step | What happens | Tool |
|------|--------------|------|
| 1. Load | Read `faqs.csv` into a DataFrame | Pandas |
| 2. Clean | Strip HTML, collapse whitespace, drop empty rows, drop case-insensitive duplicate questions | Pandas + `re` |
| 3. Chunk | Split long answers into 50-word windows with 10-word overlap; every chunk keeps its question | Python |
| 4. Embed | Convert each chunk into a 384-dim vector | `sentence-transformers` (`all-MiniLM-L6-v2`) |
| 5. Store | Save vectors, text and metadata in a persistent collection (cosine distance) | Chroma |
| 6. Query | Embed the query, fetch nearest chunks, keep the best chunk per FAQ, print the top 3 | Chroma |

## Files

```
week-02/
├── README.md              # this file
├── semantic_search.py     # full pipeline as a script
├── semantic_search.ipynb  # same pipeline as a notebook (Colab / Jupyter)
├── faqs.csv               # 26 AWS FAQs (includes deliberately messy rows)
└── requirements.txt
```

## How to run

**Locally**
```bash
cd week-02
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python semantic_search.py
```

**Google Colab**: upload `semantic_search.ipynb`, run all cells, and upload `faqs.csv` when prompted. The first run downloads the embedding model (about 90 MB).

## Example queries

Run `show("your question")` in the notebook, or edit `example_queries` in the script.

| Query | Expected top match |
|-------|--------------------|
| `How can I save money on my cloud costs?` | *How can I reduce my AWS bill?* |
| `Who is allowed to access my resources?` | IAM FAQs (*What is AWS IAM?*, *least privilege*) |
| `run code without managing servers` | *What is AWS Lambda?* |
| `keep my database online if a data center fails` | *How do I make an RDS database highly available?* |
| `store old files cheaply` | *What are the S3 storage classes?* |
| `how do I protect my account login?` | *How do I enable MFA for my AWS root account?* |

Output format:

```
🔎 Query: run code without managing servers
----------------------------------------------------------------------
1. [Lambda] What is AWS Lambda?  (similarity 0.xx)
   AWS Lambda runs your code in response to events without you managing servers...
2. ...
3. ...
```

## Design notes

- **Why chunk?** Embedding models work best on short passages, and long answers dilute meaning. Overlap stops an idea from being cut in half at a boundary.
- **Why prefix chunks with the question?** A chunk like "Enable Multi-AZ deployment..." is meaningless alone; "Q: How do I make an RDS database highly available?" gives it context.
- **Why dedupe results by FAQ?** A long FAQ produces several chunks, and without dedupe they could fill all 3 result slots.
- **Cosine similarity** is shown as `1 - distance` (closer to 1 means more similar).

## Possible improvements

- Filter by category with Chroma's `where={"category": "S3"}`
- Compare other embedding models
- Add a small evaluation set of query -> expected FAQ and measure top-3 accuracy
