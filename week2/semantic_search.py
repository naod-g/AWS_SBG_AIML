# %% [markdown]
# # Mini Semantic Search over AWS FAQs
# **Week 02 project.** Pipeline: load FAQs with Pandas -> clean -> chunk -> embed -> store in Chroma -> query and print the top 3 matches.
#
# Semantic search matches by *meaning* rather than exact keywords, so "how do I cut costs?" can find an FAQ about reducing the AWS bill.

# %%
import html
import re
from pathlib import Path

import chromadb
import pandas as pd
from sentence_transformers import SentenceTransformer

# Settings you can tweak
CSV_PATH = Path("faqs.csv")
MODEL_NAME = "all-MiniLM-L6-v2"   # small, fast, good quality embedding model
CHUNK_WORDS = 50                  # max words per chunk
CHUNK_OVERLAP = 10                # words shared between neighbouring chunks
TOP_K = 3

# %% [markdown]
# ## 1. Load the data with Pandas

# %%
if not CSV_PATH.exists():
    try:  # Google Colab: ask for the file
        from google.colab import files
        print("Please upload faqs.csv")
        files.upload()
    except ImportError:
        raise FileNotFoundError("faqs.csv not found - run this script from the week-02 folder")

raw_df = pd.read_csv(CSV_PATH)
print("Raw shape:", raw_df.shape)
raw_df.head()

# %% [markdown]
# ## 2. Clean the data
# The raw file has realistic problems: HTML tags, extra whitespace, a duplicate question with different casing, and a row with a missing answer. We fix all of them.

# %%
def clean_text(text: str) -> str:
    """Unescape HTML entities, strip tags, collapse whitespace."""
    text = html.unescape(str(text))
    text = re.sub(r"<[^>]+>", " ", text)   # remove HTML tags
    text = re.sub(r"\s+", " ", text)       # collapse spaces/newlines
    return text.strip()


df = raw_df.dropna(subset=["question", "answer"]).copy()
for col in ["category", "question", "answer"]:
    df[col] = df[col].map(clean_text)

df = df[(df["question"] != "") & (df["answer"] != "")]
# remove duplicate questions, ignoring case
df = (df.assign(_key=df["question"].str.lower())
        .drop_duplicates(subset="_key", keep="first")
        .drop(columns="_key")
        .reset_index(drop=True))

print(f"Rows before: {len(raw_df)} | after cleaning: {len(df)}")
df.head()

# %% [markdown]
# ## 3. Chunk the text
# Embedding models work best on short passages. Long answers are split into overlapping word windows so no idea gets cut off at a boundary. Each chunk keeps its question so it still makes sense on its own.

# %%
def chunk_words(text: str, size: int = CHUNK_WORDS, overlap: int = CHUNK_OVERLAP) -> list[str]:
    """Split text into chunks of at most `size` words with `overlap` words shared."""
    words = text.split()
    if len(words) <= size:
        return [text]
    step = size - overlap
    chunks = []
    for start in range(0, len(words), step):
        chunks.append(" ".join(words[start:start + size]))
        if start + size >= len(words):
            break
    return chunks


records = []
for row in df.itertuples():
    for i, chunk in enumerate(chunk_words(row.answer)):
        records.append({
            "chunk_id": f"faq{row.id}-c{i}",
            "faq_id": int(row.id),
            "category": row.category,
            "question": row.question,
            "chunk_index": i,
            "document": f"Q: {row.question}\nA: {chunk}",
        })

chunks_df = pd.DataFrame(records)
print(f"{len(df)} FAQs -> {len(chunks_df)} chunks")
chunks_df[["chunk_id", "category", "document"]].head()

# %% [markdown]
# ## 4. Embed and store in Chroma
# We turn each chunk into a vector (384 numbers) with a sentence-transformer, then store vectors, text and metadata in a Chroma collection that uses cosine distance.

# %%
model = SentenceTransformer(MODEL_NAME)

embeddings = model.encode(
    chunks_df["document"].tolist(),
    normalize_embeddings=True,
    show_progress_bar=True,
)
print("Embedding matrix shape:", embeddings.shape)

client = chromadb.PersistentClient(path="chroma_db")
try:
    client.delete_collection("aws_faqs")   # start fresh on every run
except Exception:
    pass
collection = client.create_collection("aws_faqs", metadata={"hnsw:space": "cosine"})

collection.add(
    ids=chunks_df["chunk_id"].tolist(),
    documents=chunks_df["document"].tolist(),
    embeddings=embeddings.tolist(),
    metadatas=chunks_df[["faq_id", "category", "question", "chunk_index"]].to_dict("records"),
)
print("Stored in Chroma:", collection.count(), "chunks")

# %% [markdown]
# ## 5. Query and print the top 3 matches
# We embed the query with the same model and ask Chroma for the nearest chunks. Several chunks can come from the same FAQ, so we fetch extra results and keep the best chunk per FAQ before showing the top 3.

# %%
def search(query: str, k: int = TOP_K) -> list[dict]:
    q_emb = model.encode([query], normalize_embeddings=True).tolist()
    res = collection.query(query_embeddings=q_emb, n_results=min(k * 4, collection.count()))

    seen, matches = set(), []
    for doc, meta, dist in zip(res["documents"][0], res["metadatas"][0], res["distances"][0]):
        if meta["faq_id"] in seen:
            continue
        seen.add(meta["faq_id"])
        matches.append({"score": 1 - dist, "category": meta["category"],
                        "question": meta["question"], "text": doc})
        if len(matches) == k:
            break
    return matches


def show(query: str, k: int = TOP_K) -> None:
    print(f"\n🔎 Query: {query}")
    print("-" * 70)
    for rank, m in enumerate(search(query, k), start=1):
        answer = m["text"].split("\nA: ", 1)[1]
        print(f"{rank}. [{m['category']}] {m['question']}  (similarity {m['score']:.2f})")
        print(f"   {answer[:160]}{'...' if len(answer) > 160 else ''}")


example_queries = [
    "How can I save money on my cloud costs?",
    "Who is allowed to access my resources?",
    "run code without managing servers",
    "keep my database online if a data center fails",
    "store old files cheaply",
]
for q in example_queries:
    show(q)

# %% [markdown]
# ## 6. Try your own query

# %%
show("how do I protect my account login?")

# %% [markdown]
# ## Summary
# - **Pandas** loaded and cleaned the FAQ data (HTML, whitespace, duplicates, missing values).
# - Long answers were **chunked** with overlap; each chunk keeps its question for context.
# - **sentence-transformers** produced embeddings; **Chroma** stored them with metadata.
# - Queries are matched by meaning, so they don't need to share keywords with the FAQ text.
