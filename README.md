# hist-text-utils

Reusable Python utilities for historical text analysis, developed for research on Dutch colonial records and recipe manuscripts.

## Installation

```bash
# Core only (no heavy deps)
pip install hist-text-utils

# With LLM (Ollama) support
pip install "hist-text-utils[llm]"

# With transformer embeddings (requires ~2 GB disk)
pip install "hist-text-utils[embeddings]"

# With clustering (HDBSCAN + UMAP)
pip install "hist-text-utils[cluster]"

# With visualisation
pip install "hist-text-utils[viz]"

# Everything
pip install "hist-text-utils[all]"

# Directly from the repository
pip install "git+https://github.com/HoekR/hist-text-utils.git"
```

## Modules

### `hist_text_utils.text` — Lightweight text utilities

No heavy dependencies required.

```python
from hist_text_utils.text import (
    get_vocab_stats,
    generate_ngrams,
    calculate_bigram_prob,
    generate_sentences,
    extract_food_snippets,
    deduplicate_snippets,
    get_labels_for_snippet,
)

# Vocabulary statistics
stats = get_vocab_stats(list_of_token_lists)

# N-grams
bigrams = list(generate_ngrams(["word1", "word2", "word3"], n=2))

# Extract text windows around food keywords
snippets = extract_food_snippets(df, text_col="text", food_terms=["suiker", "peper"])

# Remove near-duplicate snippets
unique = deduplicate_snippets(snippets)
```

### `hist_text_utils.embeddings` — Transformer-based embeddings

Requires `pip install "hist-text-utils[embeddings]"` (torch, transformers).
Optional upgrade to `[sbert]` for sentence-transformers.

```python
from hist_text_utils.embeddings import GysberEmbedder, extract_best_passages

embedder = GysberEmbedder(model_name="emanjavacas/GysBERT")
vec = embedder.get_embedding("eenige ponden suiker")

# Retrieve best matching passages from a list of spans
best = extract_best_passages(query="suiker", spans=["...", "..."], embedder=embedder)
```

### `hist_text_utils.llm` — LLM annotation backend

Requires `pip install "hist-text-utils[llm]"` for Ollama or `[openai]` for OpenAI.

```python
from hist_text_utils.llm import LLMBackend

# Ollama (local — Ollama must be running)
llm = LLMBackend(model="llama3.2", backend="ollama")
result = llm.analyze_text(text="...", prompt="Classify the preservation method:")

# OpenAI (OPENAI_API_KEY env variable required)
llm = LLMBackend(model="gpt-4o-mini", backend="openai")
```

### `hist_text_utils.context` — Keyword-in-context analysis and clustering

Full optional dep stack: embeddings, HDBSCAN, UMAP, spaCy.

```python
from hist_text_utils.context import ContextAnalyzer

analyzer = ContextAnalyzer(
    embedder=None,          # pass a GysberEmbedder or SentenceTransformer
    window_size=50,
    use_sentence_transformers=True,  # requires [sbert]
    st_model="emanjavacas/GysBERT",
)

# Extract keyword-in-context windows
kwic_df = analyzer.keyword_in_context(df, text_col="text", keywords=["conserveren"])

# Full pipeline: KWIC → embed → cluster → 2D layout → summarise
results, fig = analyzer.analyze_contexts_for_terms(
    df,
    text_col="text",
    keywords=["conserveren", "zouten"],
    output_dir="output/clusters",
)
```

## Project layout

```
hist_text_utils/
├── __init__.py
├── context.py      # ContextAnalyzer — KWIC, clustering, 2-D layout
├── embeddings.py   # GysberEmbedder, extract_best_passages
├── llm.py          # LLMBackend — Ollama / OpenAI integration
└── text.py         # Vocabulary, n-gram, snippet utilities
```

## Development

```bash
git clone https://github.com/HoekR/hist-text-utils.git
cd hist-text-utils
pip install -e ".[all]"
```

## Provenance

This is made for various research projects at the Huygens Instituut.
