"""
hist-text-utils — reusable utilities for historical text analysis.

Modules
-------
context     ContextAnalyzer: KWIC extraction, embedding, clustering
embeddings  GysberEmbedder and passage-retrieval helpers
llm         LLMBackend: unified Ollama / OpenAI wrapper
text        Vocabulary scoring, ngrams, snippet extraction, deduplication
"""

from importlib.metadata import version, PackageNotFoundError

try:
    __version__ = version("hist-text-utils")
except PackageNotFoundError:
    __version__ = "0.0.0"

__all__ = ["context", "embeddings", "llm", "text"]
