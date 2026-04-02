"""
hist_text_utils.embeddings
--------------------------
Transformer-based embeddings for historical (Dutch) text.

The heavy dependencies (torch, transformers) are *optional* — they are only
imported when you actually instantiate GysberEmbedder or call
_hf_mean_pool_encode(). If they are not installed, a clear ImportError is
raised with install instructions.

Typical usage
-------------
    from hist_text_utils.embeddings import GysberEmbedder, extract_best_passages

    embedder = GysberEmbedder()                       # auto-detects MPS/CUDA/CPU
    vec  = embedder.get_embedding("een historische zin")
    vecs = embedder.get_embeddings_batch(["zin 1", "zin 2"])

    scored = extract_best_passages(long_text, query_vec, embedder, top_k=3)
"""
from __future__ import annotations

from functools import lru_cache
from typing import Optional, Sequence

import numpy as np


# ---------------------------------------------------------------------------
# Internal HuggingFace mean-pool helpers (lazy import)
# ---------------------------------------------------------------------------

@lru_cache(maxsize=4)
def _load_hf_encoder(model_name: str):
    """Load HF tokenizer + model and move to best available device."""
    try:
        from transformers import AutoTokenizer, AutoModel
        import torch
    except ImportError:
        raise ImportError(
            "torch and transformers are required for HF-based embeddings. "
            "Install them with:\n  pip install torch transformers"
        )
    try:
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")
    except Exception:
        device = torch.device("cpu")

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    model.to(device)
    model.eval()
    return tokenizer, model, device


def _hf_mean_pool_encode(
    texts: Sequence[str], model_name: str, batch_size: int = 32
) -> np.ndarray:
    """Encode texts with HF model + mean pooling. Returns (n, dim) array."""
    try:
        import torch
    except ImportError:
        raise ImportError("torch is required — pip install torch")

    tokenizer, hf_model, device = _load_hf_encoder(model_name)
    embs = []
    for i in range(0, len(texts), batch_size):
        batch = list(texts[i : i + batch_size])
        inputs = tokenizer(
            batch, padding=True, truncation=True,
            return_tensors="pt", max_length=512
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.no_grad():
            out = hf_model(**inputs)
        last = out.last_hidden_state
        mask = inputs["attention_mask"].unsqueeze(-1).to(last.dtype)
        summed = (last * mask).sum(1)
        counts = mask.sum(1).clamp(min=1)
        embs.append((summed / counts).cpu().numpy())
    if not embs:
        return np.zeros((0, hf_model.config.hidden_size))
    return np.vstack(embs)


# ---------------------------------------------------------------------------
# GysberEmbedder
# ---------------------------------------------------------------------------

class GysberEmbedder:
    """
    Wraps a HuggingFace transformer model for sentence/token embeddings with
    mean pooling. Defaults to ``emanjavacas/gysbert`` (Dutch historical text),
    but accepts any HF model identifier.

    Parameters
    ----------
    model_name : str
        HuggingFace model identifier. Default: ``"emanjavacas/gysbert"``.
    device : str or None
        Force a device (``"cpu"``, ``"mps"``, ``"cuda"``). If *None*,
        auto-detects MPS > CUDA > CPU.

    Raises
    ------
    ImportError
        If ``torch`` or ``transformers`` are not installed.
    """

    def __init__(
        self,
        model_name: str = "emanjavacas/gysbert",
        device: Optional[str] = None,
    ):
        try:
            import torch
            from transformers import AutoTokenizer, AutoModel
        except ImportError:
            raise ImportError(
                "GysberEmbedder requires torch and transformers.\n"
                "  pip install torch transformers"
            )

        if device is not None:
            self.device = torch.device(device)
        elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            self.device = torch.device("mps")
        elif torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")

        print(f"Using device: {self.device}")
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(self.device)

    def get_embedding(self, text: str) -> np.ndarray:
        """Mean-pooled embedding for a single text string."""
        import torch
        inputs = self.tokenizer(
            text, return_tensors="pt", truncation=True, max_length=512
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = self.model(**inputs)
        return outputs.last_hidden_state.mean(dim=1).squeeze().cpu().numpy()

    def get_embeddings_batch(
        self, texts: list[str], batch_size: int = 64
    ) -> np.ndarray:
        """Mean-pooled embeddings for a list of texts (batched)."""
        import torch
        all_embeddings = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            inputs = self.tokenizer(
                batch, return_tensors="pt", padding=True,
                truncation=True, max_length=512,
            )
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            with torch.no_grad():
                outputs = self.model(**inputs)
            all_embeddings.append(
                outputs.last_hidden_state.mean(dim=1).cpu().numpy()
            )
        return np.vstack(all_embeddings)


# ---------------------------------------------------------------------------
# Passage retrieval
# ---------------------------------------------------------------------------

def extract_best_passages(
    text: str,
    query_emb: np.ndarray,
    embedder: GysberEmbedder,
    top_k: int = 3,
) -> list[tuple[float, str]]:
    """
    Split *text* into sentences and rank them by cosine similarity to
    *query_emb* using *embedder*.

    Returns
    -------
    list of (score, sentence) tuples, sorted best-first.
    """
    sentences = [
        s.strip()
        for s in text.replace("?", ".").replace("!", ".").split(".")
        if len(s.strip()) > 20
    ]
    if not sentences:
        return []

    sent_embs = embedder.get_embeddings_batch(sentences)
    norm_sent  = sent_embs / np.linalg.norm(sent_embs, axis=1, keepdims=True)
    norm_query = query_emb / np.linalg.norm(query_emb)
    scores     = np.dot(norm_sent, norm_query).flatten()

    top_idx = np.argsort(scores)[::-1][:top_k]
    return [(float(scores[i]), sentences[i]) for i in top_idx]
