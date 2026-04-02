"""
hist_text_utils.context
-----------------------
KWIC extraction, embedding, clustering, and metadata-aware context analysis.

The heavy optional dependencies (transformers, torch, sentence-transformers,
hdbscan, umap-learn, spacy) are imported *lazily* inside the methods that need
them. Core usage (KWIC only) requires only pandas and numpy.

Main class
----------
ContextAnalyzer
    keyword_in_context()          — extract keyword-in-context windows
    analyze_contexts_for_terms()  — KWIC → embed → cluster → summarise
    plot_context_clusters()       — scatter plot of 2D projection
    label_clusters_with_terms()   — top terms per cluster
    merge_metadata_into_contexts()— join metadata by filename key
    normalize_filename_basename() — add basename column for merges

Backward-compatible module-level functions are provided at the bottom so that
existing code using the old direct-function API continues to work unchanged.

Example
-------
    from hist_text_utils.context import ContextAnalyzer

    ca = ContextAnalyzer(embed_model_name="emanjavacas/gysbert")
    ctx_df, summary = ca.analyze_contexts_for_terms(
        model=w2v_model,
        word="zouten",
        df=recipes_df,
        meta_cols=["id", "title"],
    )
    ca.plot_context_clusters(ctx_df)
"""
from __future__ import annotations

import logging
import os
from functools import lru_cache
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


# ---------------------------------------------------------------------------
# HuggingFace mean-pool encoder (lazy, cached)
# ---------------------------------------------------------------------------

@lru_cache(maxsize=4)
def _load_hf_encoder(model_name: str):
    """Load HF tokenizer + model onto the best available device (cached)."""
    try:
        from transformers import AutoTokenizer, AutoModel
        import torch
    except ImportError:
        raise ImportError(
            "torch and transformers are required for HF-based embeddings.\n"
            "  pip install torch transformers"
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
    """Encode *texts* with HF mean pooling. Returns (n, dim) numpy array."""
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
            return_tensors="pt", max_length=512,
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.no_grad():
            out = hf_model(**inputs)
        last = out.last_hidden_state
        mask = inputs["attention_mask"].unsqueeze(-1).to(last.dtype)
        embs.append(((last * mask).sum(1) / mask.sum(1).clamp(min=1)).cpu().numpy())
    if not embs:
        return np.zeros((0, hf_model.config.hidden_size))
    return np.vstack(embs)


# ---------------------------------------------------------------------------
# ContextAnalyzer
# ---------------------------------------------------------------------------

class ContextAnalyzer:
    """
    Main class for KWIC collection, embedding, clustering, and visualisation.

    All embedding / clustering dependencies are optional — they are imported
    inside the methods that use them and degrade gracefully:

    - Embeddings: HuggingFace (if model_name has ``/`` or ``GysBERT``) →
      SentenceTransformers → w2v average → TF-IDF + SVD fallback
    - Clustering: HDBSCAN → KMeans fallback
    - 2D projection: UMAP → PCA fallback
    - Cluster labelling: spaCy → simple regex fallback

    Parameters
    ----------
    embed_model_name : str
        HuggingFace or SentenceTransformers model name. Empty string = auto.
    random_state : int
        Random seed for reproducibility.
    context_window : int
        Default number of tokens on each side of the keyword.
    cluster_min_size : int
        Minimum cluster size for HDBSCAN.
    use_umap : bool
        Try UMAP for 2D projection (falls back to PCA if unavailable).
    filename_col : str
        Default column name for document identifiers.
    text_col : str
        Default column name for document text.
    """

    def __init__(
        self,
        embed_model_name: str = "",
        random_state: int = 42,
        context_window: int = 6,
        cluster_min_size: int = 5,
        use_umap: bool = True,
        filename_col: str = "filename",
        text_col: str = "text",
    ):
        self.embed_model_name = embed_model_name
        self.random_state     = random_state
        self.context_window   = context_window
        self.cluster_min_size = cluster_min_size
        self.use_umap         = use_umap
        self.filename_col     = filename_col
        self.text_col         = text_col
        self._embedding_model = None
        self._spacy_nlp       = None

    # ── KWIC ─────────────────────────────────────────────────────────────────

    def keyword_in_context(
        self,
        word: str,
        df,
        context_window: Optional[int] = None,
        meta_cols: Optional[List[str]] = None,
        filename_col: Optional[str] = None,
        text_col: Optional[str] = None,
        match_fn: Optional[Callable[[str, str], bool]] = None,
    ) -> List[dict]:
        """
        Extract keyword-in-context windows from *df*, preserving metadata.

        Parameters
        ----------
        word         : Keyword to match (case-insensitive exact token by default).
        df           : DataFrame with document rows.
        context_window : Tokens on each side (uses instance default if None).
        meta_cols    : Extra columns to copy into each result dict.
        filename_col : Column for document identifier.
        text_col     : Column for document text.
        match_fn     : Optional ``(token, word) -> bool`` for custom matching.

        Returns
        -------
        List of dicts with keys: ``filename``, ``left_context``, ``keyword``,
        ``right_context``, ``full_context``, plus any *meta_cols*.
        """
        cw   = context_window or self.context_window
        fcol = filename_col   or self.filename_col
        tcol = text_col       or self.text_col
        mcols = list(meta_cols) if meta_cols else []
        contexts: List[dict] = []

        for _, row in df.iterrows():
            text = row.get(tcol, "")
            if not isinstance(text, str) or not text:
                continue
            words = text.split()
            for i, tok in enumerate(words):
                if match_fn is not None:
                    try:
                        matched = bool(match_fn(tok, word))
                    except Exception:
                        matched = False
                else:
                    matched = tok.lower() == word.lower()

                if matched:
                    start = max(0, i - cw)
                    end   = min(len(words), i + cw + 1)
                    ctx: dict = {
                        "filename":      row.get(fcol),
                        "left_context":  " ".join(words[start:i]),
                        "keyword":       words[i],
                        "right_context": " ".join(words[i + 1:end]),
                        "full_context":  " ".join(words[start:end]),
                    }
                    for col in mcols:
                        ctx[col] = row.get(col)
                    contexts.append(ctx)
        return contexts

    # ── Embeddings ────────────────────────────────────────────────────────────

    def _get_embeddings(self, texts: List[str]) -> np.ndarray:
        """Generate embeddings for *texts* using the configured strategy."""
        model_name = self.embed_model_name
        if model_name and ("/" in model_name or "GysBERT" in model_name):
            return _hf_mean_pool_encode(texts, model_name)
        try:
            from sentence_transformers import SentenceTransformer
            if self._embedding_model is None:
                name = model_name or "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
                self._embedding_model = SentenceTransformer(name)
            return self._embedding_model.encode(
                texts, show_progress_bar=False, convert_to_numpy=True
            )
        except ImportError:
            raise

    # ── Full pipeline ─────────────────────────────────────────────────────────

    def analyze_contexts_for_terms(
        self,
        model,
        word: str,
        df,
        top_n: int = 10,
        context_window: Optional[int] = None,
        embed_model_name: Optional[str] = None,
        cluster_min_size: Optional[int] = None,
        use_umap: Optional[bool] = None,
        similar_words_fn: Optional[Callable] = None,
        meta_cols: Optional[List[str]] = None,
        filename_col: Optional[str] = None,
        text_col: Optional[str] = None,
    ) -> Tuple[Optional[pd.DataFrame], Dict]:
        """
        Collect KWIC → embed → cluster → 2D project → summarise.

        Parameters
        ----------
        model           : Word-vector model (used as fallback embedder).
        word            : Target keyword.
        df              : Source DataFrame.
        top_n           : Number of similar words to add via *similar_words_fn*.
        similar_words_fn: ``(model, word, top_n) -> [(word, score), ...]``.
        meta_cols       : Metadata columns to carry through.

        Returns
        -------
        ``(contexts_df, summary_dict)`` — contexts_df has columns
        ``term``, ``filename``, ``left``, ``keyword``, ``right``, ``full``,
        ``cluster``, ``x``, ``y``, and any *meta_cols*.
        """
        cw   = context_window   or self.context_window
        ename = embed_model_name if embed_model_name is not None else self.embed_model_name
        cmin  = cluster_min_size or self.cluster_min_size
        umap  = use_umap         if use_umap is not None else self.use_umap
        fcol  = filename_col     or self.filename_col
        tcol  = text_col         or self.text_col

        # Gather terms
        terms = [word]
        if similar_words_fn is not None:
            try:
                sims  = similar_words_fn(model, word, top_n)
                terms = [word] + [w for w, _ in sims]
            except Exception:
                pass

        # Collect KWIC
        rows: List[dict] = []
        for t in terms:
            for c in self.keyword_in_context(
                t, df, context_window=cw, meta_cols=meta_cols,
                filename_col=fcol, text_col=tcol,
            ):
                row: dict = {
                    "term": t,
                    "filename": c.get("filename"),
                    "left":     c.get("left_context"),
                    "keyword":  c.get("keyword"),
                    "right":    c.get("right_context"),
                    "full":     c.get("full_context"),
                }
                if meta_cols:
                    for mc in meta_cols:
                        row[mc] = c.get(mc)
                rows.append(row)

        if not rows:
            return None, {"error": f"no contexts found for: {', '.join(terms)}"}

        ctx_df = pd.DataFrame(rows).reset_index(drop=True)
        texts  = ctx_df["full"].astype(str).tolist()

        # Embed — three-level fallback
        old_ename = self.embed_model_name
        if ename:
            self.embed_model_name = ename
        embeds = None
        try:
            embeds = self._get_embeddings(texts)
        except Exception:
            try:
                vecs = []
                for txt in texts:
                    toks = txt.split()
                    if hasattr(model, "get_word_vector"):
                        wv = [model.get_word_vector(w) for w in toks]
                    else:
                        try:
                            wv = [model.wv[w] for w in toks if w in model.wv]
                        except Exception:
                            wv = []
                    if wv:
                        vecs.append(np.mean(wv, axis=0))
                    else:
                        dim = getattr(model, "vector_size", 300)
                        vecs.append(np.zeros(int(dim)))
                embeds = np.vstack(vecs)
            except Exception:
                from sklearn.feature_extraction.text import TfidfVectorizer
                from sklearn.decomposition import TruncatedSVD
                X = TfidfVectorizer(max_features=2000).fit_transform(texts)
                embeds = TruncatedSVD(
                    n_components=min(128, X.shape[1] - 1),
                    random_state=self.random_state,
                ).fit_transform(X)
        finally:
            self.embed_model_name = old_ename

        # Cluster
        try:
            import hdbscan
            labels = hdbscan.HDBSCAN(
                min_cluster_size=max(2, cmin), metric="euclidean"
            ).fit_predict(embeds)
        except Exception:
            from sklearn.cluster import KMeans
            k = min(12, max(2, len(texts) // max(1, cmin)))
            labels = KMeans(
                n_clusters=k, random_state=self.random_state
            ).fit_predict(embeds)

        ctx_df["cluster"] = labels

        # 2D projection
        try:
            proj = None
            if umap:
                try:
                    import umap as umap_lib
                    proj = umap_lib.UMAP(
                        n_components=2, random_state=self.random_state
                    ).fit_transform(embeds)
                except Exception:
                    proj = None
            if proj is None:
                from sklearn.decomposition import PCA
                proj = PCA(
                    n_components=2, random_state=self.random_state
                ).fit_transform(embeds)
            ctx_df["x"] = proj[:, 0]
            ctx_df["y"] = proj[:, 1]
        except Exception:
            n = len(ctx_df)
            ctx_df["x"] = np.arange(n, dtype=float)
            ctx_df["y"] = np.zeros(n)

        # Summarise
        from sklearn.metrics.pairwise import pairwise_distances
        summary: Dict = {}
        for lbl in sorted(set(labels)):
            mask   = ctx_df["cluster"] == lbl
            sub    = ctx_df[mask]
            name   = "noise" if lbl == -1 else f"cluster_{lbl}"
            try:
                centroid = embeds[mask].mean(axis=0).reshape(1, -1)
                rep_idx  = np.argmin(pairwise_distances(centroid, embeds[mask]).ravel())
                rep      = sub.iloc[rep_idx]["full"]
            except Exception:
                rep = sub.iloc[0]["full"]
            summary[name] = {
                "n_examples":    int(len(sub)),
                "top_terms":     sub["term"].value_counts().to_dict(),
                "representative": rep,
                "examples_sample": sub["full"].head(6).tolist(),
            }

        return ctx_df, summary

    # ── Visualisation ─────────────────────────────────────────────────────────

    def plot_context_clusters(
        self,
        contexts_df: pd.DataFrame,
        cluster_col: str = "cluster",
        x_col: str = "x",
        y_col: str = "y",
        figsize: tuple = (10, 7),
        annotate_each: Optional[int] = None,
        cmap: str = "tab20",
    ):
        """Scatter plot of 2D-projected contexts coloured by cluster."""
        import matplotlib.pyplot as plt
        import seaborn as sns

        df = contexts_df.copy()
        if x_col not in df.columns or y_col not in df.columns:
            raise ValueError("contexts_df must have 'x' and 'y' columns.")
        df[cluster_col] = df[cluster_col].fillna(-1).astype(int)
        clusters = sorted(df[cluster_col].unique())
        palette  = sns.color_palette(cmap, n_colors=max(len(clusters), 3))

        plt.figure(figsize=figsize)
        for i, c in enumerate(clusters):
            sub   = df[df[cluster_col] == c]
            label = str(c) if c != -1 else "noise"
            plt.scatter(
                sub[x_col], sub[y_col],
                s=30, color=palette[i % len(palette)], alpha=0.7, label=label,
            )
            if annotate_each:
                for _, row in sub.head(annotate_each).iterrows():
                    txt = (row.get("keyword") or row.get("term") or "")[:40]
                    plt.annotate(txt, (row[x_col], row[y_col]), fontsize=8, alpha=0.9)

        plt.legend(title="cluster", bbox_to_anchor=(1.02, 1), loc="upper left")
        plt.title("Context clusters (2D projection)")
        plt.xlabel("x")
        plt.ylabel("y")
        plt.grid(True, linestyle="--", alpha=0.3)
        plt.tight_layout()
        plt.show()

    # ── Cluster labelling ─────────────────────────────────────────────────────

    def label_clusters_with_terms(
        self,
        contexts_df: pd.DataFrame,
        cluster_col: str = "cluster",
        text_col: str = "full",
        top_n: int = 6,
        language_hint: Optional[str] = None,
        extra_stopwords: Optional[list] = None,
    ) -> Tuple[Dict, pd.DataFrame]:
        """
        Return top terms per cluster using spaCy lemmatisation (or regex fallback).

        Returns
        -------
        ``(cluster_terms_dict, summary_df)``
        """
        import re
        from collections import Counter

        df = contexts_df.copy()
        df[cluster_col] = df[cluster_col].fillna(-1).astype(int)
        stop: set = set(extra_stopwords or [])

        use_spacy = False
        if self._spacy_nlp is None:
            try:
                import spacy
                candidates = (
                    [language_hint + "_core_news_sm",
                     language_hint + "_core_web_sm", "en_core_web_sm"]
                    if language_hint
                    else ["nl_core_news_sm", "fr_core_news_sm", "en_core_web_sm"]
                )
                for m in candidates:
                    try:
                        self._spacy_nlp = spacy.load(m, disable=["ner", "parser"])
                        use_spacy = True
                        break
                    except Exception:
                        continue
            except Exception:
                pass
        else:
            use_spacy = True

        cluster_terms: Dict = {}
        for c in sorted(df[cluster_col].unique()):
            rows   = df[df[cluster_col] == c][text_col].astype(str).tolist()
            tokens: List[str] = []
            if use_spacy and self._spacy_nlp is not None:
                for doc in self._spacy_nlp.pipe(rows, batch_size=32):
                    for tok in doc:
                        if tok.is_alpha and not tok.is_stop:
                            lemma = tok.lemma_.lower()
                            if lemma and lemma not in stop:
                                tokens.append(lemma)
            else:
                for t in rows:
                    for w in re.findall(r"\w+", t.lower()):
                        if len(w) > 2 and w not in stop:
                            tokens.append(w)
            cluster_terms[int(c)] = Counter(tokens).most_common(top_n)

        summary_df = pd.DataFrame([
            {"cluster": c,
             "n_examples": int((df[cluster_col] == c).sum()),
             "top_terms":  ", ".join(t for t, _ in terms)}
            for c, terms in cluster_terms.items()
        ]).sort_values("cluster").reset_index(drop=True)

        return cluster_terms, summary_df

    # ── Metadata merging ──────────────────────────────────────────────────────

    @staticmethod
    def normalize_filename_basename(
        df: pd.DataFrame,
        filename_col: str = "filename",
        out_col: str = "filename_norm",
    ) -> pd.DataFrame:
        """Add a column with the basename of each filename."""
        df[out_col] = df[filename_col].astype(str).apply(
            lambda p: os.path.basename(p) if p and not pd.isna(p) else p
        )
        return df

    @staticmethod
    def merge_metadata_into_contexts(
        contexts_df: pd.DataFrame,
        meta_df: pd.DataFrame,
        filename_col: str = "filename",
        meta_cols: Optional[List[str]] = None,
        meta_filename_col: str = "filename",
        prefer_meta_on_conflict: bool = True,
        normalize_basename: bool = True,
        filename_norm_col: str = "filename_norm",
    ) -> pd.DataFrame:
        """
        Left-join metadata from *meta_df* into *contexts_df* by filename,
        with a basename-normalised fallback merge.
        """
        ctx  = contexts_df.copy()
        meta = meta_df.copy()

        if meta_cols is None:
            meta_cols = [
                c for c in ["correspondent_namen", "date_parsed", "author", "date"]
                if c in meta.columns
            ]

        if normalize_basename:
            if filename_norm_col not in ctx.columns:
                ctx[filename_norm_col] = ctx[filename_col].astype(str).apply(
                    lambda p: p.split("/")[-1] if p else p
                )
            if filename_norm_col not in meta.columns:
                meta[filename_norm_col] = meta[meta_filename_col].astype(str).apply(
                    lambda p: p.split("/")[-1] if p else p
                )
            m1 = meta[[meta_filename_col] + meta_cols].drop_duplicates(subset=[meta_filename_col])
            merged = ctx.merge(
                m1, left_on=filename_col, right_on=meta_filename_col,
                how="left", suffixes=("", "_meta"),
            )
            m2 = meta[[filename_norm_col] + meta_cols].drop_duplicates(subset=[filename_norm_col])
            merged = merged.merge(
                m2, on=filename_norm_col, how="left", suffixes=("", "_meta2")
            )
            for mc in meta_cols:
                col2 = mc + "_meta2"
                if col2 in merged.columns:
                    merged[mc] = merged[mc].combine_first(merged[col2])
                    merged = merged.drop(columns=[col2])
            if meta_filename_col in merged.columns and meta_filename_col != filename_col:
                merged = merged.drop(columns=[meta_filename_col], errors="ignore")
            return merged

        m = meta[[meta_filename_col] + meta_cols].drop_duplicates(subset=[meta_filename_col])
        return ctx.merge(
            m, left_on=filename_col, right_on=meta_filename_col,
            how="left", suffixes=("", "_meta"),
        )


# ---------------------------------------------------------------------------
# Backward-compatible module-level function wrappers
# ---------------------------------------------------------------------------

def keyword_in_context(
    word: str,
    df,
    context_window: int = 5,
    meta_cols: Optional[List[str]] = None,
    filename_col: str = "filename",
    text_col: str = "text",
    match_fn: Optional[Callable[[str, str], bool]] = None,
) -> List[dict]:
    """Backward-compatible wrapper — see ContextAnalyzer.keyword_in_context."""
    return ContextAnalyzer(
        context_window=context_window,
        filename_col=filename_col,
        text_col=text_col,
    ).keyword_in_context(word, df, context_window, meta_cols, filename_col, text_col, match_fn)


def analyze_contexts_for_terms(
    model, word: str, df, top_n: int = 10, context_window: int = 6,
    embed_model_name: str = "", cluster_min_size: int = 5, use_umap: bool = True,
    random_state: int = 42, similar_words_fn: Optional[Callable] = None,
    meta_cols: Optional[List[str]] = None, filename_col: str = "filename",
    text_col: str = "text",
) -> Tuple[Optional[pd.DataFrame], Dict]:
    """Backward-compatible wrapper — see ContextAnalyzer.analyze_contexts_for_terms."""
    return ContextAnalyzer(
        embed_model_name=embed_model_name, random_state=random_state,
        context_window=context_window, cluster_min_size=cluster_min_size,
        use_umap=use_umap, filename_col=filename_col, text_col=text_col,
    ).analyze_contexts_for_terms(
        model, word, df, top_n, context_window, embed_model_name,
        cluster_min_size, use_umap, similar_words_fn, meta_cols, filename_col, text_col,
    )


def plot_context_clusters(
    contexts_df, cluster_col="cluster", x_col="x", y_col="y",
    figsize=(10, 7), annotate_each: Optional[int] = None, cmap="tab20",
):
    """Backward-compatible wrapper — see ContextAnalyzer.plot_context_clusters."""
    ContextAnalyzer().plot_context_clusters(
        contexts_df, cluster_col, x_col, y_col, figsize, annotate_each, cmap
    )


def label_clusters_with_terms(
    contexts_df, cluster_col="cluster", text_col="full", top_n=6,
    language_hint=None, extra_stopwords=None,
):
    """Backward-compatible wrapper — see ContextAnalyzer.label_clusters_with_terms."""
    return ContextAnalyzer().label_clusters_with_terms(
        contexts_df, cluster_col, text_col, top_n, language_hint, extra_stopwords
    )


def merge_metadata_into_contexts(
    contexts_df, meta_df, filename_col="filename", meta_cols=None,
    meta_filename_col="filename", prefer_meta_on_conflict=True,
    normalize_basename=True, filename_norm_col="filename_norm",
):
    """Backward-compatible wrapper — see ContextAnalyzer.merge_metadata_into_contexts."""
    return ContextAnalyzer.merge_metadata_into_contexts(
        contexts_df, meta_df, filename_col, meta_cols, meta_filename_col,
        prefer_meta_on_conflict, normalize_basename, filename_norm_col,
    )


def normalize_filename_basename(
    df, filename_col="filename", out_col="filename_norm"
):
    """Backward-compatible wrapper — see ContextAnalyzer.normalize_filename_basename."""
    return ContextAnalyzer.normalize_filename_basename(df, filename_col, out_col)
