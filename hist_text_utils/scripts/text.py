"""
hist_text_utils.text
--------------------
Vocabulary scoring, n-gram helpers, and snippet extraction utilities.
These functions have *no* heavy optional dependencies — just numpy and pandas.

Functions
---------
get_vocab_stats          Count vocabulary matches in a text (word-boundary).
generate_ngrams          Produce space-joined n-grams from a token list.
calculate_bigram_prob    MLE bigram probability P(w2 | w1).
generate_sentences       Random-walk sentence generator using a bigram model.
extract_food_snippets    Context windows around food-term matches in a document.
deduplicate_snippets     Remove exact and substring-overlapping snippets.
get_labels_for_snippet   Map matched terms to cluster/category labels.
"""
from __future__ import annotations

import ast
import random
import re
from typing import Optional

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Vocabulary scoring
# ---------------------------------------------------------------------------

def get_vocab_stats(text: str, vocab: list[str]) -> tuple[int, list[str]]:
    """
    Count how many terms from *vocab* appear in *text* using word boundaries.

    Returns
    -------
    (count, found_terms) where *found_terms* is the list of matched vocabulary
    entries.
    """
    text_lower = text.lower()
    found: list[str] = []
    for term in vocab:
        if term in text_lower:
            if re.search(r"\b" + re.escape(term) + r"\b", text_lower):
                found.append(term)
    return len(found), found


# ---------------------------------------------------------------------------
# N-gram language model helpers
# ---------------------------------------------------------------------------

def generate_ngrams(tokens: list[str], n: int) -> list[str]:
    """Return a list of space-joined n-grams from *tokens*."""
    return [" ".join(tokens[i : i + n]) for i in range(len(tokens) - n + 1)]


def calculate_bigram_prob(ngrams: list[str], ngram: str) -> float:
    """
    Maximum-likelihood estimate of P(w2 | w1) for the given *ngram* (``"w1 w2"``).

    Parameters
    ----------
    ngrams : list[str]
        All bigrams from the corpus (as produced by :func:`generate_ngrams`).
    ngram : str
        The bigram to estimate (e.g. ``"boter olie"``).
    """
    prefix = ngram.split(" ")[0]
    count_ngram  = ngrams.count(ngram)
    count_prefix = sum(1 for ng in ngrams if ng.startswith(prefix + " "))
    return 0.0 if count_prefix == 0 else count_ngram / count_prefix


def generate_sentences(
    vocab: list[str], bigrams: list[str], num_sentences: int
) -> list[str]:
    """
    Generate *num_sentences* two-token sentences by random-walking a bigram model.

    Parameters
    ----------
    vocab    : starting vocabulary (``vocab[0]`` is the seed word).
    bigrams  : space-joined bigrams from :func:`generate_ngrams`.
    """
    sentences: list[str] = []
    current_word = vocab[0]
    for _ in range(num_sentences):
        candidates = [ng for ng in bigrams if ng.startswith(current_word + " ")]
        if not candidates:
            break
        next_ng   = random.choice(candidates)
        next_word = next_ng.split(" ")[1]
        sentences.append(f"{current_word} {next_word}")
        current_word = next_ng.split(" ")[0]
    return sentences


# ---------------------------------------------------------------------------
# Snippet extraction
# ---------------------------------------------------------------------------

def extract_food_snippets(row: pd.Series, window: int = 2) -> list[str]:
    """
    Extract context snippets around sentences that contain food terms.

    Expects *row* to have ``"text"`` and ``"found_terms"`` fields, as
    produced by applying :func:`get_vocab_stats` to a DataFrame.

    Parameters
    ----------
    row    : pandas Series with at least ``text`` and ``found_terms``.
    window : sentences before/after the matching sentence to include.

    Returns
    -------
    List of merged snippet strings.
    """
    text  = row["text"]
    terms = row["found_terms"]

    if not terms or pd.isna(text):
        return []

    sentences = [
        s.strip()
        for s in re.split(r"(?<=[.?!])\s+", text)
        if s.strip()
    ]
    if not sentences:
        return []

    hit_indices: set[int] = set()
    for idx, sent in enumerate(sentences):
        sent_lower = sent.lower()
        for term in terms:
            if re.search(r"\b" + re.escape(term) + r"\b", sent_lower):
                hit_indices.add(idx)
                break

    if not hit_indices:
        return []

    ranges = [
        (max(0, h - window), min(len(sentences), h + window + 1))
        for h in sorted(hit_indices)
    ]

    merged: list[tuple[int, int]] = []
    cur_s, cur_e = ranges[0]
    for nxt_s, nxt_e in ranges[1:]:
        if nxt_s < cur_e:
            cur_e = max(cur_e, nxt_e)
        else:
            merged.append((cur_s, cur_e))
            cur_s, cur_e = nxt_s, nxt_e
    merged.append((cur_s, cur_e))

    return [" ".join(sentences[s:e]) for s, e in merged]


# ---------------------------------------------------------------------------
# Deduplication
# ---------------------------------------------------------------------------

def deduplicate_snippets(snippets_df: pd.DataFrame) -> pd.DataFrame:
    """
    Remove exact duplicate snippets and snippets that are strict substrings
    of a longer snippet from the same document.

    Parameters
    ----------
    snippets_df : DataFrame with at least ``doc_id`` and ``snippet`` columns.

    Returns
    -------
    Deduplicated DataFrame.
    """
    dedup = snippets_df.drop_duplicates(subset=["snippet"])
    records: list[dict] = []
    for _doc_id, group in dedup.groupby("doc_id"):
        snips = group["snippet"].tolist()
        kept  = [
            s for s in snips
            if not any((s != o) and (s in o) for o in snips)
        ]
        records.extend(group[group["snippet"].isin(kept)].to_dict("records"))
    return pd.DataFrame(records)


# ---------------------------------------------------------------------------
# Label mapping
# ---------------------------------------------------------------------------

def get_labels_for_snippet(
    matched_terms,
    keyword_to_label: dict,
) -> list:
    """
    Map a list of matched terms to their cluster/category labels.

    Parameters
    ----------
    matched_terms    : list of term strings, or a string representation of one.
    keyword_to_label : dict mapping term → label.

    Returns
    -------
    Deduplicated list of labels.
    """
    if not isinstance(matched_terms, list):
        try:
            matched_terms = ast.literal_eval(matched_terms)
        except Exception:
            return []
    return list(
        {keyword_to_label[t] for t in matched_terms if t in keyword_to_label}
    )
