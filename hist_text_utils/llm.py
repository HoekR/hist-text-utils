"""
hist_text_utils.llm
-------------------
Unified LLM wrapper supporting Ollama (local, free) and OpenAI.

All LLM dependencies are *optional* at import time — the relevant client
library is only imported when you instantiate LLMBackend with that backend.

Usage
-----
    from hist_text_utils.llm import LLMBackend

    # Local Ollama (requires `ollama serve` running):
    llm = LLMBackend(model="llama3.2", backend="ollama")

    # OpenAI (requires OPENAI_API_KEY env var + `pip install openai`):
    llm = LLMBackend(model="gpt-4o-mini", backend="openai")

    reply = llm.analyze_text(text="Does this recipe use salt?",
                             prompt="You are a food history expert. Answer YES or NO.")
"""
from __future__ import annotations

import os
from typing import Optional


class LLMBackend:
    """
    Thin, unified wrapper around Ollama or OpenAI chat completions.

    Parameters
    ----------
    model : str
        Model name.
        - Ollama:  any locally pulled model, e.g. ``"llama3.2"``, ``"mistral"``
        - OpenAI:  ``"gpt-4o-mini"``, ``"gpt-4o"``, ``"gpt-3.5-turbo"``, …
    backend : str
        ``"ollama"`` (default) or ``"openai"``.
    host : str, optional
        Custom Ollama server URL (e.g. ``"http://remote-host:11434"``).
        Ignored for OpenAI.

    Raises
    ------
    ImportError
        If the required client library is not installed.
    EnvironmentError
        If ``OPENAI_API_KEY`` is not set when using the OpenAI backend.
    ValueError
        If an unknown backend name is given.
    """

    def __init__(
        self,
        model: str = "llama3.2",
        backend: str = "ollama",
        host: Optional[str] = None,
    ):
        self.model   = model
        self.backend = backend

        if backend == "ollama":
            try:
                import ollama
            except ImportError:
                raise ImportError(
                    "The 'ollama' package is not installed.\n"
                    "  pip install ollama\n"
                    "Also make sure Ollama is running: ollama serve"
                )
            self._client = ollama.Client(host=host)

        elif backend == "openai":
            try:
                import openai
            except ImportError:
                raise ImportError(
                    "The 'openai' package is not installed.\n"
                    "  pip install openai"
                )
            api_key = os.environ.get("OPENAI_API_KEY")
            if not api_key:
                raise EnvironmentError(
                    "OPENAI_API_KEY environment variable is not set.\n"
                    "Export it before starting Jupyter:\n"
                    "  export OPENAI_API_KEY='sk-...'"
                )
            self._client = openai.OpenAI(api_key=api_key)

        else:
            raise ValueError(
                f"Unknown backend '{backend}'. Choose 'ollama' or 'openai'."
            )

    # ── public API ────────────────────────────────────────────────────────────

    def analyze_text(self, text: str, prompt: str = "") -> str:
        """
        Send *prompt* (system instruction) + *text* (user message) to the LLM.

        Parameters
        ----------
        text   : User message / query.
        prompt : System instruction prepended to the conversation.

        Returns
        -------
        str — the model's reply text.
        """
        messages: list[dict] = []
        if prompt:
            messages.append({"role": "system",  "content": prompt})
        messages.append(    {"role": "user",    "content": text})

        if self.backend == "ollama":
            response = self._client.chat(model=self.model, messages=messages)
            return response.message.content
        else:
            response = self._client.chat.completions.create(
                model=self.model, messages=messages
            )
            return response.choices[0].message.content
