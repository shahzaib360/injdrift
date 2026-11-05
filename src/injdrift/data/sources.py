"""Raw dataset loaders.

Every loader pulls a single file from the Hugging Face Hub (cached under ``HF_HOME``)
and returns plain dicts, so the rest of the code never depends on the ``datasets``
library. All of these sources are public and ungated.
"""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path


def _hub_file(repo: str, filename: str) -> Path:
    from huggingface_hub import hf_hub_download

    return Path(hf_hub_download(repo, filename, repo_type="dataset"))


def _read_parquet(repo: str, filename: str) -> list[dict]:
    import pandas as pd

    return pd.read_parquet(_hub_file(repo, filename)).to_dict("records")


@lru_cache(maxsize=1)
def alpaca_cleaned() -> list[dict]:
    """yahma/alpaca-cleaned: 51.8k {instruction, input, output} records."""
    return json.loads(_hub_file("yahma/alpaca-cleaned", "alpaca_data_cleaned.json").read_text())


@lru_cache(maxsize=1)
def alpaca_farm_eval() -> list[dict]:
    """AlpacaFarm evaluation set (805 records; 208 carry an ``input``).

    The 208 records with input are the prompt-injection benchmark used by StruQ,
    SecAlign and Meta-SecAlign.
    """
    path = _hub_file("tatsu-lab/alpaca_farm", "alpaca_farm_evaluation.json")
    return json.loads(path.read_text())


@lru_cache(maxsize=1)
def squad_validation() -> list[dict]:
    rows = _read_parquet("rajpurkar/squad", "plain_text/validation-00000-of-00001.parquet")
    for r in rows:
        r["answers"] = [str(a) for a in r["answers"]["text"]]
    return rows


@lru_cache(maxsize=1)
def dolly() -> list[dict]:
    path = _hub_file("databricks/databricks-dolly-15k", "databricks-dolly-15k.jsonl")
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


@lru_cache(maxsize=1)
def medical_flashcards() -> list[dict]:
    path = _hub_file(
        "medalpaca/medical_meadow_medical_flashcards",
        "medical_meadow_wikidoc_medical_flashcards.json",
    )
    return json.loads(path.read_text())


@lru_cache(maxsize=1)
def rag12k() -> list[dict]:
    return _read_parquet(
        "neural-bridge/rag-dataset-12000",
        "data/train-00000-of-00001-9df3a936e1f63191.parquet",
    )


@lru_cache(maxsize=1)
def glaive_function_calling() -> list[dict]:
    path = _hub_file("glaiveai/glaive-function-calling-v2", "glaive-function-calling-v2.json")
    return json.loads(path.read_text())


@lru_cache(maxsize=1)
def ultrafeedback_prefs() -> list[dict]:
    return _read_parquet(
        "HuggingFaceH4/ultrafeedback_binarized",
        "data/train_prefs-00000-of-00001.parquet",
    )
