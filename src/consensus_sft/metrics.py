from typing import Dict, List

import os

import evaluate

_CACHE_DIR = os.environ.get("HF_HOME", os.path.join("/tmp", "hf_cache"))
_rouge = evaluate.load("rouge", cache_dir=_CACHE_DIR)
_bleu = evaluate.load("bleu", cache_dir=_CACHE_DIR)


def compute_text_metrics(preds: List[str], refs: List[str]) -> Dict[str, float]:
    rouge = _rouge.compute(predictions=preds, references=refs, use_stemmer=True)
    bleu = _bleu.compute(predictions=preds, references=[[r] for r in refs])
    return {
        "bleu": float(bleu.get("bleu", 0.0)),
        "rouge1": float(rouge.get("rouge1", 0.0)),
        "rouge2": float(rouge.get("rouge2", 0.0)),
        "rougeL": float(rouge.get("rougeL", 0.0)),
    }
