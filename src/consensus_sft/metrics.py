from typing import Dict, List, Optional

import os

try:
    import evaluate
except ImportError:  # pragma: no cover - optional dependency
    evaluate = None

_CACHE_DIR = os.environ.get("HF_HOME", os.path.join("/tmp", "hf_cache"))
_rouge_eval: Optional[object] = None
_bleu_eval: Optional[object] = None


def _load_evaluate_metrics() -> None:
    global _rouge_eval, _bleu_eval
    if evaluate is None:
        return
    if _rouge_eval is None or _bleu_eval is None:
        try:
            _rouge_eval = evaluate.load("rouge", cache_dir=_CACHE_DIR)
            _bleu_eval = evaluate.load("bleu", cache_dir=_CACHE_DIR)
        except Exception:
            _rouge_eval = None
            _bleu_eval = None


def compute_text_metrics(preds: List[str], refs: List[str]) -> Dict[str, float]:
    _load_evaluate_metrics()
    if _rouge_eval is not None and _bleu_eval is not None:
        rouge = _rouge_eval.compute(predictions=preds, references=refs, use_stemmer=True)
        bleu = _bleu_eval.compute(predictions=preds, references=[[r] for r in refs])
        return {
            "bleu": float(bleu.get("bleu", 0.0)),
            "rouge1": float(rouge.get("rouge1", 0.0)),
            "rouge2": float(rouge.get("rouge2", 0.0)),
            "rougeL": float(rouge.get("rougeL", 0.0)),
        }

    rouge1 = rouge2 = rougeL = 0.0
    try:
        from rouge_score import rouge_scorer

        scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
        for pred, ref in zip(preds, refs):
            scores = scorer.score(ref, pred)
            rouge1 += scores["rouge1"].fmeasure
            rouge2 += scores["rouge2"].fmeasure
            rougeL += scores["rougeL"].fmeasure
        n = max(len(preds), 1)
        rouge1 /= n
        rouge2 /= n
        rougeL /= n
    except Exception:
        rouge1 = rouge2 = rougeL = 0.0

    bleu = 0.0
    try:
        from nltk.translate.bleu_score import corpus_bleu

        refs_tokens = [[r.split()] for r in refs]
        preds_tokens = [p.split() for p in preds]
        bleu = float(corpus_bleu(refs_tokens, preds_tokens))
    except Exception:
        bleu = 0.0

    return {
        "bleu": float(bleu),
        "rouge1": float(rouge1),
        "rouge2": float(rouge2),
        "rougeL": float(rougeL),
    }
