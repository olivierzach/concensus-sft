from consensus_sft.metrics import compute_text_metrics


def test_compute_text_metrics():
    preds = ["hello world"]
    refs = ["hello world"]
    metrics = compute_text_metrics(preds, refs)
    assert metrics["bleu"] >= 0.0
    assert metrics["rougeL"] >= 0.9
