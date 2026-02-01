import argparse
import os
import re
from typing import List, Tuple

import pandas as pd
from rouge_score import rouge_scorer


def clean_text(text: str) -> str:
    text = "" if text is None else str(text)
    text = text.replace("\u00a0", " ")
    text = re.sub(r"\s+", " ", text).strip()
    return text


def normalize_label(text: str) -> str:
    text = clean_text(text)
    text = text.strip("\"")
    text = text.replace("“", '"').replace("”", '"').replace("’", "'")
    text = text.strip()
    if text.endswith(" ."):
        text = text[:-2]
    return text


def split_sentences(text: str) -> List[str]:
    text = clean_text(text)
    if not text:
        return []
    return re.split(r"(?<=[.!?])\s+", text)


def build_prompt(query: str, title: str, abstract: str) -> str:
    parts = [
        "Instruction: Answer the question using only the abstract. If the abstract does not contain the answer, respond with 'Not enough information.'.",
        f"Question: {clean_text(query)}",
    ]
    if title:
        parts.append(f"Title: {clean_text(title)}")
    parts.append(f"Abstract: {clean_text(abstract)}")
    return " ".join(parts)


def token_set(text: str) -> set:
    return set(re.findall(r"\w+", text.lower()))


def overlap_ratio(label: str, abstract: str) -> float:
    l_set = token_set(label)
    if not l_set:
        return 0.0
    a_set = token_set(abstract)
    return len(l_set & a_set) / len(l_set)


def trim_abstract(text: str, max_sentences: int = 8) -> str:
    sentences = split_sentences(text)
    if not sentences:
        return ""
    return " ".join(sentences[:max_sentences])


def rouge_l_f1(hypothesis: str, reference: str) -> float:
    if not hypothesis or not reference:
        return 0.0
    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
    return scorer.score(reference, hypothesis)["rougeL"].fmeasure


def select_oracle_sentences(
    abstract: str,
    reference: str,
    max_sentences: int = 3,
    min_sentence_tokens: int = 5,
) -> Tuple[str, float]:
    sentences = split_sentences(abstract)
    if not sentences:
        return "", 0.0

    scored = []
    for idx, sent in enumerate(sentences):
        if len(sent.split()) < min_sentence_tokens:
            continue
        score = rouge_l_f1(sent, reference)
        scored.append((idx, sent, score))

    if not scored:
        return "", 0.0

    scored.sort(key=lambda x: x[2], reverse=True)
    selected = sorted(scored[:max_sentences], key=lambda x: x[0])
    oracle = " ".join([s[1] for s in selected])
    best_score = max([s[2] for s in selected])
    return oracle, best_score


def to_bullets(text: str, max_bullets: int = 3) -> str:
    sentences = split_sentences(text)[:max_bullets]
    if not sentences:
        return ""
    return "\n".join([f"- {s}" for s in sentences])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_csv", required=True)
    parser.add_argument("--output_csv", required=True)
    parser.add_argument("--target_style", choices=["label", "oracle", "bullets"], default="oracle")
    parser.add_argument("--oracle_reference", choices=["label", "query_label", "query_only"], default="query_label")
    parser.add_argument("--oracle_sentences", type=int, default=3)
    parser.add_argument("--min_sentence_tokens", type=int, default=5)
    parser.add_argument("--min_overlap", type=float, default=0.45)
    parser.add_argument("--max_label_tokens", type=int, default=40)
    parser.add_argument("--min_label_tokens", type=int, default=5)
    parser.add_argument("--max_abstract_sentences", type=int, default=8)
    args = parser.parse_args()

    df = pd.read_csv(args.input_csv)

    required = ["query", "title", "abstract", "label"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")

    df = df.copy()
    df["query"] = df["query"].apply(clean_text)
    df["title"] = df["title"].apply(clean_text)
    df["abstract"] = df["abstract"].apply(clean_text)
    df["label"] = df["label"].apply(normalize_label)

    df["abstract_trimmed"] = df["abstract"].apply(lambda x: trim_abstract(x, args.max_abstract_sentences))
    df["label_tokens"] = df["label"].str.split().apply(len)
    df["overlap"] = [overlap_ratio(l, a) for l, a in zip(df["label"], df["abstract"])]

    oracle_texts = []
    oracle_scores = []
    for query, label, abstract in zip(df["query"], df["label"], df["abstract"]):
        if args.oracle_reference == "query_only":
            reference = query
        elif args.oracle_reference == "label":
            reference = label
        else:
            reference = f"{query} {label}"
        oracle, score = select_oracle_sentences(
            abstract,
            reference,
            max_sentences=args.oracle_sentences,
            min_sentence_tokens=args.min_sentence_tokens,
        )
        oracle_texts.append(oracle)
        oracle_scores.append(score)

    df["oracle_text"] = oracle_texts
    df["oracle_rougeL"] = oracle_scores

    before = len(df)
    df = df[(df["label_tokens"] >= args.min_label_tokens) & (df["label_tokens"] <= args.max_label_tokens)]
    df = df[df["overlap"] >= args.min_overlap]
    after = len(df)

    df["input_text"] = [
        build_prompt(q, t, a)
        for q, t, a in zip(df["query"], df["title"], df["abstract_trimmed"])
    ]
    if args.target_style == "label":
        df["target_text"] = df["label"]
    elif args.target_style == "bullets":
        df["target_text"] = df["oracle_text"].apply(to_bullets)
    else:
        df["target_text"] = df["oracle_text"]

    keep_cols = [
        "input_text",
        "target_text",
        "query",
        "title",
        "abstract_trimmed",
        "label",
        "oracle_text",
        "oracle_rougeL",
        "overlap",
        "label_tokens",
    ]
    df_out = df[keep_cols]

    os.makedirs(os.path.dirname(args.output_csv), exist_ok=True)
    df_out.to_csv(args.output_csv, index=False)

    print(f"Saved processed dataset: {args.output_csv}")
    print(f"Rows before: {before} | after: {after} | dropped: {before - after}")


if __name__ == "__main__":
    main()
