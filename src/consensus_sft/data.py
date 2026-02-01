import re
from typing import Dict, List

import pandas as pd
from datasets import Dataset, DatasetDict


def clean_text(text: str) -> str:
    text = "" if text is None else str(text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _label_for_column(column: str) -> str:
    column = column.lower()
    if column == "query":
        return "Question"
    if column == "abstract":
        return "Context"
    if column == "title":
        return "Title"
    return column.capitalize()


def build_input_text(row: Dict[str, str], text_columns: List[str]) -> str:
    parts = []
    for col in text_columns:
        if col not in row:
            continue
        value = row[col]
        if value is None or (isinstance(value, float) and pd.isna(value)):
            continue
        value = clean_text(value)
        if not value:
            continue
        parts.append(f"{_label_for_column(col)}: {value}")
    return " ".join(parts)


def load_raw_dataframe(path: str, text_columns: List[str], label_column: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "input_text" in df.columns and "target_text" in df.columns:
        return df[["input_text", "target_text"]].copy()
    missing = [c for c in text_columns + [label_column] if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in CSV: {missing}")
    df = df.copy()
    df["input_text"] = df.apply(lambda row: build_input_text(row, text_columns), axis=1)
    df["target_text"] = df[label_column].apply(clean_text)
    return df[["input_text", "target_text"]]


def split_dataset(
    df: pd.DataFrame,
    seed: int,
    train_frac: float,
    val_frac: float,
    test_frac: float,
) -> DatasetDict:
    if round(train_frac + val_frac + test_frac, 6) != 1.0:
        raise ValueError("train/val/test fractions must sum to 1.0")
    dataset = Dataset.from_pandas(df, preserve_index=False)
    temp_split = dataset.train_test_split(test_size=val_frac + test_frac, seed=seed)
    val_ratio = val_frac / (val_frac + test_frac)
    val_test = temp_split["test"].train_test_split(test_size=1 - val_ratio, seed=seed)
    return DatasetDict(
        train=temp_split["train"],
        validation=val_test["train"],
        test=val_test["test"],
    )


def tokenize_splits(
    dataset_dict: DatasetDict,
    tokenizer,
    max_source_length: int,
    max_target_length: int,
) -> DatasetDict:
    def tokenize_batch(batch):
        model_inputs = tokenizer(
            batch["input_text"],
            max_length=max_source_length,
            truncation=True,
        )
        labels = tokenizer(
            text_target=batch["target_text"],
            max_length=max_target_length,
            truncation=True,
        )
        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    return dataset_dict.map(tokenize_batch, batched=True, remove_columns=["input_text", "target_text"])
