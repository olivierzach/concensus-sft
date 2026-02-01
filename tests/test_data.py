import pandas as pd

from consensus_sft.data import build_input_text, load_raw_dataframe


def test_build_input_text():
    row = {"query": "Q", "title": "T", "abstract": "A"}
    text = build_input_text(row, ["query", "title", "abstract"])
    assert "Question:" in text
    assert "Title:" in text
    assert "Context:" in text


def test_load_raw_dataframe_from_raw(tmp_path):
    df = pd.DataFrame(
        {
            "query": ["Q"],
            "title": ["T"],
            "abstract": ["A"],
            "label": ["L"],
        }
    )
    path = tmp_path / "data.csv"
    df.to_csv(path, index=False)
    out = load_raw_dataframe(str(path), ["query", "title", "abstract"], "label")
    assert "input_text" in out.columns
    assert "target_text" in out.columns
