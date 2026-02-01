import argparse

import onnxruntime as ort
from transformers import AutoTokenizer


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--text", required=True)
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    session = ort.InferenceSession(f"{args.model_path}/model.onnx")
    inputs = tokenizer([args.text], return_tensors="np", padding=True, truncation=True)

    ort_inputs = {k: v for k, v in inputs.items() if k in {"input_ids", "attention_mask"}}
    outputs = session.run(None, ort_inputs)
    print(outputs)


if __name__ == "__main__":
    main()
