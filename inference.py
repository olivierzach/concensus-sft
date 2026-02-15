import argparse
import time
import re

import pandas as pd
import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

# Define paths and device
MODEL_SAVE_DIR = 'outputs/consensus_clean/flan_t5_small_clean_end_es_rouge_long/best_checkpoint'
INFERENCE_DATA_PATH = 'data/input_datasets/inference_data.csv'
DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

def load_model_and_tokenizer(model_path: str):
    print("Loading the model and tokenizer...")
    model = AutoModelForSeq2SeqLM.from_pretrained(model_path).to(DEVICE)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model.eval()
    print("Model and tokenizer loaded successfully.")
    return model, tokenizer

def clean_text(text):
    text = text.lower()
    text = re.sub(r"[^\w\s]", "", text) 
    text = re.sub(r"\s+", " ", text).strip()
    return text

def preprocess_inputs(dataframe):
    formatted_inputs = [
        f"Question: {clean_text(row['query'])} Context: {clean_text(row['context'])}"
        for _, row in dataframe.iterrows()
    ]
    return formatted_inputs

def process_batch(batch_texts, model, tokenizer):
    inputs = tokenizer(
        batch_texts,
        padding=True,
        truncation=True,
        max_length=512,
        return_tensors="pt"
    ).to(DEVICE)

    with torch.no_grad():
        outputs = model.generate(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            max_length=128
        )

    return [tokenizer.decode(output, skip_special_tokens=True) for output in outputs]

def generate_answers(input_texts, model, tokenizer, batch_size=20):
    start_time = time.time()

    batches = [input_texts[i:i + batch_size] for i in range(0, len(input_texts), batch_size)]
    answers = []
    for batch in batches:
        answers.extend(process_batch(batch, model, tokenizer))

    elapsed_time = time.time() - start_time
    print(f"Inference completed in {elapsed_time:.2f} seconds for {len(input_texts)} inputs.")

    return answers


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", default=MODEL_SAVE_DIR)
    parser.add_argument("--input_csv", default=INFERENCE_DATA_PATH)
    parser.add_argument("--batch_size", type=int, default=20)
    args = parser.parse_args()

    print(f"Reading inference data from {args.input_csv}...")
    dataframe = pd.read_csv(args.input_csv)

    # Validate required columns
    if 'query' not in dataframe.columns or 'context' not in dataframe.columns:
        raise ValueError("The input CSV must contain 'query' and 'context' columns.")

    print(f"Loaded {len(dataframe)} rows from the inference dataset.")

    # Preprocess inputs
    print("Preprocessing inputs...")
    input_texts = preprocess_inputs(dataframe)

    # Run inference
    print("Running parallel inference...")
    model, tokenizer = load_model_and_tokenizer(args.model_path)
    generated_answers = generate_answers(input_texts, model, tokenizer, batch_size=args.batch_size)

    # Print results
    print("\nGenerated Answers:")
    for i, (query, context, answer) in enumerate(zip(dataframe['query'], dataframe['context'], generated_answers), 1):
        print(f"\nQuestion {i}: {query}")
        print(f"Context {i}: {context}")
        print(f"Generated Answer {i}: {answer}")

if __name__ == "__main__":
    main()
