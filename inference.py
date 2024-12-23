import os
import time
import torch
import pandas as pd
from multiprocessing import Pool, cpu_count
from transformers import AutoTokenizer, T5ForConditionalGeneration
import argparse
import re

# Define paths and device
MODEL_SAVE_DIR = 'models/assets/t5_question_answering_model'
INFERENCE_DATA_PATH = 'data/input_datasets/inference_data.csv'
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_model_and_tokenizer():
    print("Loading the model and tokenizer...")
    model = T5ForConditionalGeneration.from_pretrained(MODEL_SAVE_DIR).to(DEVICE)
    tokenizer = AutoTokenizer.from_pretrained("t5-small")
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

def process_batch(batch_texts):
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

def generate_answers(input_texts, batch_size=20):
    start_time = time.time()

    batches = [input_texts[i:i + batch_size] for i in range(0, len(input_texts), batch_size)]
    with Pool(processes=cpu_count()) as pool:
        results = pool.map(process_batch, batches)
    answers = [answer for batch in results for answer in batch]

    elapsed_time = time.time() - start_time
    print(f"Inference completed in {elapsed_time:.2f} seconds for {len(input_texts)} inputs.")

    return answers


def main():
    print(f"Reading inference data from {INFERENCE_DATA_PATH}...")
    dataframe = pd.read_csv(INFERENCE_DATA_PATH)

    # Validate required columns
    if 'query' not in dataframe.columns or 'context' not in dataframe.columns:
        raise ValueError("The input CSV must contain 'query' and 'context' columns.")

    print(f"Loaded {len(dataframe)} rows from the inference dataset.")

    # Preprocess inputs
    print("Preprocessing inputs...")
    input_texts = preprocess_inputs(dataframe)

    # Run inference
    print("Running parallel inference...")
    global model, tokenizer
    model, tokenizer = load_model_and_tokenizer()
    generated_answers = generate_answers(input_texts, batch_size=20)

    # Print results
    print("\nGenerated Answers:")
    for i, (query, context, answer) in enumerate(zip(dataframe['query'], dataframe['context'], generated_answers), 1):
        print(f"\nQuestion {i}: {query}")
        print(f"Context {i}: {context}")
        print(f"Generated Answer {i}: {answer}")

if __name__ == "__main__":
    main()
