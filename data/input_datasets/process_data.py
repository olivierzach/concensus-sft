import pandas as pd 
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import re 
import pickle

# load the training data - do some manual inspection
PATH_TO_TRAINING_DATA ='input_datasets/training_data.csv'
df = pd.read_csv(PATH_TO_TRAINING_DATA)
print(df.head())
print(df.shape)
print(df.columns)

def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)  
    text = re.sub(r'\s+', ' ', text).strip()
    return text

cols_to_clean = ['query', 'abstract', 'title', 'label']
for col in cols_to_clean:
    df[col] = df[col].apply(clean_text)
print(df[['query', 'abstract', 'title', 'label']].head())

def combine_columns(row, columns):
    """
    Combines specified columns into a single string dynamically.
    """
    return " ".join([f"{col.capitalize()}: {row[col]}" for col in columns if col in row and pd.notna(row[col])])

columns_to_combine = ['query', 'title', 'abstract']
df['input_text'] = df.apply(combine_columns, columns=columns_to_combine, axis=1)

# Verify the result
print(df[['title', 'query', 'abstract']])
print(df['input_text'].head())


tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
def tokenize_text(input_text, max_length=512):
    """
    Tokenizes the input text using the Hugging Face tokenizer.
    Returns a dictionary with input_ids and attention_mask.
    """
    return tokenizer(
        input_text,
        truncation=True,
        padding='max_length',
        max_length=max_length,
    )  

df['tokenized_input'] = df['input_text'].apply(tokenize_text)
df['tokenized_label'] = df['label'].apply(tokenize_text, max_length=128)

SAVE_PATH = 'output_datasets/preprocessed_data.pkl'
df.to_pickle(SAVE_PATH)
print(f"Preprocessed dataset saved to: {SAVE_PATH}")




