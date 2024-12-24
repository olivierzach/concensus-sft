import os
import re
import pandas as pd
import torch
from transformers import AutoTokenizer, T5ForConditionalGeneration, AdamW
from torch.utils.data import Dataset, DataLoader, random_split
from torch.optim.lr_scheduler import StepLR
from nltk.translate.bleu_score import sentence_bleu
from rouge_score import rouge_scorer
from sklearn.metrics import accuracy_score

# Paths
PATH_TO_TRAINING_DATA = '../data/input_datasets/training_data.csv'
SAVE_PATH = '../data/output_datasets/preprocessed_data.pkl'
MODEL_SAVE_DIR = 'assets/t5_question_answering_model'
CHECKPOINT_DIR = 'assets/checkpoints'

# Device setup
os.makedirs(MODEL_SAVE_DIR, exist_ok=True)
os.makedirs(CHECKPOINT_DIR, exist_ok=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load and preprocess data
def clean_text(text):
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def combine_columns(row, columns):
    return " ".join([f"{col.capitalize()}: {row[col]}" for col in columns if col in row and pd.notna(row[col])])

df = pd.read_csv(PATH_TO_TRAINING_DATA)
cols_to_clean = ['query', 'abstract', 'title', 'label']
for col in cols_to_clean:
    df[col] = df[col].apply(clean_text)

columns_to_combine = ['abstract']
df['context'] = df.apply(combine_columns, columns=columns_to_combine, axis=1)
df['input_text'] = df.apply(lambda row: f"Question: {row['query']} Context: {row['context']}", axis=1)

# Tokenizer
tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-small")
def tokenize_text(text, max_length):
    return tokenizer(
        text,
        truncation=True,
        padding='max_length',
        max_length=max_length,
    )

MAX_LENGTH = 128
df['tokenized_input'] = df['input_text'].apply(lambda x: tokenize_text(x, max_length=MAX_LENGTH))
df['tokenized_label'] = df['label'].apply(lambda x: tokenize_text(x, max_length=MAX_LENGTH))

df.to_pickle(SAVE_PATH)
print(f"Preprocessed dataset saved to: {SAVE_PATH}")

# Dataset class
class QuestionAnsweringDataset(Dataset):
    def __init__(self, dataframe):
        self.data = dataframe

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        tokenized_input = row['tokenized_input']
        tokenized_label = row['tokenized_label']
        return {
            'input_ids': torch.tensor(tokenized_input['input_ids']),
            'attention_mask': torch.tensor(tokenized_input['attention_mask']),
            'labels': torch.tensor(tokenized_label['input_ids']).masked_fill(
                torch.tensor(tokenized_label['attention_mask']) == 0, -100
            ),
        }

# Prepare dataset
df = pd.read_pickle(SAVE_PATH)
dataset = QuestionAnsweringDataset(df)

# Test a single item
sample = dataset[0]
print("Input IDs:", sample['input_ids'])
print("Attention Mask:", sample['attention_mask'])
print("Labels:", sample['labels'])

# Validate shapes
assert len(sample['input_ids']) == MAX_LENGTH, "Input IDs length mismatch"
assert len(sample['attention_mask']) == MAX_LENGTH, "Attention Mask length mismatch"
assert len(sample['labels']) == MAX_LENGTH, "Labels length mismatch"

# Validate alignment of padding
input_ids, attention_mask, labels = sample['input_ids'], sample['attention_mask'], sample['labels']
padding_positions = (input_ids == 0)
assert all(attention_mask[padding_positions] == 0), "Attention mask padding mismatch"
assert all(labels[padding_positions] == -100), "Labels padding mismatch"

# validation splits
train_size = int(0.8 * len(dataset))
val_size = int(0.1 * len(dataset))
test_size = len(dataset) - train_size - val_size
train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])

# create data loaders
train_loader = DataLoader(train_dataset, batch_size=12, shuffle=True, num_workers=8, pin_memory=True)
val_loader = DataLoader(val_dataset, batch_size=12, num_workers=8, pin_memory=True)
test_loader = DataLoader(test_dataset, batch_size=12, num_workers=8, pin_memory=True)

# Model setup
model = T5ForConditionalGeneration.from_pretrained("google/flan-t5-small").to(device)
optimizer = AdamW(model.parameters(), lr=1e-5)
scheduler = StepLR(optimizer, step_size=10, gamma=0.1)

# Save checkpoints
def save_checkpoint(epoch, model, optimizer, scheduler, checkpoint_dir):
    checkpoint_path = os.path.join(checkpoint_dir, f"checkpoint_epoch_{epoch}.pt")
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
    }, checkpoint_path)
    print(f"Checkpoint saved: {checkpoint_path}")

# Training and evaluation
def evaluate_model(loader, model):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for batch in loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            total_loss += outputs.loss.item()
    return total_loss / len(loader)

def predict(loader, model, tokenizer):
    model.eval()
    predictions = []
    references = []
    with torch.no_grad():
        for batch in loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels']

            outputs = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_length=512,
                num_beams=4,
                early_stopping=True
            )

            decoded_preds = [tokenizer.decode(output, skip_special_tokens=True) for output in outputs]
            decoded_labels = [
                tokenizer.decode(label[label != -100], skip_special_tokens=True)
                for label in labels
            ]

            predictions.extend(decoded_preds)
            references.extend(decoded_labels)

    return pd.DataFrame({"reference": references, "predicted": predictions})

def compute_metrics(df):
    references = df['reference'].tolist()
    predictions = df['predicted'].tolist()

    bleu_scores = [sentence_bleu([ref.split()], pred.split()) for ref, pred in zip(references, predictions)]
    avg_bleu = sum(bleu_scores) / len(bleu_scores)

    rouge = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    rouge_scores = [rouge.score(ref, pred) for ref, pred in zip(references, predictions)]
    avg_rouge = {
        key: sum(score[key].fmeasure for score in rouge_scores) / len(rouge_scores)
        for key in ['rouge1', 'rouge2', 'rougeL']
    }

    print(f"BLEU: {avg_bleu:.4f}")
    print(f"ROUGE: {avg_rouge}")

# Training loop
epochs = 5
best_val_loss = float("inf")
for epoch in range(1, epochs + 1):
    model.train()
    total_loss = 0
    print(f"\nStarting Epoch {epoch}...")
    
    for batch_idx, batch in enumerate(train_loader, start=1):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)

        optimizer.zero_grad()
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        # Verbose logging
        if batch_idx % 5 == 0 or batch_idx == len(train_loader):
            sample_input_ids = input_ids[0].tolist() 
            sample_labels = labels[0].tolist()

            decoded_input = tokenizer.decode(sample_input_ids, skip_special_tokens=True)
            decoded_label = tokenizer.decode([id for id in sample_labels if id != -100], skip_special_tokens=True)

            print(f"  Batch {batch_idx}/{len(train_loader)}: Loss={loss.item():.4f}")
            print(f"    Decoded Input: {decoded_input}")
            print(f"    Decoded Label: {decoded_label}")

    scheduler.step()
    train_loss = total_loss / len(train_loader)
    val_loss = evaluate_model(val_loader, model)

    print(f"Epoch {epoch}: Train Loss={train_loss:.4f}, Validation Loss={val_loss:.4f}")

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        save_checkpoint(epoch, model, optimizer, scheduler, CHECKPOINT_DIR)
        print(f"  Checkpoint saved for Epoch {epoch}")

# Save final model
model.save_pretrained(MODEL_SAVE_DIR)
tokenizer.save_pretrained(MODEL_SAVE_DIR)
print(f"Final model and tokenizer saved to {MODEL_SAVE_DIR}")

# Evaluation
results_df = predict(test_loader, model, tokenizer)
results_df.to_csv("test_predictions.csv", index=False)
compute_metrics(results_df)