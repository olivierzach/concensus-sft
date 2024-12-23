import os
import pandas as pd
import re
import torch
from transformers import AutoTokenizer, T5ForConditionalGeneration, AdamW
from torch.utils.data import Dataset, DataLoader, random_split
from torch.optim.lr_scheduler import StepLR
from utils.build_torch_dataset import QuestionAnsweringDataset

PATH_TO_TRAINING_DATA = '../data/output_datasets/preprocessed_data.pkl'
MODEL_SAVE_DIR = 'assets/t5_question_answering_model'
CHECKPOINT_DIR = 'assets/checkpoints'
os.makedirs(MODEL_SAVE_DIR, exist_ok=True)
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# load the training data
df = pd.read_pickle(PATH_TO_TRAINING_DATA)
print(df.shape)
print(df.columns)
dataset = QuestionAnsweringDataset(df)

# validation splits
train_size = int(0.8 * len(dataset))  
val_size = int(0.1 * len(dataset))  
test_size = len(dataset) - train_size - val_size  
train_dataset, val_dataset, test_dataset = random_split(
    dataset, [train_size, val_size, test_size]
)
print(f"Dataset split: Train={len(train_dataset)}, Validation={len(val_dataset)}, Test={len(test_dataset)}")

# create data loaders 
train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=8)
test_loader = DataLoader(test_dataset, batch_size=8)

# load model and optimizer
model = T5ForConditionalGeneration.from_pretrained("t5-small").to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)
scheduler = StepLR(optimizer, step_size=10, gamma=0.1)

# save checkpoints
def save_checkpoint(epoch, model, optimizer, scheduler, checkpoint_dir):
    checkpoint_path = os.path.join(checkpoint_dir, f"checkpoint_epoch_{epoch}.pt")
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
    }, checkpoint_path)
    print(f"Checkpoint saved: {checkpoint_path}")


def evaluate_model(loader, model, device):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for batch in loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            total_loss += outputs.loss.item()

    avg_loss = total_loss / len(loader)
    return avg_loss

# training loop - slow because GPU poor...
epochs = 3
best_val_loss = float("inf")
print(f"Starting training for {epochs} epochs...\n")
for epoch in range(1, epochs + 1):
    print(f"Epoch {epoch}/{epochs}")
    print("-" * 40)

    model.train()
    total_loss = 0 
    
    for batch_idx, batch in enumerate(train_loader):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)

        # forward pass
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss 
        total_loss += loss.item()

        # backward pass
        optimizer.zero_grad()  
        loss.backward() 
        optimizer.step()

        # log progress 
        if (batch_idx + 1) % 5 == 0 or (batch_idx + 1) == len(train_loader):
            print(f"  Batch {batch_idx + 1}/{len(train_loader)}, Loss: {loss.item():.4f}")

    scheduler.step()

    # train performance
    avg_train_loss = total_loss / len(train_loader)
    print(f"\nEpoch {epoch} Training Complete - Average Loss: {avg_train_loss:.4f}\n")

    # validation performance
    print("Evaluating on validation set...")
    val_loss = evaluate_model(val_loader, model, device)
    print(f"Epoch {epoch} Validation Loss: {val_loss:.4f}\n")

    # save checkpoints
    if val_loss < best_val_loss:
        print("Validation loss improved. Saving checkpoint...")
        best_val_loss = val_loss
        save_checkpoint(epoch, model, optimizer, scheduler, CHECKPOINT_DIR)
    else:
        print("Validation loss did not improve. No checkpoint saved.")

# save final model
model.save_pretrained(MODEL_SAVE_DIR)
print(f"Final model saved to {MODEL_SAVE_DIR}")

# evaluate on test set
test_loss = evaluate_model(test_loader, model, device)
print(f"Test Loss: {test_loss:.4f}")
