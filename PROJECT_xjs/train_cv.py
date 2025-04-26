import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW
from transformers import  get_linear_schedule_with_warmup
from tqdm import tqdm
from data_loader import load_and_preprocess_data, tokenize_and_align_labels, create_label_mappings, NERDataset
from sklearn.model_selection import KFold
import pandas as pd
from datasets import Dataset as HFDataset
from network import get_model
from transformers import AutoTokenizer

# Step 1: Load tokenizer
tokenizer = AutoTokenizer.from_pretrained("bert-large-cased")

# Step 2: Load and preprocess full dataset
df = load_and_preprocess_data("train.csv")
# Create label mappings
label_list, label2id, id2label, class_weights = create_label_mappings(df)
# Convert to Hugging Face dataset
hf_ds = HFDataset.from_pandas(df)
# Tokenize and align labels
hf_ds = hf_ds.map(
    lambda examples: tokenize_and_align_labels(examples, tokenizer, label2id),
    batched=True,
    remove_columns=["Sentence", "NER Tag", "id"]
)

# Prepare K-Fold Cross-Validation
n_splits = 5
kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
batch_size = 32
epochs = 3
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Track metrics across folds
fold_losses = []
for fold, (train_idx, val_idx) in enumerate(kf.split(hf_ds), start=1):
    print(f"Starting fold {fold}/{n_splits}")
    # Create subsets for this fold
    train_sub = hf_ds.select(train_idx)
    val_sub = hf_ds.select(val_idx)
    train_dataset = NERDataset(train_sub, tokenizer)
    val_dataset = NERDataset(val_sub, tokenizer)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # Initialize model, optimizer, scheduler per fold
    model = get_model(len(label2id)).to(device)
    optimizer = AdamW(model.parameters(), lr=2e-5)
    total_steps = len(train_loader) * epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)

    best_val_loss = float('inf')
    for epoch in range(1, epochs+1):
        # Training
        model.train()
        train_loss = 0.0
        for batch in tqdm(train_loader, desc=f"Fold {fold} Train Epoch {epoch}"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            train_loss += loss.item()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
        avg_train = train_loss / len(train_loader)
        print(f"Fold {fold}, Epoch {epoch} Training Loss: {avg_train}")

        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f"Fold {fold} Val Epoch {epoch}"):
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['labels'].to(device)
                outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
                val_loss += outputs.loss.item()
        avg_val = val_loss / len(val_loader)
        print(f"Fold {fold}, Epoch {epoch} Validation Loss: {avg_val}")

        if avg_val < best_val_loss:
            best_val_loss = avg_val
            best_state = model.state_dict().copy()

    # Save best model for this fold
    model.load_state_dict(best_state)
    model.save_pretrained(f"bert_large_fold{fold}")
    tokenizer.save_pretrained(f"bert_large_fold{fold}")
    print(f"Fold {fold} best validation loss: {best_val_loss}")
    fold_losses.append(best_val_loss)

print(f"Cross-validation completed. Mean validation loss: {sum(fold_losses)/len(fold_losses)}")