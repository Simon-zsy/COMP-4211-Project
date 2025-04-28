import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup
from tqdm import tqdm
from data_loader import prepare_data_loaders
from network import get_model
from torch.cuda.amp import autocast, GradScaler


MODEL_NAME = "microsoft/deberta-v3-large"

# Step 1: Load data loaders and tokenizer
# Set total batch size to 32, to be split across GPUs
total_batch_size = 32
train_loader, eval_loader, label2id, num_labels, tokenizer = prepare_data_loaders(
    data_path=".", 
    model_name=MODEL_NAME, 
    batch_size=total_batch_size
)

# Step 2: Load model
model = get_model(MODEL_NAME, num_labels)

# Step 3: Set up device and DataParallel
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
num_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 0
if num_gpus > 1:
    print(f"Using {num_gpus} GPUs with DataParallel. Total batch size: {total_batch_size}, per-GPU batch size: {total_batch_size // num_gpus}")
    model = nn.DataParallel(model)
else:
    print(f"Using single GPU or CPU. Total batch size: {total_batch_size}")
model.to(device)

# Step 4: Set up optimizer and scheduler
optimizer = AdamW(model.parameters(), lr=2e-5)
epochs = 3
total_steps = len(train_loader) * epochs
scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)
scaler = torch.amp.GradScaler(device)  # For mixed precision training

# Step 5: Training loop
# Initialize variables for tracking the best model
best_val_loss = float('inf')
best_model_state = None

for epoch in range(epochs):
    # Training
    model.train()
    total_train_loss = 0
    for batch in tqdm(train_loader, desc=f"Training Epoch {epoch+1}"):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
        
        with torch.amp.autocast(device):
            outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        loss = loss.mean()  # Reduce loss to scalar for multi-GPU
        total_train_loss += loss.item()  # Log scalar loss
        
        # loss.backward()
        # torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        # optimizer.step()
        # scheduler.step()
        # optimizer.zero_grad()
        
        scaler.scale(loss).backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()
        optimizer.zero_grad()
    
    print(f"Epoch {epoch+1}, Training Loss: {total_train_loss / len(train_loader)}")
    
    # Validation
    model.eval()
    total_eval_loss = 0
    with torch.no_grad():
        for batch in tqdm(eval_loader, desc=f"Validation Epoch {epoch+1}"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            loss = loss.mean()  # Reduce loss to scalar for multi-GPU
            total_eval_loss += loss.item()  # Log scalar loss
    
    avg_val_loss = total_eval_loss / len(eval_loader)
    print(f"Epoch {epoch+1}, Validation Loss: {avg_val_loss}")
    
    # If the current validation loss is lower than the previous best, save the model state
    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        # Save the state of the underlying model (not the DataParallel wrapper)
        best_model_state = model.module.state_dict().copy() if isinstance(model, nn.DataParallel) else model.state_dict().copy()
        print(f"New best model found at epoch {epoch+1} with validation loss: {avg_val_loss}")

# After training, load the best model state
if best_model_state is not None:
    if isinstance(model, nn.DataParallel):
        model.module.load_state_dict(best_model_state)
    else:
        model.load_state_dict(best_model_state)
    print(f"Loaded best model with validation loss: {best_val_loss}")

# Step 6: Save the best model and tokenizer
# Save the underlying model (not the DataParallel wrapper)
model_to_save = model.module if isinstance(model, nn.DataParallel) else model
model_to_save.save_pretrained(MODEL_NAME + "_best_model")
tokenizer.save_pretrained(MODEL_NAME + "_best_model")