import torch
from torch.utils.data import DataLoader
from torch.utils.data import DataLoader
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup
from tqdm import tqdm
from data_loader import prepare_data_loaders
from network import get_model
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch.nn.init as init

# Step 1: Load tokenizer
tokenizer = AutoTokenizer.from_pretrained("0.864020")

# Step 2: Load data loaders
train_loader, eval_loader, label2id, num_labels = prepare_data_loaders(data_path=".", model_name="0.864020", batch_size=32)

# Step 3: Load model
model = get_model(num_labels)

# Freeze BERT weights
for param in model.bert.parameters():
    param.requires_grad = False

# Initialize classifier weights with Kaiming initialization
if hasattr(model, 'classifier'):
    if hasattr(model.classifier, 'weight'):
        init.kaiming_normal_(model.classifier.weight, nonlinearity='relu')
    if hasattr(model.classifier, 'bias') and model.classifier.bias is not None:
        init.zeros_(model.classifier.bias)
elif hasattr(model, 'out_proj'):  # Some models use out_proj
    if hasattr(model.out_proj, 'weight'):
        init.kaiming_normal_(model.out_proj.weight, nonlinearity='relu')
    if hasattr(model.out_proj, 'bias') and model.out_proj.bias is not None:
        init.zeros_(model.out_proj.bias)
# Add more conditions if your model has a different classifier structure

# Step 4: Set up optimizer and scheduler
optimizer = AdamW(model.parameters(), lr=2e-5)
epochs = 10
total_steps = len(train_loader) * epochs
scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=100, num_training_steps=total_steps)

# Step 5: Training loop
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

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
        
        outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        total_train_loss += loss.item()
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
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
            total_eval_loss += loss.item()
    
    avg_val_loss = total_eval_loss / len(eval_loader)
    print(f"Epoch {epoch+1}, Validation Loss: {avg_val_loss}")
    
    # If the current validation loss is lower than the previous best, save the model state
    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        best_model_state = model.state_dict().copy()
        print(f"New best model found at epoch {epoch+1} with validation loss: {avg_val_loss}")

# After training, load the best model state
if best_model_state is not None:
    model.load_state_dict(best_model_state)
    print(f"Loaded best model with validation loss: {best_val_loss}")

# Step 6: Save the best model
model.save_pretrained("train_bert_2e-5")
tokenizer.save_pretrained("train_bert_2e-5")