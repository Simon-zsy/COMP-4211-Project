import pandas as pd
from datasets import Dataset as HFDataset
from ast import literal_eval
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer
from torch.utils.data import Dataset, DataLoader
import torch
from collections import Counter

def create_label_mappings(train_df):
    """Generate label mappings from training data."""
    all_tags = set()
    for tags in train_df['NER Tag']:
        tags = eval(tags) if isinstance(tags, str) else tags
        all_tags.update(tags)
    
    label_list = sorted(list(all_tags))
    if 'O' not in label_list:
        label_list.insert(0, 'O')
    
    label2id = {label: idx for idx, label in enumerate(label_list)}
    id2label = {idx: label for label, idx in label2id.items()}
    
    # Compute class weights for imbalanced labels
    tag_counts = Counter()
    for tags in train_df['NER Tag']:
        tags = eval(tags) if isinstance(tags, str) else tags
        tag_counts.update(tags)
    
    total_tags = sum(tag_counts.values())
    weights = {label: total_tags / (len(label_list) * tag_counts.get(label, 1)) for label in label_list}
    class_weights = torch.tensor([weights.get(label, 1.0) for label in label_list], dtype=torch.float)
    
    return label_list, label2id, id2label, class_weights

def normalize_token(token):
    """Normalize non-standard characters to standard equivalents or a placeholder."""
    non_standard_map = {
        '\x85': ' ',  # Next Line (NEL)
        '\x91': "'",  # Left single quote
        '\x92': "'s", # Right single quote (used as possessive)
        '\x93': '"',  # Left double quote
        '\x94': '"',  # Right double quote
        '\x95': '•',  # Bullet
        '\x96': '-',  # En dash
        '\x97': '-',  # Em dash
        '\x99': '™',  # Trademark
        '\xa0': ' ',  # Non-breaking space
        '\x9c': ' ',  # String terminator
        '\x9d': ' ',  # Operating system command
        '\x9e': ' ',  # Privacy message
        '\x9f': ' ',  # Application program command
        '\u00AD': '-', # Soft hyphen
        '\u200B': ' ', # Zero-width space
        '\u200C': '',  # Zero-width non-joiner
        '\u200D': '',  # Zero-width joiner
        '\u202F': ' '  # Narrow non-breaking space
    }
    
    # Initialize list to log unhandled characters
    unhandled_chars = []
    
    # Replace each character in the token
    normalized_token = ''
    for char in token:
        if char in non_standard_map:
            normalized_token += non_standard_map[char]
        elif ord(char) < 32 or ord(char) >= 127:  # Control or non-ASCII characters
            unhandled_chars.append((char, ord(char)))
            normalized_token += '[UNK]'  # Replace with placeholder
        else:
            normalized_token += char
    
    # Log unhandled characters if any
    if unhandled_chars:
        with open("unhandled_tokens.txt", "a") as f:
            f.write(f"Token: {token}, Unhandled chars: {unhandled_chars}\n")
    
    return normalized_token if normalized_token.strip() else '[UNK]'  # Avoid empty tokens

class NERDataset(Dataset):
    def __init__(self, hf_dataset, tokenizer, max_len=256):
        """
        Args:
            hf_dataset: HuggingFace Dataset 对象，已经过tokenize_and_align_labels处理
            tokenizer: HuggingFace tokenizer
            max_len: 最大序列长度
        """
        self.dataset = hf_dataset
        self.tokenizer = tokenizer
        self.max_len = max_len
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        # 直接使用预处理过的数据，不再需要处理原始的句子和标签
        example = self.dataset[idx]
        
        # 这里假设example中已经包含input_ids, attention_mask和label_ids
        input_ids = torch.tensor(example['input_ids'], dtype=torch.long)
        attention_mask = torch.tensor(example['attention_mask'], dtype=torch.long)
        
        # 确保labels键存在，否则尝试使用label_ids
        if 'labels' in example:
            labels = torch.tensor(example['labels'], dtype=torch.long)
        elif 'label_ids' in example:
            labels = torch.tensor(example['label_ids'], dtype=torch.long)
        else:
            raise KeyError("Neither 'labels' nor 'label_ids' found in dataset")
        
        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': labels
        }

def load_and_preprocess_data(file_path):
    """Load and preprocess data, normalizing non-standard characters."""
    try:
        df = pd.read_csv(file_path)
        df['Sentence'] = df['Sentence'].apply(literal_eval)
        df['Sentence'] = df['Sentence'].apply(lambda x: [normalize_token(w.strip()) for w in x if w.strip()])
        df['NER Tag'] = df['NER Tag'].apply(literal_eval)
        
        # Validate lengths and collect valid rows
        valid_rows = []
        skipped_rows = []
        for idx, row in df.iterrows():
            if len(row['Sentence']) != len(row['NER Tag']):
                skipped_rows.append(f"Index {idx}: Sentence length {len(row['Sentence'])}, Tag length {len(row['NER Tag'])}, Sentence: {row['Sentence']}, Tags: {row['NER Tag']}")
            else:
                valid_rows.append(row)
        
        # Log skipped rows
        if skipped_rows:
            with open("skipped_rows.txt", "w") as f:
                f.write("\n".join(skipped_rows))
            print(f"Skipped {len(skipped_rows)} rows with mismatched lengths. Details in 'skipped_rows.txt'")
        
        # Create new DataFrame with valid rows
        if not valid_rows:
            raise ValueError("No valid rows remaining after filtering mismatches")
        valid_df = pd.DataFrame(valid_rows).reset_index(drop=True)
        
        return valid_df
    except Exception as e:
        print(f"Failed to load data from {file_path}: {e}")
        raise

def tokenize_and_align_labels(examples, tokenizer, label2id):
    """Tokenize sentences and align NER tags."""
    tokenized_inputs = tokenizer(
        examples["Sentence"],
        truncation=True,
        is_split_into_words=True,
        padding="max_length",
        max_length=256,
        return_special_tokens_mask=True
    )
    
    labels = []
    for i, tags in enumerate(examples["NER Tag"]):
        word_ids = tokenized_inputs.word_ids(batch_index=i)
        label_ids = []
        previous_word_idx = None
        for word_idx in word_ids:
            if word_idx is None:
                label_ids.append(-100)  # Special tokens
            elif word_idx != previous_word_idx:
                if word_idx < len(tags):
                    label_ids.append(label2id[tags[word_idx]])
                else:
                    label_ids.append(-100)  # Fallback
            else:
                label_ids.append(-100)  # Ignore subwords
            previous_word_idx = word_idx
        labels.append(label_ids)
    
    tokenized_inputs["label_ids"] = labels
    return tokenized_inputs

def prepare_data_loaders(data_path, model_name="bert-base-cased", batch_size=32, validation_split=0.2):
    """Prepare PyTorch DataLoaders for training and validation."""
    try:
        # Load and preprocess data
        full_df = load_and_preprocess_data(f"{data_path}/train.csv")
        
        # Split into train and validation
        train_df, eval_df = train_test_split(full_df, test_size=validation_split, random_state=42)
        train_df = train_df.reset_index(drop=True)
        eval_df = eval_df.reset_index(drop=True)
        
        print(f"Split data: {len(train_df)} training samples, {len(eval_df)} validation samples")
        
        # Convert to Hugging Face Dataset
        train_ds = HFDataset.from_pandas(train_df)
        eval_ds = HFDataset.from_pandas(eval_df)
        
        # Load tokenizer and label mappings
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        _, label2id, _, _ = create_label_mappings(train_df)
        
        # Tokenize and align labels
        train_ds = train_ds.map(
            lambda examples: tokenize_and_align_labels(examples, tokenizer, label2id),
            batched=True,
            remove_columns=["Sentence", "NER Tag", "id"]  # 移除原始列
        )
        eval_ds = eval_ds.map(
            lambda examples: tokenize_and_align_labels(examples, tokenizer, label2id),
            batched=True,
            remove_columns=["Sentence", "NER Tag", "id"]
        )
        
        # 直接使用预处理后的数据创建PyTorch datasets
        # 注意这里不再需要使用tokenizer进行处理
        train_dataset = NERDataset(train_ds, tokenizer)
        eval_dataset = NERDataset(eval_ds, tokenizer)
        
        # Create DataLoaders
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        eval_loader = DataLoader(eval_dataset, batch_size=batch_size, shuffle=False)
        
        return train_loader, eval_loader, label2id, len(label2id)
    except Exception as e:
        print(f"Failed to prepare data loaders: {e}")
        raise