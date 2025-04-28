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

def correct_mistags(df):
    """Correct mislabeled 'O' tags based on most frequent entity annotations."""
    from collections import defaultdict, Counter
    # Define tokens to skip auto-replacement (common stopwords/punctuation)
    skip_tokens = { 'the','a','an','and','of','to','in','for','with','on','that','from',',','.',';','-','–','—','\'' }
    # Minimum number of entity occurrences required for replacement
    min_entity_count = 2
    # Build mapping from token to counts of O vs entity tags
    word_tag_counts = defaultdict(Counter)
    word_o_counts = defaultdict(int)
    for tokens, tags in zip(df['Sentence'], df['NER Tag']):
        for token, tag in zip(tokens, tags):
            if tag == 'O':
                word_o_counts[token] += 1
            else:
                word_tag_counts[token][tag] += 1
    # Replace 'O' with most common entity tag when available
    corrected = []
    for tokens, tags in zip(df['Sentence'], df['NER Tag']):
        new_tags = []
        for token, tag in zip(tokens, tags):
            # Skip auto-correction for common tokens
            if token.lower() in skip_tokens:
                new_tags.append(tag)
                continue
            if tag == 'O' and token in word_tag_counts and word_tag_counts[token]:
                # only replace if the top entity tag occurs more often than O and has sufficient frequency
                top_tag, top_count = word_tag_counts[token].most_common(1)[0]
                if top_count > word_o_counts.get(token, 0) and top_count >= min_entity_count:
                    new_tags.append(top_tag)
                else:
                    new_tags.append(tag)
            else:
                new_tags.append(tag)
        corrected.append(new_tags)
    df['NER Tag'] = corrected
    return df

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
        
        # 这里假设example中已经包含input_ids, attention_mask和labels
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

def load_and_preprocess_data(file_path, apply_mistag_correction=False):
    """Load and preprocess data, normalizing non-standard characters."""
    try:
        df = pd.read_csv(file_path)
        df['Sentence'] = df['Sentence'].apply(literal_eval)
        df['Sentence'] = df['Sentence'].apply(lambda x: [normalize_token(w.strip()) for w in x if w.strip()])
        df['NER Tag'] = df['NER Tag'].apply(literal_eval)
        # Correct mislabeled 'O' tags based on dataset-wide entity occurrences
        
        if apply_mistag_correction:
            df = correct_mistags(df)
        
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
    
    tokenized_inputs["labels"] = labels
    return tokenized_inputs

def prepare_data_loaders(data_path, model_name="bert-large-cased", batch_size=32, 
                         validation_split=0.2, apply_mistag_correction=False):
    """Prepare PyTorch DataLoaders for training and validation."""
    try:
        # Load and preprocess data
        full_df = load_and_preprocess_data(f"{data_path}/train.csv", 
                                          apply_mistag_correction=apply_mistag_correction)
        
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
        train_dataset = NERDataset(train_ds, tokenizer)
        eval_dataset = NERDataset(eval_ds, tokenizer)
        
        # Create DataLoaders
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        eval_loader = DataLoader(eval_dataset, batch_size=batch_size, shuffle=False)
        
        return train_loader, eval_loader, label2id, len(label2id), tokenizer
    except Exception as e:
        print(f"Failed to prepare data loaders: {e}")
        raise

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Test correct_mistags and log changed tags.")
    parser.add_argument("--file", type=str, default="train.csv", help="Path to input CSV file")
    args = parser.parse_args()
    # Load raw data
    df = pd.read_csv(args.file)
    df['Sentence'] = df['Sentence'].apply(literal_eval)
    df['NER Tag'] = df['NER Tag'].apply(literal_eval)
    # Keep original tags and record ids for comparison
    orig_tags = [list(tags) for tags in df['NER Tag']]
    orig_ids = df['id'].tolist()
    # Apply correction
    corr_df = correct_mistags(df.copy())
    new_tags = corr_df['NER Tag']
    # Collect changes
    changes = []
    for idx, (tokens, o_tags, n_tags) in enumerate(zip(df['Sentence'], orig_tags, new_tags)):
        sent_id = orig_ids[idx]
        for token, o, n in zip(tokens, o_tags, n_tags):
            if o == 'O' and n != 'O':
                changes.append(f"sentence id: {sent_id}, replaced the tag for '{token}' with '{n}'")
    # Write to log file
    log_file = "changed_tags.txt"
    with open(log_file, "w", encoding="utf-8") as f:
        f.write("\n".join(changes))
    print(f"Logged {len(changes)} changes to '{log_file}'")