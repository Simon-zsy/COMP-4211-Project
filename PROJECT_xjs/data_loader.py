import pandas as pd
from datasets import Dataset
from ast import literal_eval
import random
import os
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer
from network import create_label_mappings

def normalize_token(token):
    """Normalize non-standard characters to standard equivalents."""
    non_standard_map = {
        '\x92': "'s",
        '\xa0': ' ',
        '\x96': '-',
        '\x97': '-',
        '\x93': '"',
        '\x94': '"'
    }
    return non_standard_map.get(token, token)

def load_data(file_path):
    """Load and preprocess data, normalizing non-standard characters."""
    try:
        df = pd.read_csv(file_path)
        df['Sentence'] = df['Sentence'].apply(literal_eval)
        df['Sentence'] = df['Sentence'].apply(
            lambda x: [normalize_token(w) for w in x]
        )
        if 'NER Tag' in df.columns:
            df['NER Tag'] = df['NER Tag'].apply(literal_eval)
        return df
    except Exception as e:
        print(f"Failed to load data from {file_path}: {e}")
        raise

def augment_sentence(sentence, tags=None):
    """Data augmentation: add non-standard or normalized characters and swap entities."""
    sentence = sentence.copy()
    tags = tags.copy() if tags is not None else None
    
    if random.random() < 0.15:
        insert_idx = random.randint(0, len(sentence))
        char_options = ['\x92', "'s", '\xa0', ' ', '\x96', '-']
        sentence.insert(insert_idx, random.choice(char_options))
        if tags:
            tags.insert(insert_idx, 'O')
    
    if random.random() < 0.05 and tags:
        entity_types = ['per', 'geo', 'gpe', 'org', 'tim']
        for i, tag in enumerate(tags):
            if tag.startswith('B-') and tag[2:] in entity_types:
                sentence[i] = f"Entity_{tag[2:]}_{random.randint(1, 100)}"
                break
    
    return sentence, tags

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
                label_ids.append(label2id[tags[word_idx]])
            else:
                # For subwords, use the same label
                label_ids.append(label2id[tags[word_idx]])
            previous_word_idx = word_idx
        labels.append(label_ids)
    
    tokenized_inputs["labels"] = labels
    return tokenized_inputs

def prepare_datasets(data_path, model_name="bert-large-cased", validation_split=0.2):
    """Prepare tokenized training, validation, and test datasets from a single file."""
    try:
        # 加载单一的训练文件
        full_df = load_data(f"{data_path}/train.csv")
        
        # 拆分为训练集和验证集
        train_df, eval_df = train_test_split(full_df, test_size=validation_split, random_state=42)
        train_df = train_df.reset_index(drop=True)
        eval_df = eval_df.reset_index(drop=True)
        
        print(f"Split data: {len(train_df)} training samples, {len(eval_df)} validation samples")
        
        # 应用数据增强（仅训练集）
        augmented_rows = []
        for _, row in train_df.iterrows():
            sentence, tags = augment_sentence(row['Sentence'], row['NER Tag'])
            augmented_rows.append({'Sentence': sentence, 'NER Tag': tags})
        
        # 确保增强后的数据格式与原始数据一致
        aug_df = pd.DataFrame(augmented_rows)
        train_df = pd.concat([train_df, aug_df], ignore_index=True)
        print(f"After augmentation: {len(train_df)} training samples")
        
        # 转换为Dataset对象
        train_ds = Dataset.from_pandas(train_df)
        eval_ds = Dataset.from_pandas(eval_df)
        
        # 加载 tokenizer 和标签映射
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        _, label2id, _, _ = create_label_mappings(train_df)
        
        # 标记化和对齐标签
        train_ds = train_ds.map(
            lambda examples: tokenize_and_align_labels(examples, tokenizer, label2id),
            batched=True,
            remove_columns=["Sentence", "NER Tag"]
        )
        eval_ds = eval_ds.map(
            lambda examples: tokenize_and_align_labels(examples, tokenizer, label2id),
            batched=True,
            remove_columns=["Sentence", "NER Tag"]
        )
        
        return train_ds, eval_ds, train_df
    except Exception as e:
        print(f"Failed to prepare datasets: {e}")
        raise