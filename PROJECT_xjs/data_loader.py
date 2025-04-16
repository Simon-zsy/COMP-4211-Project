import pandas as pd
import os
from ast import literal_eval
from sklearn.model_selection import train_test_split
from datasets import Dataset
from transformers import AutoTokenizer

def load_data(file_path):
    """加载CSV数据并转换字符串列表为Python列表"""
    df = pd.read_csv(file_path)
    df['Sentence'] = df['Sentence'].apply(literal_eval)
    df['NER Tag'] = df['NER Tag'].apply(literal_eval)
    return df

def prepare_datasets(data_path, test_size=0.2, random_state=42):
    """加载并划分数据集为训练集和验证集"""
    train_df = load_data(os.path.join(data_path, "train.csv"))
    train_df, val_df = train_test_split(train_df, test_size=test_size, random_state=random_state)
    train_ds = Dataset.from_pandas(train_df)
    val_ds = Dataset.from_pandas(val_df)
    return train_ds, val_ds, train_df

def tokenize_and_align_labels(examples, tokenizer, label2id):
    """分词并对齐标签"""
    tokenized_inputs = tokenizer(
        examples["Sentence"],
        truncation=True,
        is_split_into_words=True,
        padding="max_length",
        max_length=128
    )

    labels = []
    for i, tags in enumerate(examples["NER Tag"]):
        word_ids = tokenized_inputs.word_ids(batch_index=i)
        previous_word_idx = None
        label_ids = []
        for word_idx in word_ids:
            if word_idx is None:
                label_ids.append(-100)
            elif word_idx != previous_word_idx:
                label_ids.append(label2id[tags[word_idx]])
            else:
                label_ids.append(-100)
            previous_word_idx = word_idx
        labels.append(label_ids)

    tokenized_inputs["labels"] = labels
    return tokenized_inputs

def preprocess_data(train_ds, val_ds, tokenizer, label2id):
    """应用分词和标签对齐"""
    tokenized_train = train_ds.map(tokenize_and_align_labels, batched=True, fn_kwargs={"tokenizer": tokenizer, "label2id": label2id})
    tokenized_val = val_ds.map(tokenize_and_align_labels, batched=True, fn_kwargs={"tokenizer": tokenizer, "label2id": label2id})
    return tokenized_train, tokenized_val