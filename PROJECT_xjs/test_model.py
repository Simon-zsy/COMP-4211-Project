import os
import pandas as pd
import numpy as np
from ast import literal_eval
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForTokenClassification, Trainer
from network import create_label_mappings
from data_loader import load_data, normalize_token, prepare_datasets

# 配置
DATASET_PATH = "/localdata/szhoubx/rm/connext-backup/PROJECT_xjs"
MODEL_PATH = "/localdata/szhoubx/rm/connext-backup/PROJECT_xjs/results/checkpoint-24000"  # Matches saved model directory
OUTPUT_FILE = "./submission.csv"
DEBUG_IDS = [1316]  # For Sentence 1 and 2 (update with correct IDs if known)

def predict_and_save(model_path, test_ds, train_df, output_file=OUTPUT_FILE):
    try:
        model = AutoModelForTokenClassification.from_pretrained(model_path)
        # Load tokenizer from bert-base-cased, as it was used during training
        tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")
    except Exception as e:
        print(f"Failed to load model or tokenizer: {e}")
        raise

    try:
        label_list, label2id, id2label, _ = create_label_mappings(train_df)
    except Exception as e:
        print(f"Failed to create label mappings: {e}")
        id2label = model.config.id2label
        label2id = model.config.label2id
        label_list = list(label2id.keys())

    def tokenize(examples):
        tokenized = tokenizer(
            examples["Sentence"],
            truncation=True,
            is_split_into_words=True,
            padding="max_length",
            max_length=256,
            return_special_tokens_mask=True
        )
        return tokenized

    try:
        tokenized_test = test_ds.map(tokenize, batched=True, remove_columns=["Sentence"])
    except Exception as e:
        print(f"Failed to tokenize test dataset: {e}")
        raise

    trainer = Trainer(
        model=model,
        tokenizer=tokenizer,
        args=None,
    )

    try:
        predictions = trainer.predict(tokenized_test)
        if isinstance(predictions.predictions, list):  # CRF decoded tags
            preds = predictions.predictions
        else:  # Logits
            preds = np.argmax(predictions.predictions, axis=2)
    except Exception as e:
        print(f"Prediction failed: {e}")
        raise

    final_preds = []
    for i in range(len(tokenized_test)):
        input_ids = tokenized_test[i]["input_ids"]
        special_tokens_mask = tokenized_test[i]["special_tokens_mask"]
        sentence = test_ds[i]["Sentence"]
        sentence_length = len(sentence)

        encoding = tokenizer(
            sentence,
            truncation=True,
            is_split_into_words=True,
            padding=False,
            max_length=256,
            return_special_tokens_mask=True
        )
        word_ids = encoding.word_ids()
        tokens = tokenizer.convert_ids_to_tokens(encoding["input_ids"])

        debug = (test_ds[i]["id"] in DEBUG_IDS or
                 any(w in ['\xa0', '\x92', '\x96', ''] for w in sentence))
        if debug:
            print(f"ID {test_ds[i]['id']} Sentence: {sentence}")
            print(f"Sentence length: {sentence_length}")
            print(f"Tokens: {tokens}")
            print(f"Word IDs: {word_ids}")

        if isinstance(preds[i], list):  # CRF decoded tags
            pred_labels = [id2label[tag] for tag in preds[i]]
        else:  # Logits-based predictions
            pred_labels = [id2label[tag] for tag in preds[i]]

        word_to_tag = {}
        for idx, (word_idx, pred) in enumerate(zip(word_ids, pred_labels)):
            if word_idx is None:
                continue
            if word_idx not in word_to_tag:
                word_to_tag[word_idx] = pred

        word_labels = []
        for word_idx in range(sentence_length):
            normalized = normalize_token(sentence[word_idx])
            if normalized in ["'s", ' ', '-', '"', ''] or not normalized.strip():
                word_labels.append("O")
            else:
                word_labels.append(word_to_tag.get(word_idx, "O"))

        if debug:
            print(f"Predicted tags: {word_labels}")
            print(f"Predicted tags length: {len(word_labels)}")

        final_preds.append(word_labels)

    try:
        test_df = test_ds.to_pandas()
        test_df["NER Tag"] = final_preds
        test_df["NER Tag"] = test_df["NER Tag"].apply(lambda x: str(x))
        for idx, row in test_df.iterrows():
            sentence = literal_eval(row["Sentence"]) if isinstance(row["Sentence"], str) else row["Sentence"]
            predicted_tags = literal_eval(row["NER Tag"])
            if len(predicted_tags) != len(sentence):
                print(f"Warning: ID {row['id']} has mismatched lengths: "
                      f"Predicted {len(predicted_tags)}, Sentence {len(sentence)}")
        submission_df = test_df[["id", "NER Tag"]]
        submission_df.to_csv(output_file, index=False)
        print(f"Submission saved to {output_file}")
        backup_path = "/localdata/szhoubx/rm/connext-backup/PROJECT_xjs/submission.csv"
        try:
            submission_df.to_csv(backup_path, index=False)
            print(f"Backup saved to {backup_path}")
        except Exception as e:
            print(f"Failed to backup submission to {backup_path}: {e}")
    except Exception as e:
        print(f"Failed to save submission: {e}")
        raise

if __name__ == "__main__":
    try:
        test_df = load_data("test.csv")
        test_ds = Dataset.from_pandas(test_df)
        print(f"Test dataset size: {len(test_ds)}")
        train_ds, _, train_df = prepare_datasets(DATASET_PATH)
        predict_and_save(MODEL_PATH, test_ds, train_df)
    except Exception as e:
        print(f"Error in main: {e}")
        raise