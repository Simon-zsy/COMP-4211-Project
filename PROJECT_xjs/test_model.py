import pandas as pd
from ast import literal_eval
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForTokenClassification, Trainer
import numpy as np
from data_loader import create_label_mappings
from data_loader import load_and_preprocess_data

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

def load_test_data(file_path):
    """Load and preprocess test data, normalizing non-standard characters."""
    try:
        df = pd.read_csv(file_path)
        df['Sentence'] = df['Sentence'].apply(literal_eval)
        df['Sentence'] = df['Sentence'].apply(lambda x: [normalize_token(w.strip()) for w in x if w.strip()])
        return df
    except Exception as e:
        print(f"Failed to load test data from {file_path}: {e}")
        raise

def predict_and_save(model_path, test_ds, train_df, output_file="submission.csv"):
    """Predict NER tags and save submission."""
    try:
        # Load model and tokenizer
        model = AutoModelForTokenClassification.from_pretrained(model_path)
        tokenizer = AutoTokenizer.from_pretrained(model_path)
    except Exception as e:
        print(f"Failed to load model or tokenizer from {model_path}: {e}")
        raise
    
    # Load label mappings from training data
    try:
        label_list, label2id, id2label, _ = create_label_mappings(train_df)
    except Exception as e:
        print(f"Failed to create label mappings: {e}")
        id2label = model.config.id2label
        label2id = model.config.label2id
        if not id2label:
            print("Warning: id2label not found in model config. Using default labels.")
            id2label = {i: f"LABEL_{i}" for i in range(model.config.num_labels)}
        label_list = list(label2id.keys())
    
    def tokenize(examples):
        """Tokenize sentences for prediction."""
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
        # Tokenize test dataset
        tokenized_test = test_ds.map(tokenize, batched=True, remove_columns=["Sentence", "id"])
    except Exception as e:
        print(f"Failed to tokenize test dataset: {e}")
        raise
    
    # Initialize Trainer
    trainer = Trainer(model=model, tokenizer=tokenizer)
    
    try:
        # Predict
        predictions = trainer.predict(tokenized_test)
        preds = np.argmax(predictions.predictions, axis=2)  # Convert logits to label IDs
    except Exception as e:
        print(f"Prediction failed: {e}")
        raise
    
    # Process predictions
    final_preds = []
    test_df = test_ds.to_pandas()
    
    for i in range(len(test_ds)):
        sentence = test_ds[i]["Sentence"]
        sentence_length = len(sentence)
        pred_ids = preds[i]
        
        # Re-tokenize to get word_ids
        encoding = tokenizer(
            sentence,
            truncation=True,
            is_split_into_words=True,
            padding=False,
            max_length=256,
            return_special_tokens_mask=True
        )
        word_ids = encoding.word_ids()
        
        # Align predictions with words
        word_to_tag = {}
        for idx, word_idx in enumerate(word_ids):
            if word_idx is None:
                continue
            if word_idx not in word_to_tag:
                pred_tag = id2label[pred_ids[idx]] if pred_ids[idx] != -100 else 'O'
                word_to_tag[word_idx] = pred_tag
        
        # Assign tags to each word
        word_labels = []
        for word_idx in range(sentence_length):
            normalized = normalize_token(sentence[word_idx])
            if normalized in ["'s", ' ', '-', '"', ''] or not normalized.strip():
                word_labels.append("O")
            else:
                word_labels.append(word_to_tag.get(word_idx, "O"))
        
        # Format as JSON-like string
        word_labels_str = str(word_labels)
        final_preds.append({'id': test_ds[i]['id'], 'NER Tag': word_labels_str})
    
    # Save submission
    try:
        submission_df = pd.DataFrame(final_preds)
        # Validate lengths
        for idx, row in submission_df.iterrows():
            sentence = test_ds[idx]["Sentence"]
            predicted_tags = literal_eval(row["NER Tag"])
            if len(predicted_tags) != len(sentence):
                print(f"Warning: ID {row['id']} has mismatched lengths: "
                      f"Predicted {len(predicted_tags)}, Sentence {len(sentence)}")
        submission_df.to_csv(output_file, index=False)
        print(f"Submission saved to {output_file}")
    except Exception as e:
        print(f"Failed to save submission to {output_file}: {e}")
        raise

def main():
    # Configuration
    model_path = "fine_tuned_bert_ner"
    test_file = "test.csv"
    train_file = "/localdata/szhoubx/rm/connext-backup/PROJECT_xjs/train.csv"
    output_file = "submission.csv"
    
    # Load test and train data
    test_df = load_test_data(test_file)
    test_ds = Dataset.from_pandas(test_df)
    print(f"Test dataset size: {len(test_ds)}")
    
    train_df = load_and_preprocess_data(train_file)
    print(f"Train dataset size: {len(train_df)}")
    
    # Predict and save
    predict_and_save(model_path, test_ds, train_df, output_file)

if __name__ == "__main__":
    main()