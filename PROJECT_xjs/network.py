from transformers import BertForTokenClassification

def get_model(num_labels):
    return BertForTokenClassification.from_pretrained("bert-base-cased", num_labels=num_labels)