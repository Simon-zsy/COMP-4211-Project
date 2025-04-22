from transformers import BertForTokenClassification

def get_model(num_labels):
    return BertForTokenClassification.from_pretrained("bert-large-cased", num_labels=num_labels)