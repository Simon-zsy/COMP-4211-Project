from transformers import AutoModelForTokenClassification

def get_model(model, num_labels):
    return AutoModelForTokenClassification.from_pretrained(model, num_labels=num_labels)