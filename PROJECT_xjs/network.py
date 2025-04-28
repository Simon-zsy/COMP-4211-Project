from transformers import DebertaV2ForTokenClassification

def get_model(num_labels):
    return DebertaV2ForTokenClassification.from_pretrained("microsoft/deberta-v3-large", num_labels=num_labels)