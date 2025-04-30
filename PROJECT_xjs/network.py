import torch.nn as nn
from transformers import BertForTokenClassification

def get_model(num_labels, dropout_rate=0.3):
    # Load the pretrained model
    model = BertForTokenClassification.from_pretrained("0.864020", num_labels=num_labels)
    
    # Get the input dimension for the classifier
    classifier_input_size = model.classifier.in_features
    
    # Replace the classifier with a custom one that includes dropout
    model.classifier = nn.Sequential(
        nn.Dropout(dropout_rate),  # Add dropout before classification
        nn.Linear(classifier_input_size, num_labels)
    )
    
    return model