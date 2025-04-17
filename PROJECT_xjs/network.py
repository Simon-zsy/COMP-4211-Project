import torch
import torch.nn as nn
import numpy as np
from transformers import AutoModelForTokenClassification
from torchcrf import CRF
from seqeval.metrics import f1_score, precision_score, recall_score
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

def make_compute_metrics(id2label):
    """Factory function to create compute_metrics with id2label."""
    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        predictions = np.argmax(logits, axis=2)
        
        true_labels = [[id2label[l] for l in label if l != -100] for label in labels]
        pred_labels = [[id2label[p] for p, l in zip(pred, label) if l != -100] 
                      for pred, label in zip(predictions, labels)]
        
        return {
            "f1": f1_score(true_labels, pred_labels),
            "precision": precision_score(true_labels, pred_labels),
            "recall": recall_score(true_labels, pred_labels)
        }
    return compute_metrics

class BertWithAttentionAndCRF(AutoModelForTokenClassification):
    """Custom BERT model with attention and CRF layers."""
    def __init__(self, config):
        super().__init__(config)
        self.attention = nn.Linear(config.hidden_size, 1)
        self.dropout = nn.Dropout(0.3)
        self.crf = CRF(num_tags=config.num_labels, batch_first=True)
    
    def forward(self, input_ids, attention_mask=None, labels=None):
        outputs = self.bert(input_ids, attention_mask=attention_mask)
        sequence_output = outputs[0]
        
        # Apply attention
        attention_scores = torch.softmax(self.attention(sequence_output), dim=1)
        sequence_output = sequence_output * attention_scores
        sequence_output = self.dropout(sequence_output)
        
        emissions = self.classifier(sequence_output)
        
        # Apply class weights to emissions
        weighted_emissions = emissions * self.class_weights.to(emissions.device).view(1, 1, -1)
        
        if labels is not None:
            # CRF negative log-likelihood loss
            loss = -self.crf(weighted_emissions, labels, mask=attention_mask.bool(), reduction='mean')
            return {'loss': loss, 'logits': emissions}
        else:
            # CRF decoding for inference
            decoded_tags = self.crf.decode(weighted_emissions, mask=attention_mask.bool())
            logits = torch.zeros_like(emissions)
            for i, tags in enumerate(decoded_tags):
                for j, tag in enumerate(tags):
                    logits[i, j, tag] = 1.0
            return {'logits': logits}