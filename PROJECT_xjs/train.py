import torch
from transformers import AutoModelForTokenClassification, Trainer, TrainingArguments
from network import create_label_mappings, make_compute_metrics, BertWithAttentionAndCRF
from data_loader import prepare_datasets
import os

# 配置
DATASET_PATH = "/localdata/szhoubx/rm/connext-backup/PROJECT_xjs"
MODEL_PATH = "./ner_model2"

def main():
    # 加载数据
    train_ds, eval_ds, train_df = prepare_datasets(DATASET_PATH)
    print(f"Training dataset size: {len(train_ds)}, Validation dataset size: {len(eval_ds)}")
    
    # 标签映射
    label_list, label2id, id2label, class_weights = create_label_mappings(train_df)
    
    # 加载模型
    model = BertWithAttentionAndCRF.from_pretrained(
        "bert-base-cased",
        num_labels=len(label_list),
        id2label=id2label,
        label2id=label2id
    )
    model.class_weights = class_weights
    
    # 冻结前 8 层
    for param in model.bert.encoder.layer[:8].parameters():
        param.requires_grad = False
    
    # 训练参数
    training_args = TrainingArguments(
        output_dir="./results",
        eval_strategy="epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        num_train_epochs=10,
        weight_decay=0.1,
        save_strategy="epoch",
        warmup_steps=500,
        load_best_model_at_end=True,
        logging_dir='./logs',
        logging_steps=100,
        report_to="none",
        lr_scheduler_type="cosine",
        metric_for_best_model="f1",
        greater_is_better=True
    )
    
    # 自定义 Trainer 以在第 2 epoch 后解冻层
    class CustomTrainer(Trainer):
        def training_step(self, model, inputs, gradient_accumulation_steps=1):
            if self.state.epoch >= 2:
                for param in model.bert.encoder.layer.parameters():
                    param.requires_grad = True
            return super().training_step(model, inputs)
    
    # 训练
    trainer = CustomTrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        compute_metrics=make_compute_metrics(id2label)
    )
    trainer.train()
    
    # 保存最佳模型
    trainer.save_model(MODEL_PATH)
    print(f"Best model saved to {MODEL_PATH}")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"Error in training: {e}")
        raise