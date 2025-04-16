import os
os.environ["CUDA_VISIBLE_DEVICES"] = "5,6,7"

from transformers import (
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForTokenClassification
)
from network import initialize_model, create_label_mappings, compute_metrics
from data_loader import load_data, prepare_datasets, preprocess_data

DATASET_PATH = "/localdata/szhoubx/rm/connext-backup/PROJECT_xjs"
MODEL_CHECKPOINT = "bert-base-cased"

def main():
    # 1. 准备数据集
    train_ds, val_ds, train_df = prepare_datasets(DATASET_PATH)

    # 2. 创建标签映射
    label_list, label2id, id2label = create_label_mappings(train_df)

    # 3. 加载Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL_CHECKPOINT)

    # 4. 预处理数据
    tokenized_train, tokenized_val = preprocess_data(train_ds, val_ds, tokenizer, label2id)

    # 5. 初始化模型
    model = initialize_model(MODEL_CHECKPOINT, label_list, id2label, label2id)

    # 6. 设置训练参数
    training_args = TrainingArguments(
        output_dir="./results",
        eval_strategy="epoch",
        learning_rate=1e-4,
        per_device_train_batch_size=128,
        per_device_eval_batch_size=128,
        num_train_epochs=50,
        weight_decay=0.01,
        save_strategy="epoch",
        load_best_model_at_end=True,
        logging_dir='./logs',
        report_to="none",
        # 启用 cosine scheduler
        lr_scheduler_type="cosine"  # 设置为余弦调度器
    )

    # 7. 创建数据收集器
    data_collator = DataCollatorForTokenClassification(tokenizer)

    # 8. 初始化Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_val,
        data_collator=data_collator,
        tokenizer=tokenizer,
        compute_metrics=lambda p: compute_metrics(p, label_list)
    )

    # 9. 训练模型
    trainer.train()

    # 10. 保存模型
    model.save_pretrained("./ner_model2")
    tokenizer.save_pretrained("./ner_model2")

if __name__ == "__main__":
    main()