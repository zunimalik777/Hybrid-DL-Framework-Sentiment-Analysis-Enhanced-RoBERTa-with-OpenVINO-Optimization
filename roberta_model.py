"""
Objective 1: RoBERTa

Fine-tunes `roberta-base` (HuggingFace Transformers) as the transformer
champion model for binary sentiment classification.
"""

import numpy as np
import torch
from torch.utils.data import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
    DataCollatorWithPadding,
)

from . import config


class TextClassificationDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len=config.MAX_SEQ_LEN):
        self.encodings = tokenizer(
            list(texts), truncation=True, max_length=max_len, padding=False
        )
        self.labels = list(labels)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        item = {k: torch.tensor(v[idx]) for k, v in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels[idx])
        return item


def load_roberta(model_name=config.ROBERTA_MODEL_NAME, num_labels=config.ROBERTA_NUM_CLASSES):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=num_labels)
    return tokenizer, model


def compute_metrics_fn(eval_pred):
    from sklearn.metrics import accuracy_score, precision_recall_fscore_support

    logits, labels = eval_pred
    preds = np.argmax(logits, axis=1)
    acc = accuracy_score(labels, preds)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average="binary")
    return {"accuracy": acc, "precision": precision, "recall": recall, "f1": f1}


def train_roberta(train_df, val_df, output_dir=None, epochs=config.ROBERTA_EPOCHS,
                   batch_size=config.ROBERTA_BATCH_SIZE, lr=config.ROBERTA_LR):
    output_dir = output_dir or f"{config.MODELS_DIR}/roberta_checkpoints"
    tokenizer, model = load_roberta()

    train_ds = TextClassificationDataset(train_df["text"], train_df["label"], tokenizer)
    val_ds = TextClassificationDataset(val_df["text"], val_df["label"], tokenizer)
    collator = DataCollatorWithPadding(tokenizer=tokenizer)

    args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size * 2,
        learning_rate=lr,
        weight_decay=config.ROBERTA_WEIGHT_DECAY,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        logging_steps=50,
        fp16=torch.cuda.is_available(),
        report_to=[],
        seed=config.SEED,
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=collator,
        compute_metrics=compute_metrics_fn,
    )
    trainer.train()
    return tokenizer, model, trainer


def evaluate_roberta(model, tokenizer, test_df, batch_size=config.ROBERTA_BATCH_SIZE * 2,
                      device=config.DEVICE):
    model.to(device)
    model.eval()
    collator = DataCollatorWithPadding(tokenizer=tokenizer)
    test_ds = TextClassificationDataset(test_df["text"], test_df["label"], tokenizer)

    from torch.utils.data import DataLoader
    loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, collate_fn=collator)

    all_preds, all_labels = [], []
    with torch.no_grad():
        for batch in loader:
            labels = batch.pop("labels")
            batch = {k: v.to(device) for k, v in batch.items()}
            logits = model(**batch).logits
            preds = logits.argmax(dim=1).cpu().numpy()
            all_preds.extend(preds.tolist())
            all_labels.extend(labels.numpy().tolist())
    return np.array(all_preds), np.array(all_labels)
