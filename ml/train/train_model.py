from transformers import AutoModelForSequenceClassification, AutoTokenizer, Trainer, TrainingArguments
from datasets import load_dataset
import numpy as np
import logging
import os
import re
import torch
from sklearn.utils.class_weight import compute_class_weight
from torch.nn import CrossEntropyLoss
from collections import Counter
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(_name_)

def clean_text(text):
    text = re.sub(r'[^\w\s]', '', text.lower()).strip()
    return text if text else "empty"

def compute_metrics(pred):
    labels = pred.label_ids
    preds = np.argmax(pred.predictions, axis=1)
    acc = accuracy_score(labels, preds)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average="weighted")
    return {"accuracy": acc, "f1": f1, "precision": precision, "recall": recall}

def train_model():
    try:
        logger.info("Loading LIAR dataset")
        dataset = load_dataset("liar")

        def preprocess(example):
            label_map = {0: 0, 1: 1, 2: 2, 3: 2, 4: 1, 5: 0}  # fake=0, misleading=1, credible=2
            example["label"] = label_map[example["label"]]
            text = clean_text(example["statement"])
            return {"text": text, "label": example["label"]}

        dataset = dataset.map(preprocess)

        train_labels = [example["label"] for example in dataset["train"]]
        label_counts = Counter(train_labels)
        print(f"Label distribution in training set: {label_counts}")

        train_dataset = dataset["train"].filter(lambda x: len(x["text"].split()) > 3)
        eval_dataset = dataset["validation"].filter(lambda x: len(x["text"].split()) > 3)

        model_name = "roberta-base"
        logger.info(f"Loading tokenizer and model: {model_name}")
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            num_labels=3,
            id2label={0: "fake", 1: "misleading", 2: "credible"},
            label2id={"fake": 0, "misleading": 1, "credible": 2}
        )

        def tokenize_function(examples):
            return tokenizer(examples["text"], padding="max_length", truncation=True, max_length=256)

        logger.info("Tokenizing datasets")
        train_dataset = train_dataset.map(tokenize_function, batched=True)
        eval_dataset = eval_dataset.map(tokenize_function, batched=True)

        output_dir = "./ml/models/roberta-fake-news-bin-v2"
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        training_args = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=10,
            learning_rate=1e-5,
            per_device_train_batch_size=8,
            per_device_eval_batch_size=8,
            warmup_steps=500,
            weight_decay=0.01,
            logging_dir='./logs',
            logging_steps=10,
            evaluation_strategy="epoch",
            save_strategy="epoch",
            load_best_model_at_end=True,
            metric_for_best_model="accuracy",
            greater_is_better=True,
            save_safetensors=False
        )

        # Compute class weights and convert to tensor
        class_weights = compute_class_weight(class_weight="balanced", classes=np.unique(train_labels), y=train_labels)
        class_weights_tensor = torch.tensor(class_weights, dtype=torch.float)

        class WeightedTrainer(Trainer):
            def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
                labels = inputs.get("labels")
                outputs = model(**inputs)
                logits = outputs.get("logits")
                loss_fct = CrossEntropyLoss(weight=class_weights_tensor.to(model.device))
                loss = loss_fct(logits, labels)
                return (loss, outputs) if return_outputs else loss

        trainer = WeightedTrainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            compute_metrics=compute_metrics
        )

        logger.info("Starting training")
        trainer.train()

        logger.info("Saving model and tokenizer")
        try:
            model.save_pretrained(output_dir, safe_serialization=False)
            tokenizer.save_pretrained(output_dir)
            logger.info("Training complete")
        except Exception as e:
            logger.error(f"Failed to save model/tokenizer: {e}")

    except Exception as e:
        logger.error(f"Training failed: {str(e)}")
        raise

if _name_ == "_main_":
    train_model()