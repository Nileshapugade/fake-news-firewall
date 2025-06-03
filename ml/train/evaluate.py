from transformers import RobertaForSequenceClassification, RobertaTokenizer, Trainer, TrainingArguments
from datasets import load_dataset

def evaluate_model():
    dataset = load_dataset("csv", data_files={"test": "ml/data/processed_data.csv"})
    tokenizer = RobertaTokenizer.from_pretrained("./ml/models/roberta-fake-news")
    
    def tokenize_function(examples):
        return tokenizer(examples["text"], padding="max_length", truncation=True)

    tokenized_dataset = dataset.map(tokenize_function, batched=True)
    model = RobertaForSequenceClassification.from_pretrained("./ml/models/roberta-fake-news")

    trainer = Trainer(model=model)
    metrics = trainer.evaluate(tokenized_dataset["test"])
    print(metrics)

if _name_ == "_main_":
    evaluate_model()