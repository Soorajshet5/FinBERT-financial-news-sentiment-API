from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer, DataCollatorWithPadding

dataset = load_dataset("zeroshot/twitter-financial-news-sentiment")
tokenizer = AutoTokenizer.from_pretrained("ProsusAI/finbert")

def tokenize(batch):
    return tokenizer(batch["text"], truncation=True, max_length=512)

tokenized = dataset["train"].map(tokenize, batched=True)
tokenized = tokenized.rename_column("label", "labels")
tokenized.set_format("torch", columns=["input_ids", "attention_mask", "labels"])

model = AutoModelForSequenceClassification.from_pretrained("ProsusAI/finbert", num_labels=3)
collator = DataCollatorWithPadding(tokenizer=tokenizer)

args = TrainingArguments(
    output_dir="./finbert-finetuned",
    num_train_epochs=3,
    per_device_train_batch_size=16,
    save_steps=100,
    logging_steps=50,
)

trainer = Trainer(
    model=model,
    args=args,
    train_dataset=tokenized,
    data_collator=collator,
)

trainer.train()
model.save_pretrained("./finbert-finetuned")
print("Done")
