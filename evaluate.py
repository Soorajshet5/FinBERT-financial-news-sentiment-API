from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.metrics import classification_report
import torch
import torch.nn.functional as F

# Load test set
dataset = load_dataset("zeroshot/twitter-financial-news-sentiment")
test_data = dataset["validation"]

# Load your fine-tuned model

tokenizer = AutoTokenizer.from_pretrained("ProsusAI/finbert")
model = AutoModelForSequenceClassification.from_pretrained("ProsusAI/finbert")
model.eval()

# Run predictions
all_preds = []
all_labels = []

for item in test_data:
    inputs = tokenizer(item["text"], return_tensors="pt", truncation=True, max_length=512)
    with torch.no_grad():
        logits = model(**inputs).logits
    pred = int(logits.argmax())
    all_preds.append(pred)
    all_labels.append(item["label"])

# Print results
print(classification_report(all_labels, all_preds, target_names=["bearish", "bullish", "neutral"]))