import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from datasets import load_dataset
from transformers import DataCollatorWithPadding
from torch.utils.data import DataLoader


model_name = "fine-tuned-emotion-model"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)


dataset = load_dataset("dair-ai/emotion")


def preprocess_data(examples):
    return tokenizer(examples["text"], truncation=True, padding=True)


encoded_test_dataset = dataset["test"].map(preprocess_data, batched=True)
encoded_test_dataset = encoded_test_dataset.rename_column("label", "labels")
encoded_test_dataset.set_format(
    "torch", columns=["input_ids", "attention_mask", "labels"]
)


test_dataloader = DataLoader(
    encoded_test_dataset, batch_size=4, collate_fn=DataCollatorWithPadding(tokenizer)
)


model.eval()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

all_preds = []
all_labels = []

with torch.no_grad():
    for batch in test_dataloader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        outputs = model(input_ids, attention_mask=attention_mask)
        preds = torch.argmax(outputs.logits, dim=-1)

        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())


from sklearn.metrics import accuracy_score, f1_score, classification_report

accuracy = accuracy_score(all_labels, all_preds)
f1 = f1_score(all_labels, all_preds, average="weighted")
report = classification_report(
    all_labels, all_preds, target_names=dataset["train"].features["label"].names
)

print(f"Accuracy: {accuracy}")
print(f"F1 Score: {f1}")
print(f"Classification Report:\n{report}")
