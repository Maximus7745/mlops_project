from datasets import load_from_disk
from transformers import AutoTokenizer


dataset = load_from_disk("emotion_dataset")


small_train_dataset = dataset["train"].train_test_split(test_size=0.9)["train"]
small_val_dataset = dataset["validation"].train_test_split(test_size=0.9)["train"]


model_name = "distilbert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)


def preprocess_data(examples):
    return tokenizer(examples["text"], truncation=True, padding=True)


encoded_train_dataset = small_train_dataset.map(preprocess_data, batched=True)
encoded_val_dataset = small_val_dataset.map(preprocess_data, batched=True)

encoded_train_dataset = encoded_train_dataset.rename_column("label", "labels")
encoded_val_dataset = encoded_val_dataset.rename_column("label", "labels")

encoded_train_dataset.set_format(
    "torch", columns=["input_ids", "attention_mask", "labels"]
)
encoded_val_dataset.set_format(
    "torch", columns=["input_ids", "attention_mask", "labels"]
)


encoded_train_dataset.save_to_disk("encoded_train_dataset")
encoded_val_dataset.save_to_disk("encoded_val_dataset")
print("Data preprocessed and saved to disk.")
