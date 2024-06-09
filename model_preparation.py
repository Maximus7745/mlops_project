from transformers import AutoTokenizer, AutoModelForSequenceClassification, DataCollatorWithPadding
from torch.utils.data import DataLoader
from transformers import AdamW
from tqdm import tqdm
import torch
from datasets import load_from_disk


encoded_train_dataset = load_from_disk("encoded_train_dataset")
encoded_val_dataset = load_from_disk("encoded_val_dataset")


model_name = "distilbert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=6)


data_collator = DataCollatorWithPadding(tokenizer)

train_dataloader = DataLoader(encoded_train_dataset, batch_size=4, shuffle=True, collate_fn=data_collator)
val_dataloader = DataLoader(encoded_val_dataset, batch_size=4, collate_fn=data_collator)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)

for epoch in range(1):  
    model.train()
    for batch in tqdm(train_dataloader):
        optimizer.zero_grad()
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        loss.backward()
        optimizer.step()

    model.eval()

model.save_pretrained("fine-tuned-emotion-model")
tokenizer.save_pretrained("fine-tuned-emotion-model")
print("Model trained and saved.")
