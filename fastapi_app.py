from fastapi import FastAPI
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch


model_name = "fine-tuned-emotion-model"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)


model.eval()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)


app = FastAPI()


class TextInput(BaseModel):
    text: str


def predict_emotion(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True).to(
        device
    )
    with torch.no_grad():
        outputs = model(**inputs)
    logits = outputs.logits
    predicted_class_id = torch.argmax(logits, dim=-1).item()
    labels = [
        "sadness",
        "joy",
        "love",
        "anger",
        "fear",
        "surprise",
    ]  # Замените на ваши метки
    return labels[predicted_class_id]


@app.post("/predict/")
def predict(input: TextInput):
    predicted_emotion = predict_emotion(input.text)
    return {"text": input.text, "predicted_emotion": predicted_emotion}
