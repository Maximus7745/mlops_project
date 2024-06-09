from datasets import load_dataset


dataset = load_dataset("dair-ai/emotion")

dataset.save_to_disk("emotion_dataset")
print("Dataset loaded and saved to disk.")