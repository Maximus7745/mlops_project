import pytest
from datasets import load_dataset
import numpy as np


dataset = load_dataset("dair-ai/emotion")

def test_no_empty_texts():
    for split in dataset.keys():
        texts = dataset[split]["text"]
        assert all(text.strip() for text in texts), f"Found empty text in {split} split."

def test_text_length():
    max_length = 512
    for split in dataset.keys():
        texts = dataset[split]["text"]
        assert all(len(text.split()) <= max_length for text in texts), f"Found text longer than {max_length} tokens in {split} split."

def test_label_distribution():
    for split in dataset.keys():
        labels = dataset[split]["label"]
        unique, counts = np.unique(labels, return_counts=True)
        distribution = dict(zip(unique, counts))
        print(f"Label distribution in {split} split: {distribution}")
        assert len(unique) == 6, f"Unexpected number of unique labels in {split} split."
        assert all(count > 0 for count in counts), f"Found label with zero instances in {split} split."
