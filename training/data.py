from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from transformers import BertTokenizerFast

from backend.config.settings import settings


class FakeNewsDataset(Dataset):
    def __init__(self, texts: list[str], labels: list[int], tokenizer: BertTokenizerFast) -> None:
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        encoding = self.tokenizer(
            self.texts[idx],
            truncation=True,
            padding="max_length",
            max_length=settings.max_length,
            return_tensors="pt",
        )
        return {
            "input_ids": encoding["input_ids"].squeeze(),
            "attention_mask": encoding["attention_mask"].squeeze(),
            "labels": torch.tensor(self.labels[idx], dtype=torch.long),
        }


def load_dataset(path: Path) -> pd.DataFrame:
    data = pd.read_csv(path)
    data = data.dropna(subset=["title", "text", "label"])
    data = data.drop_duplicates(subset=["title", "text"])
    data["text"] = data["title"].fillna("") + " " + data["text"].fillna("")
    data["text"] = data["text"].astype(str).str.strip()
    return data[["text", "label"]].reset_index(drop=True)


def prepare_data_loaders(
    dataframe: pd.DataFrame,
    tokenizer: BertTokenizerFast,
) -> dict[str, DataLoader]:
    label_map = {"FAKE": 0, "REAL": 1}
    dataframe = dataframe[dataframe["label"].isin(label_map)]
    texts = dataframe["text"].tolist()
    labels = [label_map[label] for label in dataframe["label"].tolist()]

    train_texts, temp_texts, train_labels, temp_labels = train_test_split(
        texts, labels, test_size=0.30, random_state=42
    )
    val_texts, test_texts, val_labels, test_labels = train_test_split(
        temp_texts,
        temp_labels,
        test_size=0.50,
        random_state=42,
    )

    train_dataset = FakeNewsDataset(train_texts, train_labels, tokenizer)
    val_dataset = FakeNewsDataset(val_texts, val_labels, tokenizer)
    test_dataset = FakeNewsDataset(test_texts, test_labels, tokenizer)

    return {
        "train": DataLoader(train_dataset, batch_size=settings.batch_size, shuffle=True),
        "validation": DataLoader(val_dataset, batch_size=settings.batch_size, shuffle=False),
        "test": DataLoader(test_dataset, batch_size=settings.batch_size, shuffle=False),
        "label_map": label_map,
    }


def save_metadata(model_dir: Path, metadata: dict[str, Any]) -> None:
    metadata_path = model_dir / "metadata.json"
    model_dir.mkdir(parents=True, exist_ok=True)
    with open(metadata_path, "w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=4)


def save_labels(model_dir: Path, label_map: dict[str, int]) -> None:
    labels_path = model_dir / "labels.json"
    inverse_labels = {str(value): key for key, value in label_map.items()}
    with open(labels_path, "w", encoding="utf-8") as handle:
        json.dump(inverse_labels, handle, indent=4)
