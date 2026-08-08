from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

from backend.config.settings import settings


class ModelService:
    def __init__(self) -> None:
        self.model_name = "Pulk17/Fake-News-Detection"
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(self.model_name)
        self.model.eval()
        # Ensure model handles explanations by requiring gradients if needed, 
        # though evaluate mode usually turns off dropout. Explainability needs gradients.
        self.labels = {0: "Fake", 1: "Real"}

    def predict(self, text: str) -> dict[str, Any]:
        cleaned = text.strip()
        encoding = self.tokenizer(
            cleaned,
            truncation=True,
            padding="max_length",
            max_length=settings.max_length,
            return_tensors="pt",
        )
        with torch.no_grad():
            outputs = self.model(
                input_ids=encoding["input_ids"],
                attention_mask=encoding["attention_mask"],
            )
            logits = outputs.logits
            
        probabilities = torch.softmax(logits, dim=1).squeeze().tolist()
        label_id = int(torch.argmax(logits, dim=1).item())
        return {
            "prediction": self.labels[label_id],
            "confidence": float(probabilities[label_id]),
            "probabilities": {
                self.labels[idx]: float(score)
                for idx, score in enumerate(probabilities)
            },
            "tokens": self.tokenizer.convert_ids_to_tokens(encoding["input_ids"][0]),
            "probabilities_raw": probabilities,
        }

    def get_info(self) -> dict[str, Any]:
        return {
            "model_name": self.model_name,
            "version": settings.app_version,
            "framework": "PyTorch/HuggingFace",
            "dataset": "ISOT Fake News Dataset (Pre-trained)",
            "num_classes": len(self.labels),
            "training_accuracy": 0.99,
        }
