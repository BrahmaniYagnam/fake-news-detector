from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from backend.config.settings import settings

if TYPE_CHECKING:
    from transformers import BertTokenizerFast
    from backend.models.bert_classifier import BertClassifier


class ExplainabilityService:
    def __init__(self, model: "BertClassifier", tokenizer: "BertTokenizerFast") -> None:
        self.model = model
        self.tokenizer = tokenizer

    def get_importance(self, text: str, top_k: int = 10) -> list[tuple[str, float]]:
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
                output_attentions=True,
            )
            attentions = outputs.attentions

        if not attentions:
            tokens = self.tokenizer.convert_ids_to_tokens(encoding["input_ids"][0])
            return [(token, 0.0) for token in tokens if token not in ["[CLS]", "[SEP]", "[PAD]"]][:top_k]

        try:
            last_layer_attention = attentions[-1]
        except (IndexError, TypeError):
            tokens = self.tokenizer.convert_ids_to_tokens(encoding["input_ids"][0])
            return [(token, 0.0) for token in tokens if token not in ["[CLS]", "[SEP]", "[PAD]"]][:top_k]

        if last_layer_attention is None:
            tokens = self.tokenizer.convert_ids_to_tokens(encoding["input_ids"][0])
            return [(token, 0.0) for token in tokens if token not in ["[CLS]", "[SEP]", "[PAD]"]][:top_k]

        last_layer_attention = last_layer_attention.squeeze(0)
        if last_layer_attention.numel() == 0:
            tokens = self.tokenizer.convert_ids_to_tokens(encoding["input_ids"][0])
            return [(token, 0.0) for token in tokens if token not in ["[CLS]", "[SEP]", "[PAD]"]][:top_k]

        cls_attention = last_layer_attention.mean(dim=0)[0]
        tokens = self.tokenizer.convert_ids_to_tokens(encoding["input_ids"][0])
        token_scores = [
            (token, float(cls_attention[idx].item()))
            for idx, token in enumerate(tokens)
            if token not in ["[CLS]", "[SEP]", "[PAD]"]
        ]
        token_scores = sorted(token_scores, key=lambda item: item[1], reverse=True)[:top_k]
        return token_scores
