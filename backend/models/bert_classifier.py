from __future__ import annotations

import torch
import torch.nn as nn
from transformers import BertConfig, BertModel


class BertClassifier(nn.Module):
    def __init__(self, dropout: float = 0.3, num_labels: int = 2, pretrained: bool = True) -> None:
        super().__init__()
        if pretrained:
            self.bert = BertModel.from_pretrained("bert-base-uncased")
        else:
            config = BertConfig()
            self.bert = BertModel(config)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(self.bert.config.hidden_size, num_labels)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        output_attentions: bool = False,
    ) -> torch.Tensor:
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_attentions=output_attentions,
        )
        pooled = outputs.pooler_output
        dropped = self.dropout(pooled)
        logits = self.classifier(dropped)
        if output_attentions:
            return logits, outputs.attentions
        return logits
