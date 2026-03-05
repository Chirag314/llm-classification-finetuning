import torch
import torch.nn as nn
from transformers import AutoModel

class CrossEncoderClassifier(nn.Module):
    def __init__(self, model_name: str, num_labels: int = 3):
        super().__init__()
        self.backbone = AutoModel.from_pretrained(model_name)
        hidden = self.backbone.config.hidden_size
        self.dropout = nn.Dropout(0.1)
        self.head = nn.Linear(hidden, num_labels)

    def forward(self, input_ids, attention_mask):
        out = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        cls = out.last_hidden_state[:, 0, :]
        logits = self.head(self.dropout(cls))
        return logits