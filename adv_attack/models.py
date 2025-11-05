import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel
from typing import List


class EmbeddingClassifier(nn.Module):
    """
    Embedding-based classifier for binary SMS phishing detection.
    Compatible with AutoAttack by accepting continuous embeddings as input.
    """

    def __init__(self, model_name: str = "bert-base-uncased"):
        super().__init__()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.encoder = AutoModel.from_pretrained(model_name)

        hidden = self.encoder.config.hidden_size
        self.classifier = nn.Sequential(
            nn.Linear(hidden, 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, 2),
        )

        self.encoder = self.encoder.to(self.device)
        self.classifier = self.classifier.to(self.device)
        self.encoder.eval()
        self.classifier.eval()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(x)

    def encode_texts(self, texts: List[str]) -> torch.Tensor:
        with torch.no_grad():
            inputs = self.tokenizer(
                texts,
                padding=True,
                truncation=True,
                max_length=128,
                return_tensors="pt",
            ).to(self.device)
            outputs = self.encoder(**inputs)
            embeddings = outputs.last_hidden_state[:, 0, :]
        return embeddings






