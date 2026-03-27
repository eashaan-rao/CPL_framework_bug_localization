"""
BLAZE embedding model.

Implements the core architecture from:
  "BLAZE: Cross-Language and Cross-Project Bug Localization via
   Dynamic Chunking and Hard Example Learning"

Key components:
  - BlazeEmbedding: pre-trained transformer + modal-specific ResidualAdapters
  - NTXentLoss: in-batch contrastive loss with hard example scaling
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer
from sentence_transformers.models import Pooling
from typing import List


class CustomPooling(Pooling):
    """Wraps sentence_transformers Pooling to accept a raw transformer output."""
    def forward(self, embedding, attention_mask):
        return super().forward({
            "token_embeddings": embedding.last_hidden_state,
            "attention_mask": attention_mask,
        })


class ResidualBlock(nn.Module):
    def __init__(self, dimension: int, dropout_rate: float = 0.1):
        super().__init__()
        self.dense = nn.Linear(dimension, dimension)
        self.activation = nn.PReLU(dimension)
        self.dropout = nn.Dropout(dropout_rate)
        self.norm = nn.LayerNorm(dimension)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.dense(x)
        out = self.activation(out)
        out = self.dropout(out)
        out = out + x  # skip connection
        return self.norm(out)


class ResidualAdapter(nn.Module):
    """Stack of residual blocks providing modal-specific feature projection."""
    def __init__(self, layers: int, dimension: int):
        super().__init__()
        self.blocks = nn.ModuleList([ResidualBlock(dimension) for _ in range(layers)])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for block in self.blocks:
            x = block(x)
        return x


class BlazeEmbedding(nn.Module):
    """
    BLAZE dual-encoder: one shared transformer backbone with two modal-specific
    ResidualAdapters — one for bug reports, one for source code chunks.

    The transformer backbone is intended to be used frozen (FREEZE_TRANSFORMER=True
    in pipeline.py) to keep GPU memory within bounds; only the adapters and pooler
    are trained.  This is consistent with the adapter-based fine-tuning spirit of
    the BLAZE paper while respecting our hardware constraints.

    Inputs are prefixed following the bge-code-v1 convention:
      bug reports  → "search_query: <text>"
      code chunks  → "search_document: <text>"

    Mean pooling is used (standard for bge-code-v1, a Qwen2-based decoder model).

    Args:
        model_name:      HuggingFace model id for the backbone transformer.
        dimension:       Hidden dimension of the backbone (must match its output).
        max_length:      Token truncation length for both modalities.
        residual_layers: Depth of each ResidualAdapter.
        report_prefix:   Text prepended to every bug report before tokenization.
        source_prefix:   Text prepended to every code chunk before tokenization.
    """
    def __init__(
        self,
        model_name: str,
        dimension: int,
        max_length: int = 512,
        residual_layers: int = 2,
        report_prefix: str = "search_query: ",
        source_prefix: str = "search_document: ",
    ):
        super().__init__()
        self.transformer = AutoModel.from_pretrained(model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.dimension = dimension
        self.max_length = max_length
        self.report_prefix = report_prefix
        self.source_prefix = source_prefix
        # Mean pooling is the standard for bge-code-v1 (Qwen2 decoder-only)
        self.pooler = CustomPooling(
            word_embedding_dimension=dimension, pooling_mode="mean"
        )
        self.report_adapter = ResidualAdapter(layers=residual_layers, dimension=dimension)
        self.source_adapter = ResidualAdapter(layers=residual_layers, dimension=dimension)

    def _tokenize(self, texts: List[str], prefix: str, device: str) -> dict:
        prefixed = [prefix + t for t in texts]
        encoded = self.tokenizer(
            prefixed,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        return {k: v.to(device) for k, v in encoded.items()}

    def get_report_embedding(self, texts: List[str]) -> torch.Tensor:
        """Encode bug report texts → L2-unnormalized embeddings."""
        device = next(self.transformer.parameters()).device
        encoded = self._tokenize(texts, self.report_prefix, device)
        out = self.transformer(**encoded)
        pooled = self.pooler(out, encoded["attention_mask"])["sentence_embedding"]
        # Detach from the frozen transformer's computation graph so PyTorch
        # does not store activations through the backbone for backprop.
        if not self.transformer.training:
            pooled = pooled.detach()
        return self.report_adapter(pooled)

    def get_source_embedding(self, texts: List[str]) -> torch.Tensor:
        """Encode source code chunk texts → L2-unnormalized embeddings."""
        device = next(self.transformer.parameters()).device
        encoded = self._tokenize(texts, self.source_prefix, device)
        out = self.transformer(**encoded)
        pooled = self.pooler(out, encoded["attention_mask"])["sentence_embedding"]
        if not self.transformer.training:
            pooled = pooled.detach()
        return self.source_adapter(pooled)

    def forward(self, report_texts: List[str], code_texts: List[str]) -> dict:
        return {
            "report": self.get_report_embedding(report_texts),
            "code": self.get_source_embedding(code_texts),
        }


class NTXentLoss(nn.Module):
    """
    Normalized Temperature-scaled Cross-Entropy loss with hard example scaling,
    as described in the BLAZE paper (Section 3.2).

    For a batch of (report_emb, code_emb, label) triples:
      - Pairs sharing the same label are positives; others are negatives.
      - Hard negatives  (high sim, but negative label) get scaled up   by hard_negative_scale.
      - Hard positives  (low  sim, but positive label) get scaled up   by hard_positive_scale.
      - All positive pairs are additionally scaled by positive_scale to boost their gradient.
    """
    def __init__(
        self,
        tau: float = 0.3,
        hard_negative_scale: float = 1.2,
        hard_positive_scale: float = 1.2,
        positive_scale: float = 1.001,
    ):
        super().__init__()
        self.tau = tau
        self.hard_negative_scale = hard_negative_scale
        self.hard_positive_scale = hard_positive_scale
        self.positive_scale = positive_scale

    def forward(
        self,
        report_emb: torch.Tensor,
        code_emb: torch.Tensor,
        labels: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            report_emb: (B, D) bug report embeddings.
            code_emb:   (B, D) code chunk embeddings (positive for same label).
            labels:     (B,)   integer issue IDs — items with matching IDs are positives.
        """
        ea = F.normalize(report_emb, p=2, dim=1)
        eb = F.normalize(code_emb, p=2, dim=1)

        # (B, B) cosine similarity matrix, temperature-scaled
        sim_matrix = torch.mm(ea, eb.t()) / self.tau

        # Numerical stability: subtract row-wise max before exp
        max_sim = sim_matrix.max(dim=1, keepdim=True).values
        sim_exp = torch.exp(sim_matrix - max_sim)

        # Positive / negative masks
        pos_mask = (labels.unsqueeze(1) == labels.unsqueeze(0)).float()
        neg_mask = 1.0 - pos_mask
        median_sim = sim_matrix.median()

        hard_pos_mask = (sim_matrix < median_sim).float() * pos_mask
        hard_neg_mask = (sim_matrix > median_sim).float() * neg_mask

        # Apply hard-example scaling
        sim_exp_scaled = (
            sim_exp
            * (1.0 + hard_neg_mask * (self.hard_negative_scale - 1.0))
            * (1.0 + hard_pos_mask * (self.hard_positive_scale - 1.0))
            * (1.0 + pos_mask * (self.positive_scale - 1.0))
        )

        pos_sum = (sim_exp_scaled * pos_mask).sum(dim=1).clamp(min=1e-10)
        all_sum = sim_exp_scaled.sum(dim=1)

        return -torch.log(pos_sum / all_sum).mean()
