from abc import ABC, abstractmethod

import torch
import torch.nn as nn


class ScoringModuleBase(nn.Module, ABC):
    def __init__(self, embed_size: int, hidden_size: int | None = None):
        super(ScoringModuleBase, self).__init__()
        self.embed_size = embed_size
        self.hidden_size = hidden_size or embed_size
        self.W1 = nn.Linear(self.embed_size, self.hidden_size)
        self.W2 = nn.Linear(self.embed_size, self.hidden_size)

    def forward(
        self, time_embeds: torch.Tensor, mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        """
        Args:
            time_embeds: torch.Tensor, shape (batch_size, seq_len, embed_size)
            mask: torch.Tensor, shape (batch_size, seq_len)
        Returns:
            torch.Tensor, shape (batch_size, seq_len, seq_len)
        """
        batch_size, seq_len, _ = time_embeds.size()
        # Compute the scores
        W1: torch.Tensor = self.W1(time_embeds)  # (batch_size, seq_len, hidden_size)
        W2: torch.Tensor = self.W2(time_embeds)  # (batch_size, seq_len, hidden_size)
        # Pairwise addition
        scores = self.compute_scores(W1, W2)  # (batch_size, seq_len, seq_len)

        # Mask out the scores
        if mask is not None:
            mask = mask.unsqueeze(1)
            mask = mask.expand(batch_size, seq_len, seq_len)
            scores = scores.masked_fill(mask, float("-inf"))

        return scores

    @abstractmethod
    def compute_scores(self, W1: torch.Tensor, W2: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError("Subclasses must implement compute_scores method")


class ScoringModuleAddn(ScoringModuleBase):
    def __init__(self, embed_size: int, hidden_size: int | None = None):
        super(ScoringModuleAddn, self).__init__(embed_size, hidden_size)
        self.b = nn.Parameter(torch.zeros(self.hidden_size))
        self.v = nn.Parameter(torch.zeros(self.hidden_size))
        # Initialize weights
        nn.init.uniform_(self.b)
        nn.init.uniform_(self.v)

    def compute_scores(self, W1: torch.Tensor, W2: torch.Tensor) -> torch.Tensor:
        scores = (
            W1.unsqueeze(1) + W2.unsqueeze(2) + self.b[None, None, None, :]
        )  # (batch_size, seq_len, seq_len, hidden_size)
        scores = torch.tanh(scores)
        scores = scores @ self.v  # (batch_size, seq_len, seq_len)
        return scores


class ScoringModuleMult(ScoringModuleBase):
    def __init__(self, embed_size: int, hidden_size: int | None = None):
        super(ScoringModuleMult, self).__init__(embed_size, hidden_size)

    def compute_scores(self, W1: torch.Tensor, W2: torch.Tensor) -> torch.Tensor:
        scores = torch.bmm(W1, W2.transpose(1, 2))  # (batch_size, seq_len, seq_len)
        return scores
