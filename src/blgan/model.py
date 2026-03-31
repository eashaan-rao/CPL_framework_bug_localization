"""
BL-GAN model components — paper-faithful implementation.

Architecture follows:
  "BL-GAN: Semi-Supervised Bug Localization via Generative Adversarial Network"
  IEEE TKDE 2023.

Key design decisions matching the paper:
  - Shared learned vocabulary (no BGE embeddings as model input).
  - Bug encoder: Embedding → PositionalEncoding → TransformerEncoder → mean pool.
  - Code content encoder (D only): GCN on AST graph token IDs.
  - File path encoder (D only): BiLSTM over path token IDs.
  - Generator: LSTM tree agent performing directory tree traversal (REINFORCE).
  - G and D each have their own nn.Embedding — weights are NOT shared.
"""

import math
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torch.distributions import Categorical
from torch_geometric.nn import GCNConv, global_mean_pool
from torch_geometric.data import Batch, Data

MAX_DEPTH = 15  # maximum tree traversal steps in DirectoryTreeAgent


# ---------------------------------------------------------------------------
# Positional encoding (sinusoidal)
# ---------------------------------------------------------------------------

class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding as in Vaswani et al. (2017).

    Args:
        d_model (int): Model embedding dimension.
        max_len (int): Maximum sequence length to pre-compute.
        dropout (float): Dropout applied after adding positional encodings.
    """

    def __init__(self, d_model: int, max_len: int = 512, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)  # (max_len, 1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float) * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, L, d_model)
        Returns:
            (B, L, d_model) with positional encodings added.
        """
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


# ---------------------------------------------------------------------------
# Bug report Transformer encoder
# ---------------------------------------------------------------------------

class BugTransformerEncoder(nn.Module):
    """Encodes tokenised bug reports via Transformer then mean-pools.

    Architecture:
        token_ids → Embedding → PositionalEncoding → TransformerEncoder
                  → mean pool (non-padding positions) → b_vec

    Args:
        embedding   : Shared nn.Embedding instance (passed in from parent).
        embed_dim   : Embedding dimension (must match embedding.embedding_dim).
        nhead       : Number of attention heads (default 6).
        num_layers  : Number of TransformerEncoder layers (default 2).
        dropout     : Dropout probability (default 0.5).
    """

    def __init__(
        self,
        embedding: nn.Embedding,
        embed_dim: int,
        nhead: int = 6,
        num_layers: int = 2,
        dropout: float = 0.5,
    ):
        super().__init__()
        self.embedding = embedding
        self.pos_enc = PositionalEncoding(embed_dim, max_len=512, dropout=dropout)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=nhead,
            dim_feedforward=embed_dim * 4,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

    def forward(self, token_ids: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        """
        Args:
            token_ids : LongTensor (B, L) — padded with 0.
            lengths   : LongTensor (B,)  — actual sequence lengths.
        Returns:
            (B, embed_dim) mean-pooled representation.
        """
        # Padding mask: True = position should be ignored (PyTorch convention)
        src_key_padding_mask = (token_ids == 0)  # (B, L)

        emb = self.embedding(token_ids)          # (B, L, E)
        emb = self.pos_enc(emb)                  # (B, L, E)

        out = self.transformer(emb, src_key_padding_mask=src_key_padding_mask)  # (B, L, E)

        # Mean pool over non-padding positions
        # lengths: (B,) — create mask (B, L)
        L = token_ids.size(1)
        range_tensor = torch.arange(L, device=token_ids.device).unsqueeze(0)  # (1, L)
        lengths_clamped = lengths.clamp(min=1).unsqueeze(1)                    # (B, 1)
        non_pad_mask = range_tensor < lengths_clamped                          # (B, L)

        out_masked = out * non_pad_mask.unsqueeze(-1).float()                  # (B, L, E)
        summed = out_masked.sum(dim=1)                                         # (B, E)
        pooled = summed / lengths_clamped.float()                              # (B, E)

        return pooled


# ---------------------------------------------------------------------------
# Code content encoder (GCN on AST)
# ---------------------------------------------------------------------------

class CodeGCNEncoder(nn.Module):
    """Encodes an AST graph via GCN on embedded token IDs.

    Architecture:
        node_ids (LongTensor N,) → Embedding → 3× GCNConv(300→300, ReLU, Dropout)
                                 → global_mean_pool → c_tilde (embed_dim,)

    Args:
        embedding    : Shared nn.Embedding instance.
        embed_dim    : Embedding / GCN feature dimension (default 300).
        num_gcn_layers: Number of GCN layers (default 3).
        dropout      : Dropout probability after each GCN (default 0.5).
    """

    def __init__(
        self,
        embedding: nn.Embedding,
        embed_dim: int,
        num_gcn_layers: int = 3,
        dropout: float = 0.5,
    ):
        super().__init__()
        self.embedding = embedding
        self.dropout = nn.Dropout(dropout)

        self.convs = nn.ModuleList()
        for _ in range(num_gcn_layers):
            self.convs.append(GCNConv(embed_dim, embed_dim))

    def forward(self, batch_data: Batch) -> torch.Tensor:
        """
        Args:
            batch_data: torch_geometric Batch.
                        batch_data.x          — LongTensor (N_total,) node token IDs.
                        batch_data.edge_index — LongTensor (2, E_total).
                        batch_data.batch      — LongTensor (N_total,) graph membership.
        Returns:
            (B, embed_dim) graph-level representations.
        """
        x = self.embedding(batch_data.x)          # (N_total, embed_dim)
        edge_index = batch_data.edge_index
        batch_vec = batch_data.batch

        for conv in self.convs:
            x = conv(x, edge_index)
            x = F.relu(x)
            x = self.dropout(x)

        out = global_mean_pool(x, batch_vec)       # (B, embed_dim)
        return out


# ---------------------------------------------------------------------------
# File path BiLSTM encoder
# ---------------------------------------------------------------------------

class FilePathBiLSTM(nn.Module):
    """Encodes a tokenised file path via BiLSTM.

    Architecture:
        path_ids → Embedding → BiLSTM(num_layers) → concat(last_fwd, last_bwd)
                             → c_hat (hidden_dim * 2,)

    Args:
        embedding   : Shared nn.Embedding instance.
        embed_dim   : Embedding dimension.
        hidden_dim  : LSTM hidden dimension per direction (default 256).
        num_layers  : Number of BiLSTM layers (default 2).
        dropout     : Dropout probability (default 0.5).
    """

    def __init__(
        self,
        embedding: nn.Embedding,
        embed_dim: int,
        hidden_dim: int = 256,
        num_layers: int = 2,
        dropout: float = 0.5,
    ):
        super().__init__()
        self.embedding = embedding
        self.lstm = nn.LSTM(
            embed_dim,
            hidden_dim,
            num_layers=num_layers,
            bidirectional=True,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.output_dim = hidden_dim * 2
        self.dropout = nn.Dropout(dropout)

    def forward(self, path_ids: torch.Tensor, path_lengths: torch.Tensor) -> torch.Tensor:
        """
        Args:
            path_ids    : LongTensor (B, L) — padded with 0.
            path_lengths: LongTensor (B,)  — actual sequence lengths (>= 1).
        Returns:
            (B, hidden_dim * 2)
        """
        path_lengths = path_lengths.clamp(min=1).cpu()
        emb = self.embedding(path_ids)             # (B, L, E)
        self.lstm.flatten_parameters()
        packed = pack_padded_sequence(emb, path_lengths, batch_first=True, enforce_sorted=False)
        _, (hidden, _) = self.lstm(packed)
        # hidden: (num_layers * 2, B, H); take last layer's fwd + bwd
        fwd = hidden[-2]                           # (B, H)
        bwd = hidden[-1]                           # (B, H)
        return self.dropout(torch.cat([fwd, bwd], dim=1))  # (B, 2H)


# ---------------------------------------------------------------------------
# Directory tree traversal agent (Generator)
# ---------------------------------------------------------------------------

class DirectoryTreeAgent(nn.Module):
    """LSTM-based directory tree traversal agent for the BL-GAN Generator.

    At each traversal step the agent receives the bug encoding b_G, updates
    its LSTM hidden state, computes child scores via dot product, and selects
    a child node stochastically (REINFORCE) or greedily.  It stops when it
    reaches a leaf node.

    Args:
        embedding   : Generator's own nn.Embedding.
        embed_dim   : Embedding dimension (also LSTM input size).
        hidden_dim  : LSTM hidden dimension (default 256).
        num_layers  : Number of LSTM layers (default 2).
    """

    def __init__(
        self,
        embedding: nn.Embedding,
        embed_dim: int,
        hidden_dim: int = 256,
        num_layers: int = 2,
    ):
        super().__init__()
        self.embedding = embedding
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        self.lstm = nn.LSTM(embed_dim, hidden_dim, num_layers=num_layers, batch_first=False)

        # Project bug encoding to initial LSTM h and c
        self.init_h = nn.Linear(embed_dim, num_layers * hidden_dim)
        self.init_c = nn.Linear(embed_dim, num_layers * hidden_dim)

        # Project LSTM output back to embed_dim for child scoring via dot product
        self.score_proj = nn.Linear(hidden_dim, embed_dim)

    def _embed_node(self, name_tokens: List[int], device: torch.device) -> torch.Tensor:
        """Mean-pool embeddings of a list of token IDs to get a node embedding.

        Args:
            name_tokens: List of integer vocabulary IDs.
            device     : Target device.
        Returns:
            (embed_dim,) tensor; zeros if name_tokens is empty.
        """
        if not name_tokens:
            return torch.zeros(self.embedding.embedding_dim, device=device)
        ids = torch.tensor(name_tokens, dtype=torch.long, device=device)
        embs = self.embedding(ids)   # (K, embed_dim)
        return embs.mean(dim=0)      # (embed_dim,)

    def traverse(
        self,
        b_G: torch.Tensor,
        dir_tree: dict,
        device: torch.device,
        greedy: bool = False,
    ) -> Tuple[str, torch.Tensor]:
        """Traverse the directory tree using the LSTM policy.

        Args:
            b_G     : (embed_dim,) bug encoding (with gradients).
            dir_tree: Dict keyed by path string.
                      Each value: {'name_tokens': List[int], 'is_leaf': bool,
                                   'children': List[str], 'sha': Optional[str]}
                      Root node has key ''.
            device  : Torch device.
            greedy  : If True, always pick the highest-scoring child.
        Returns:
            (file_path: str, episode_log_prob: scalar Tensor with grad)
        """
        # Initialise LSTM hidden state from bug encoding
        h0 = self.init_h(b_G)  # (num_layers * hidden_dim,)
        c0 = self.init_c(b_G)  # (num_layers * hidden_dim,)
        # Reshape to (num_layers, 1, hidden_dim)
        h = h0.view(self.num_layers, 1, self.hidden_dim)
        c = c0.view(self.num_layers, 1, self.hidden_dim)

        current_node = ''
        episode_log_prob = torch.zeros(1, device=device, requires_grad=False)
        # We accumulate log probs (which do have grad via b_G → h → scores)
        episode_log_prob = b_G.sum() * 0.0  # zero scalar with grad attached to b_G graph

        for _ in range(MAX_DEPTH):
            node_info = dir_tree.get(current_node)
            if node_info is None:
                break
            if node_info.get('is_leaf', False):
                break
            children = node_info.get('children', [])
            if not children:
                break

            # LSTM step: input is b_G (detached is NOT done — we need grad for REINFORCE)
            lstm_input = b_G.unsqueeze(0).unsqueeze(0)   # (1, 1, embed_dim)
            out, (h, c) = self.lstm(lstm_input, (h, c))
            h_last = h[-1].squeeze(0)                    # (hidden_dim,)
            h_proj = self.score_proj(h_last)             # (embed_dim,)

            # Score each child via dot product with its name embedding
            child_embs = torch.stack(
                [self._embed_node(dir_tree[ch]['name_tokens'], device) for ch in children],
                dim=0,
            )  # (num_children, embed_dim)

            scores = child_embs @ h_proj  # (num_children,)

            log_probs_children = F.log_softmax(scores, dim=0)  # (num_children,)

            if greedy:
                idx = log_probs_children.argmax()
            else:
                dist = Categorical(logits=scores)
                idx = dist.sample()

            episode_log_prob = episode_log_prob + log_probs_children[idx]
            current_node = children[idx.item()]

        return current_node, episode_log_prob


# ---------------------------------------------------------------------------
# Discriminator
# ---------------------------------------------------------------------------

class BLGANDiscriminator(nn.Module):
    """BL-GAN Discriminator D.

    Scores (bug, code file) pairs via:
      - Bug encoder: Transformer over token IDs → b_D
      - Code content encoder: GCN on AST graph → c_tilde
      - File path encoder: BiLSTM over path token IDs → c_hat
      - c = cat(c_tilde, c_hat)
      - score = MLP(cat(b_D, c))
      - D(c|b) = sigmoid(score)

    All sub-encoders share one nn.Embedding (D's own embedding, separate from G).

    Args:
        vocab_size        : Vocabulary size (including PAD=0 and UNK=1).
        embed_dim         : Shared embedding / model dimension (default 300).
        nhead             : Transformer attention heads (default 6).
        transformer_layers: TransformerEncoder depth (default 2).
        gcn_layers        : GCN depth (default 3).
        path_hidden       : BiLSTM hidden dim per direction (default 256).
        path_lstm_layers  : BiLSTM num_layers (default 2).
        dropout           : Dropout probability (default 0.5).
    """

    def __init__(
        self,
        vocab_size: int,
        embed_dim: int = 300,
        nhead: int = 6,
        transformer_layers: int = 2,
        gcn_layers: int = 3,
        path_hidden: int = 256,
        path_lstm_layers: int = 2,
        dropout: float = 0.5,
    ):
        super().__init__()
        # Shared embedding for all D sub-encoders
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)

        self.bug_encoder = BugTransformerEncoder(
            self.embedding, embed_dim, nhead, transformer_layers, dropout
        )
        self.code_encoder = CodeGCNEncoder(
            self.embedding, embed_dim, gcn_layers, dropout
        )
        self.path_encoder = FilePathBiLSTM(
            self.embedding, embed_dim, path_hidden, path_lstm_layers, dropout
        )

        # code_dim = c_tilde (embed_dim) ⊕ c_hat (path_hidden * 2)
        code_dim = embed_dim + path_hidden * 2
        scorer_in = embed_dim + code_dim  # b_D ⊕ c

        self.scorer = nn.Sequential(
            nn.Linear(scorer_in, 512),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 1),
        )

    def score(
        self,
        bug_ids: torch.Tensor,
        bug_lengths: torch.Tensor,
        graph_batch: Batch,
        path_ids: torch.Tensor,
        path_lengths: torch.Tensor,
    ) -> torch.Tensor:
        """Compute raw (pre-sigmoid) relevance scores.

        Args:
            bug_ids     : LongTensor (B, L_bug)  — padded bug token IDs.
            bug_lengths : LongTensor (B,)         — actual bug lengths.
            graph_batch : torch_geometric Batch of AST graphs (B graphs total).
            path_ids    : LongTensor (B, L_path)  — padded path token IDs.
            path_lengths: LongTensor (B,)          — actual path lengths.
        Returns:
            (B,) raw scalar scores.
        """
        b_D = self.bug_encoder(bug_ids, bug_lengths)       # (B, embed_dim)
        c_tilde = self.code_encoder(graph_batch)           # (B, embed_dim)
        c_hat = self.path_encoder(path_ids, path_lengths)  # (B, path_hidden*2)
        c = torch.cat([c_tilde, c_hat], dim=1)             # (B, code_dim)
        pair_vec = torch.cat([b_D, c], dim=1)              # (B, scorer_in)
        return self.scorer(pair_vec).squeeze(1)            # (B,)

    def forward(
        self,
        bug_ids: torch.Tensor,
        bug_lengths: torch.Tensor,
        graph_batch: Batch,
        path_ids: torch.Tensor,
        path_lengths: torch.Tensor,
    ) -> torch.Tensor:
        """Returns sigmoid probability D(c|b) in [0, 1]. Shape: (B,)."""
        return torch.sigmoid(self.score(bug_ids, bug_lengths, graph_batch, path_ids, path_lengths))


# ---------------------------------------------------------------------------
# Generator
# ---------------------------------------------------------------------------

class BLGANGenerator(nn.Module):
    """BL-GAN Generator G.

    Encodes each bug report via its own Transformer encoder, then uses
    a DirectoryTreeAgent to traverse the project's directory tree and
    select a candidate file path via REINFORCE.

    G has its own nn.Embedding — weights are NOT shared with the Discriminator.

    Args:
        vocab_size        : Vocabulary size.
        embed_dim         : Embedding / model dimension (default 300).
        nhead             : Transformer attention heads (default 6).
        transformer_layers: TransformerEncoder depth (default 2).
        agent_hidden      : Tree agent LSTM hidden dim (default 256).
        agent_layers      : Tree agent LSTM num_layers (default 2).
        dropout           : Dropout probability (default 0.5).
    """

    def __init__(
        self,
        vocab_size: int,
        embed_dim: int = 300,
        nhead: int = 6,
        transformer_layers: int = 2,
        agent_hidden: int = 256,
        agent_layers: int = 2,
        dropout: float = 0.5,
    ):
        super().__init__()
        # Generator's own embedding — separate from Discriminator
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)

        self.bug_encoder = BugTransformerEncoder(
            self.embedding, embed_dim, nhead, transformer_layers, dropout
        )
        self.tree_agent = DirectoryTreeAgent(
            self.embedding, embed_dim, agent_hidden, agent_layers
        )

    def encode_bug(self, bug_ids: torch.Tensor, bug_lengths: torch.Tensor) -> torch.Tensor:
        """Encode a batch of bug reports.

        Args:
            bug_ids    : LongTensor (B, L).
            bug_lengths: LongTensor (B,).
        Returns:
            (B, embed_dim) bug representations.
        """
        return self.bug_encoder(bug_ids, bug_lengths)

    def traverse(
        self,
        b_G: torch.Tensor,
        dir_tree: dict,
        device: torch.device,
        greedy: bool = False,
    ) -> Tuple[str, torch.Tensor]:
        """Run one tree traversal episode for a single bug encoding.

        Args:
            b_G     : (embed_dim,) single bug encoding (with grad).
            dir_tree: Directory tree dict (see DirectoryTreeAgent.traverse).
            device  : Torch device.
            greedy  : Whether to use greedy selection.
        Returns:
            (file_path: str, episode_log_prob: scalar Tensor with grad)
        """
        return self.tree_agent.traverse(b_G, dir_tree, device, greedy=greedy)


# ---------------------------------------------------------------------------
# Top-level BLGAN wrapper
# ---------------------------------------------------------------------------

class BLGAN(nn.Module):
    """Top-level BL-GAN wrapper holding Discriminator and Generator.

    Args:
        params (dict): Keys:
            vocab_size, embed_dim, nhead, transformer_layers,
            gcn_layers, path_hidden, path_lstm_layers,
            agent_hidden, agent_layers, dropout.
    """

    def __init__(self, params: dict):
        super().__init__()
        self.discriminator = BLGANDiscriminator(
            vocab_size=params['vocab_size'],
            embed_dim=params['embed_dim'],
            nhead=params['nhead'],
            transformer_layers=params['transformer_layers'],
            gcn_layers=params['gcn_layers'],
            path_hidden=params['path_hidden'],
            path_lstm_layers=params['path_lstm_layers'],
            dropout=params['dropout'],
        )
        self.generator = BLGANGenerator(
            vocab_size=params['vocab_size'],
            embed_dim=params['embed_dim'],
            nhead=params['nhead'],
            transformer_layers=params['transformer_layers'],
            agent_hidden=params['agent_hidden'],
            agent_layers=params['agent_layers'],
            dropout=params['dropout'],
        )

    def discriminator_parameters(self):
        """Iterator over Discriminator parameters."""
        return self.discriminator.parameters()

    def generator_parameters(self):
        """Iterator over Generator parameters."""
        return self.generator.parameters()
