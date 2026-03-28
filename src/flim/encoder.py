"""
CodeBERT Chunked Bi-Encoder for FLIM.

No-truncation policy (user requirement):
  Every text is tokenised in full, then split into overlapping chunks of
  ≤ CHUNK_SIZE tokens.  Each chunk is encoded independently via CodeBERT's
  [CLS] representation; chunk embeddings are mean-pooled to produce one
  768-dim vector per text.

Fine-tuning strategy (bi-encoder with random chunk sampling):
  • Pre-tokenise all texts into chunk lists (done once).
  • Each training step randomly samples ONE chunk per text — the model
    therefore sees every part of long documents across epochs (data
    augmentation, no information discarded).
  • Bug and function texts are encoded separately (bi-encoder).
  • Loss: BCEWithLogitsLoss on cosine_similarity(bug_emb, func_emb).
"""

import os
import random

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm

CODEBERT_MODEL_NAME = 'microsoft/codebert-base'
CODEBERT_DIM        = 768
CHUNK_SIZE          = 450   # tokens per chunk  (512 − [CLS] − [SEP] = 510; use 450 for safety)
CHUNK_OVERLAP       = 50    # overlapping tokens between consecutive chunks
MIN_CHUNK_LEN       = 5     # discard chunks shorter than this (unless it's the only one)


# ─────────────────────────────────────────────────────────────────────────────
#  Encoder class
# ─────────────────────────────────────────────────────────────────────────────

class CodeBERTEncoder:
    """
    CodeBERT bi-encoder with sliding-window chunking.

    Encoding (inference, no truncation):
        text → tokenise → sliding-window chunks (CHUNK_SIZE, overlap CHUNK_OVERLAP)
             → [CLS] chunk [SEP] → forward → [CLS] embedding per chunk
             → mean-pool all chunk embeddings → 768-dim vector

    Fine-tuning:
        (bug_text, func_text, label) pairs
        → pre-tokenise to chunk lists (one-time)
        → each training step: randomly sample 1 chunk per text (augmentation)
        → forward independently → cosine_similarity → BCE loss
    """

    def __init__(self, model_name: str = CODEBERT_MODEL_NAME, device: str | None = None):
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"[CodeBERTEncoder] Loading {model_name} on {self.device} …")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model     = AutoModel.from_pretrained(model_name)
        self.model.to(self.device)
        self.model.eval()

    # ------------------------------------------------------------------ #
    #  Tokenisation helpers
    # ------------------------------------------------------------------ #

    def _tokenise(self, text: str) -> list[int]:
        """Tokenise text → list of token IDs (no special tokens added)."""
        return self.tokenizer.encode(text, add_special_tokens=False)

    def _chunk_ids(self, ids: list[int]) -> list[list[int]]:
        """
        Split a token-ID list into overlapping windows of ≤ CHUNK_SIZE tokens.

        Examples
        --------
        len(ids)=900, CHUNK_SIZE=450, CHUNK_OVERLAP=50
          chunk 0: ids[   0 : 450]
          chunk 1: ids[ 400 : 850]
          chunk 2: ids[ 800 : 900]   ← shorter last chunk (kept if ≥ MIN_CHUNK_LEN)
        """
        if not ids:
            return [[self.tokenizer.unk_token_id]]

        step   = CHUNK_SIZE - CHUNK_OVERLAP
        chunks = []
        start  = 0
        while start < len(ids):
            end   = min(start + CHUNK_SIZE, len(ids))
            chunk = ids[start:end]
            if len(chunk) >= MIN_CHUNK_LEN or not chunks:
                chunks.append(chunk)
            start += step

        return chunks or [[ids[0]]]

    # ------------------------------------------------------------------ #
    #  Low-level forward pass (shared by inference and fine-tuning)
    # ------------------------------------------------------------------ #

    def _encode_chunk_ids(
        self,
        chunk_ids_batch: list[list[int]],
        no_grad: bool = True,
    ) -> torch.Tensor:
        """
        Forward one batch of chunk token-ID lists through CodeBERT.

        Returns
        -------
        Tensor of shape (len(chunk_ids_batch), 768) — [CLS] embeddings.
        Always on CPU when no_grad=True (inference); on self.device when
        no_grad=False (fine-tuning, gradient needed).
        """
        cls_id = self.tokenizer.cls_token_id
        sep_id = self.tokenizer.sep_token_id
        pad_id = self.tokenizer.pad_token_id

        # Prepend [CLS] and append [SEP] to each chunk
        full_seqs = [[cls_id] + ids + [sep_id] for ids in chunk_ids_batch]
        max_len   = max(len(s) for s in full_seqs)

        input_ids = torch.tensor(
            [s + [pad_id] * (max_len - len(s)) for s in full_seqs],
            dtype=torch.long, device=self.device,
        )
        attn_mask = torch.tensor(
            [[1] * len(s) + [0] * (max_len - len(s)) for s in full_seqs],
            dtype=torch.long, device=self.device,
        )

        if no_grad:
            with torch.no_grad():
                out = self.model(input_ids=input_ids, attention_mask=attn_mask)
            return out.last_hidden_state[:, 0, :].float().cpu()
        else:
            out = self.model(input_ids=input_ids, attention_mask=attn_mask)
            return out.last_hidden_state[:, 0, :].float()  # stays on device

    # ------------------------------------------------------------------ #
    #  Public inference API
    # ------------------------------------------------------------------ #

    def encode(self, text: str, batch_size: int = 64) -> np.ndarray:
        """
        Encode one text → 768-dim float32 numpy array.
        Long texts are chunked; all chunk embeddings are mean-pooled.
        """
        ids    = self._tokenise(text)
        chunks = self._chunk_ids(ids)

        all_embs = []
        for i in range(0, len(chunks), batch_size):
            embs = self._encode_chunk_ids(chunks[i : i + batch_size], no_grad=True)
            all_embs.append(embs.numpy())

        return np.vstack(all_embs).mean(axis=0).astype('float32')

    def encode_batch(self, texts: list[str], batch_size: int = 64) -> np.ndarray:
        """
        Encode a list of texts → (N, 768) float32 array.

        All chunks from all texts are batched together for GPU efficiency.
        Each text's chunks are mean-pooled individually → one row per text.
        """
        if not texts:
            return np.zeros((0, CODEBERT_DIM), dtype='float32')

        # Build flat list: (text_idx, chunk_ids)
        flat: list[tuple[int, list[int]]] = []
        for t_idx, text in enumerate(texts):
            ids = self._tokenise(text)
            for chunk in self._chunk_ids(ids):
                flat.append((t_idx, chunk))

        # Accumulators per text
        emb_acc   = np.zeros((len(texts), CODEBERT_DIM), dtype='float32')
        count_acc = np.zeros(len(texts), dtype='int32')

        for i in range(0, len(flat), batch_size):
            batch      = flat[i : i + batch_size]
            t_idxs     = [item[0] for item in batch]
            chunk_ids  = [item[1] for item in batch]
            embs       = self._encode_chunk_ids(chunk_ids, no_grad=True).numpy()
            for j, t_idx in enumerate(t_idxs):
                emb_acc[t_idx]   += embs[j]
                count_acc[t_idx] += 1

        # Mean pool (avoid divide-by-zero for texts that somehow got no chunks)
        counts = np.maximum(count_acc, 1).reshape(-1, 1)
        return (emb_acc / counts).astype('float32')

    # ------------------------------------------------------------------ #
    #  Fine-tuning
    # ------------------------------------------------------------------ #

    def fine_tune(
        self,
        bug_texts:  list[str],
        func_texts: list[str],
        labels:     list[int],
        epochs:              int   = 3,
        batch_size:          int   = 32,
        lr:                  float = 2e-5,
        max_steps_per_epoch: int   = 5000,
    ) -> None:
        """
        Fine-tune CodeBERT as a bi-encoder on (bug_text, func_text, label) triples.

        No truncation: texts are pre-tokenised into chunk lists.  Each training
        step randomly draws ONE chunk per text — the model sees all parts of
        long documents across epochs (random-crop augmentation).

        Loss
        ----
        BCEWithLogitsLoss(cosine_similarity(bug_emb, func_emb), label)
        """
        n_pos = sum(labels)
        print(f"[CodeBERTEncoder] Fine-tuning on {len(labels)} pairs "
              f"({n_pos} pos / {len(labels)-n_pos} neg).")

        # ── Pre-tokenise (one-time cost) ──────────────────────────────────
        print("[CodeBERTEncoder]   Pre-tokenising bug texts …")
        bug_chunk_lists  = [self._chunk_ids(self._tokenise(t))
                             for t in tqdm(bug_texts,  leave=False)]
        print("[CodeBERTEncoder]   Pre-tokenising function texts …")
        func_chunk_lists = [self._chunk_ids(self._tokenise(t))
                             for t in tqdm(func_texts, leave=False)]

        dataset = _ChunkPairDataset(bug_chunk_lists, func_chunk_lists, labels)
        loader  = DataLoader(
            dataset, batch_size=batch_size, shuffle=True,
            collate_fn=_chunk_pair_collate, drop_last=False,
        )

        # ── Optimiser & loss ──────────────────────────────────────────────
        self.model.train()
        optimizer = optim.AdamW(self.model.parameters(), lr=lr, weight_decay=0.01)
        criterion = nn.BCEWithLogitsLoss()

        cls_id = self.tokenizer.cls_token_id
        sep_id = self.tokenizer.sep_token_id
        pad_id = self.tokenizer.pad_token_id

        # ── Training loop ─────────────────────────────────────────────────
        for epoch in range(epochs):
            total_loss = 0.0
            n_steps    = 0

            for step, (bug_chunks_b, func_chunks_b, labels_b) in enumerate(
                tqdm(loader, desc=f"  Epoch {epoch+1}/{epochs}", leave=False)
            ):
                if step >= max_steps_per_epoch:
                    break

                # bug_chunks_b / func_chunks_b: list of B randomly-sampled chunk-id lists
                # (one chunk per sample, chosen inside _ChunkPairDataset.__getitem__)

                def _pad_and_wrap(chunks: list[list[int]]) -> tuple[torch.Tensor, torch.Tensor]:
                    full    = [[cls_id] + c + [sep_id] for c in chunks]
                    max_len = max(len(f) for f in full)
                    ids  = torch.tensor(
                        [f + [pad_id] * (max_len - len(f)) for f in full],
                        dtype=torch.long, device=self.device,
                    )
                    mask = torch.tensor(
                        [[1]*len(f) + [0]*(max_len - len(f)) for f in full],
                        dtype=torch.long, device=self.device,
                    )
                    return ids, mask

                bug_ids,  bug_mask  = _pad_and_wrap(bug_chunks_b)
                func_ids, func_mask = _pad_and_wrap(func_chunks_b)

                # Separate forward passes (bi-encoder)
                bug_out  = self.model(input_ids=bug_ids,  attention_mask=bug_mask)
                func_out = self.model(input_ids=func_ids, attention_mask=func_mask)

                bug_emb  = bug_out.last_hidden_state[:, 0, :].float()   # (B, 768)
                func_emb = func_out.last_hidden_state[:, 0, :].float()  # (B, 768)

                cos_sim = nn.functional.cosine_similarity(bug_emb, func_emb, dim=1)  # (B,)
                lbl     = labels_b.float().to(self.device)

                loss = criterion(cos_sim, lbl)

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                optimizer.step()

                total_loss += loss.item()
                n_steps    += 1

            avg_loss = total_loss / max(n_steps, 1)
            print(f"[CodeBERTEncoder]   Epoch {epoch+1}: avg_loss={avg_loss:.4f}  steps={n_steps}")

        self.model.eval()

    # ------------------------------------------------------------------ #
    #  Persistence
    # ------------------------------------------------------------------ #

    def save(self, path: str) -> None:
        """Save fine-tuned model weights to disk."""
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        torch.save(self.model.state_dict(), path)
        print(f"[CodeBERTEncoder] Weights saved → {path}")

    def load(self, path: str) -> None:
        """Restore fine-tuned model weights from disk."""
        self.model.load_state_dict(
            torch.load(path, map_location=self.device, weights_only=True)
        )
        self.model.eval()
        print(f"[CodeBERTEncoder] Weights loaded ← {path}")


# ─────────────────────────────────────────────────────────────────────────────
#  Dataset & collate helpers for fine-tuning
# ─────────────────────────────────────────────────────────────────────────────

class _ChunkPairDataset(Dataset):
    """
    Each item is one (bug_chunk, func_chunk, label) triple.
    The chunk is drawn randomly from the pre-tokenised chunk list for that
    text, so the model sees different parts of long texts across epochs.
    """

    def __init__(
        self,
        bug_chunk_lists:  list[list[list[int]]],
        func_chunk_lists: list[list[list[int]]],
        labels:           list[int],
    ):
        self.bug_chunk_lists  = bug_chunk_lists
        self.func_chunk_lists = func_chunk_lists
        self.labels           = labels

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int):
        bug_chunk  = random.choice(self.bug_chunk_lists[idx])
        func_chunk = random.choice(self.func_chunk_lists[idx])
        return bug_chunk, func_chunk, self.labels[idx]


def _chunk_pair_collate(batch):
    """Collate variable-length chunk-id lists into Python lists + a label tensor."""
    bug_chunks  = [item[0] for item in batch]
    func_chunks = [item[1] for item in batch]
    labels      = torch.tensor([item[2] for item in batch], dtype=torch.long)
    return bug_chunks, func_chunks, labels
