# TRANP-CNN: Technical Implementation Guide

**Full Name:** Transfer Natural and Programming Language CNN
**Paper:** "Deep Transfer Bug Localization" — Xuan Huo, Ferdian Thung, Ming Li, David Lo, Shu-Ting Shi
**Published:** IEEE Transactions on Software Engineering, Vol. 47, No. 7, July 2021
**Our Implementation:** `src/tranp_cnn/model.py`, `src/tranp_cnn/pipeline.py`, `src/tranp_cnn/data_prep.py`

---

## 1. What Problem Does It Solve?

Bug localization is the task of, given a bug report written in natural language, ranking all source code files in a project to identify which file(s) contain the bug.

The **cross-project** variant (what we care about) asks: can we train on Project A's bug history and use that model to localise bugs in Project B — a project it has never seen?

TRANP-CNN's answer is: **share the feature extraction CNNs** across projects, but keep **project-specific prediction heads** (fully-connected layers) so the scoring function can adapt. Transfer = shared weights. Natural + Programming = two separate CNNs for the two input modalities.

---

## 2. High-Level Architecture

```
Bug Report (text)          Source Code File (token sequences)
      │                              │
   [N_CNN]                        [P_CNN]
  (Natural Language)         (Programming Language)
  384-dim vector               300-dim vector
      │                              │
      └──────────[Concat]────────────┘
                 684-dim vector
                      │
           ┌──────────┴──────────┐
      [fc_source]           [fc_target]
     Linear(684→256→2)    Linear(684→256→2)
           │                     │
        logits               logits
```

Both N_CNN and P_CNN are **shared** — identical weights used for all projects. Only the final classification heads differ per project. This is the transfer mechanism.

---

## 3. Input Representation

### Bug Report Tokenisation
- **Tokeniser:** BAAI/bge-code-v1 (HuggingFace tokeniser, not the embedding model itself)
- **Max length:** 512 tokens
- **Padding:** to 512 with pad token
- **Why 512?** Standard transformer context limit; bug reports are typically short (title + description), so 512 captures the full text.
- **Result:** a 1D integer tensor of shape `(512,)` — each entry is a token ID (vocabulary index)

### Source Code File Tokenisation
This is the key structural difference vs. a standard text CNN. Code has **hierarchical structure**: files contain lines; lines contain tokens. TRANP-CNN respects this hierarchy.

- **Step 1 — Split file into lines (statements):** up to `max_lines = 768` lines
- **Step 2 — Tokenise each line:** up to `max_line_len = 21` tokens per line
- **Result:** a 2D integer tensor of shape `(768, 21)` — rows = lines, columns = tokens within a line

**Why 21 tokens per line?** 768 lines × 21 tokens = 16,128 max tokens per file. Most source code lines have fewer than 20 meaningful tokens; 21 captures the median well without excessive padding.

**Why 768 lines?** Empirically derived from the distribution of file lengths in the training corpora; 768 covers ~95% of files.

---

## 4. Embedding Layer

Both bug and code tokens go through a shared **embedding matrix**:

- **Vocab size:** size of bge-code-v1 tokeniser vocabulary (~32,000 tokens)
- **Embedding dim:** 512
- **Initialisation:** randomly initialised, trained end-to-end (not using the actual BGE transformer — just its tokeniser vocabulary)

**Why 512?** The original paper used 100-dim GloVe embeddings. We upgraded to 512-dim to match the dimensionality of the BGE tokeniser vocabulary and to carry more semantic information per token before the CNN. A larger embedding dim allows the CNN to extract richer n-gram features.

**Why not use the full BGE transformer (1536-dim)?** Speed. Loading a 7B-parameter BGE model for every sample in a 192-batch training loop is infeasible. The tokeniser is reused; the embedding matrix is a lightweight learned table that captures distributional semantics through training.

---

## 5. N_CNN — Bug Report Encoder

```python
# Input shape: (batch_size, 512)       ← 512 token IDs per bug report
# After embedding: (batch_size, 512, 512)  ← (batch, seq_len, embed_dim)
# After unsqueeze: (batch_size, 1, 512, 512) ← add channel dim for Conv2d
```

### Why Conv2d for text?

A `Conv2d` with kernel `(k, embedding_dim)` slides a window of `k` words across the entire sequence. At each position, it sees `k` consecutive word embeddings simultaneously — this is equivalent to capturing **k-gram features** (k consecutive words as a unit).

- Kernel size 3 → trigram features ("null pointer exception", "index out of bounds")
- Kernel size 4 → 4-gram features
- Kernel size 5 → 5-gram features

The kernel width equals `embedding_dim` so the kernel covers the full embedding vector at each position. Output after convolution is `(batch, num_filters, seq_len - k + 1, 1)`.

### Filter counts and kernel sizes

| Kernel Size | Num Filters | Output per filter | After max-pool |
|-------------|-------------|-------------------|----------------|
| 3           | 128         | (batch, 128, 510, 1) | (batch, 128) |
| 4           | 128         | (batch, 128, 509, 1) | (batch, 128) |
| 5           | 128         | (batch, 128, 508, 1) | (batch, 128) |

After **global max-pooling** over the sequence dimension and squeezing: each kernel size produces a `(batch, 128)` vector. The three vectors are concatenated:

```
N_CNN output: (batch, 128 × 3) = (batch, 384)
```

**Why 128 filters?** The original paper used 128 for bug reports. 128 is large enough for each filter to specialise in detecting a different kind of n-gram pattern (e.g., one filter learns "NullPointerException", another learns "array index"). More filters → more expressive, but diminishing returns and more parameters.

**Why kernel sizes [3, 4, 5]?** Standard Kim (2014) text-CNN configuration. These three sizes capture short-range (3), medium (4), and slightly longer (5) n-gram patterns. Bug reports typically contain short diagnostic phrases, so these small kernels are appropriate.

**Why global max-pooling?** It selects the **most activated position** across the entire sequence for each filter. This gives a fixed-size output regardless of input length and focuses the representation on the most discriminative features found anywhere in the text.

---

## 6. P_CNN — Source Code File Encoder (Hierarchical)

This is the most distinctive component of TRANP-CNN. It processes code in **two stages** to respect the hierarchical line → file structure.

### Stage 1: Within-Statement Convolutions

```
Input per line: (max_line_len, embed_dim) = (21, 512)
Reshape: (1, 21, 512) ← add channel dim
```

For each of the `768` lines independently:
- Apply Conv2d kernels of sizes [3, 4, 5], each with **100 filters**
- Kernel covers `(k, 512)` — reads k consecutive tokens within a line
- After max-pool: each kernel size produces `(100,)` vector
- Concatenate: `(100 × 3,)` = `(300,)` per line

This step produces a **statement vector** for each line — a 300-dim summary of what tokens appear in that line.

**Why process lines individually first?** A line like `int count = getChildren().size();` has internal token-level patterns that are semantically meaningful before considering its relation to adjacent lines.

**Why 100 filters here?** Fewer than the bug-report CNN (128) because code tokens are more syntactically regular; you need fewer pattern detectors per line. 100 is a parameter inherited from the original paper.

### Stage 2: Between-Statement Convolutions

```
Input: (batch, 768, 300)   ← 768 statement vectors of 300-dim each
Reshape: (batch, 1, 768, 300)
```

Now apply Conv2d kernels of sizes **[3, 5, 7]**, each with **100 filters**:

| Kernel Size | What it captures |
|-------------|-----------------|
| 3           | 3 consecutive lines — local code blocks |
| 5           | 5 consecutive lines — small functions/loops |
| 7           | 7 consecutive lines — medium-sized code segments |

After max-pool over the 768 positions: each kernel produces `(100,)`. Concatenate three:
```
P_CNN output: (batch, 100 × 3) = (batch, 300)
```

**Why different kernel sizes from Stage 1?** Stage 2 operates at the line level, where relevant patterns span more tokens (3–7 lines = a loop, a try-catch block, a method body). Using odd-sized kernels [3, 5, 7] ensures symmetrical context windows.

**Why global max-pool across lines?** Same reasoning as N_CNN — focus on the most discriminative line-level pattern found anywhere in the file.

### Memory-Optimised Variant: P_CNN_Parallelized

Processing 768 lines through Stage 1 simultaneously requires holding all line embeddings in GPU memory. For large batches this OOM-crashes. The parallelized variant chunks the 768 lines into blocks of `chunk_size=5000` tokens and processes them sequentially, writing results to a pre-allocated buffer. This avoids OOM while maintaining the same output.

---

## 7. Feature Fusion and Classification Heads

After both encoders:

```
Bug vector:  (batch, 384)   from N_CNN
Code vector: (batch, 300)   from P_CNN

Concatenated: (batch, 684)
```

This concatenated vector is fed to **project-specific** fully-connected layers:

```python
fc_source = nn.Sequential(
    nn.Linear(684, 256),
    nn.ReLU(),
    nn.Dropout(0.5),
    nn.Linear(256, 2)        # 2 logits: not-buggy, buggy
)

fc_target = nn.Sequential(
    nn.Linear(684, 256),
    nn.ReLU(),
    nn.Dropout(0.5),
    nn.Linear(256, 2)
)
```

**Why 256 hidden dim?** A bottleneck that forces the model to compress the 684-dim joint representation. Larger than 684 would expand the space unnecessarily; much smaller would lose information. 256 (≈684/2.7) is a standard intermediate compression.

**Why separate heads per project?** The CNN features are shared (transfer), but the scoring function must adapt to each project's code style, bug report style, and vocabulary distribution. Separate heads allow this calibration without interfering with the shared feature extraction.

**Why 2 output logits instead of 1 score?** Cross-entropy loss with class labels (0=not-buggy, 1=buggy) is more stable than sigmoid binary cross-entropy in practice. The probability of class 1 (softmax output[1]) is used as the ranking score.

---

## 8. Loss Function

### Class Imbalance
Bug localisation is extremely imbalanced: for each bug, there are 1–3 ground-truth files but ~300 candidates. So ~99% of samples are labelled 0 (not-buggy).

**Solution:** Weighted cross-entropy. Class weights are computed as:

```python
class_weights = total_samples / (num_classes * class_counts)
# weight[1] (buggy class) ≈ 100× weight[0]
```

This forces the model to pay more attention to the rare positive examples.

### Hybrid Loss (Cross-Project Scenario)

When both source and target data are available (CP-transfer scenario):

```
L_total = L_source + L_target
```

Both project losses are summed. The shared CNN layers receive gradients from both, reinforcing general patterns. Each head only receives gradients from its own project, learning project-specific scoring.

---

## 9. Training Configuration

| Parameter | Value | Why |
|-----------|-------|-----|
| Epochs | 20 | Enough for convergence; early stopping prevents overfitting |
| Batch size | 192 | Large batch for stable gradient estimates; fits GPU memory with mixed precision |
| Learning rate | 0.001 | Standard Adam LR |
| Weight decay | 1e-5 | L2 regularisation to prevent overfitting on small target datasets |
| Optimizer | AdamW | Adam + decoupled weight decay |
| Mixed precision | Yes (torch.amp) | Cuts GPU memory use in half, 1.5–2× speedup |
| Early stopping | 3 epochs patience | Prevents overfitting on WP-small (tiny training sets) |
| Validation split | 20% of training set | Used only for early stopping criterion |

---

## 10. Data Pipeline (`data_prep.py`)

### FAISS Candidate Generation

Rather than evaluating every (bug, file) pair (millions of pairs), we use FAISS to pre-filter to the top 300 candidates:

1. For each commit: build a `faiss.IndexFlatIP` (inner-product, with L2-normalised embeddings = cosine similarity) from all file embeddings at that commit snapshot.
2. Encode the bug report embedding (from BGE transformer — the embedding database pre-built separately).
3. Search for top-300 nearest files.
4. Force-add ground-truth files to the training set (even if FAISS misses them).

This reduces the per-bug candidate space from thousands of files to 300, making batch training feasible.

### Memory-Mapped Preprocessing

Tokenised code tensors are large: `(num_unique_blobs, 768, 21)` integers. For a project with 10,000 unique blobs this is `10,000 × 768 × 21 × 2 bytes = ~320 MB`. Storing this in RAM for multiple projects simultaneously causes OOM.

**Solution:** Write the array to a `.npy` file backed by a memory-mapped file. Each blob is tokenised once and written to its position in the mmap. At training time, only the accessed rows are loaded from disk. This gives O(unique_blobs) disk space with O(batch_size) RAM per step.

---

## 11. Enhancements vs. Original Paper

| Original Paper | Our Implementation | Reason |
|----------------|-------------------|--------|
| GloVe 100-dim embeddings | BGE-code-v1 tokeniser vocabulary, 512-dim learned embeddings | BGE tokeniser is code-aware; 512-dim carries more information |
| BM25 keyword candidates | FAISS cosine similarity candidates | FAISS captures semantic similarity, not just keyword overlap |
| Numpy-based evaluation | Full PyTorch training with mixed precision | Speed and memory efficiency |
| Single project evaluation | 4-scenario cross-project framework (WP-small, WP-large, CP-cold-start, CP-transfer) | Research contribution: systematic CPL evaluation |
| No FAISS score baseline | Saves FAISS rank and model rank per bug | Enables rank_displacement analysis (how much model improves over FAISS) |
| No memory mapping | Memory-mapped code arrays | Handles large datasets without RAM exhaustion |
| AdaGrad optimizer | AdamW with weight decay | More stable convergence |

---

## 12. Evaluation Metrics

All metrics are computed over the **test set** (held-out 20% of target project bugs).

For each bug, the model scores all ~300 candidates and ranks them.

| Metric | Formula | Intuition |
|--------|---------|-----------|
| **Top-1** | % of bugs where ground-truth file is rank 1 | How often does the model get it exactly right? |
| **Top-5** | % of bugs where ground-truth file is in top 5 | Is the file easy to find with a few checks? |
| **Top-10** | % of bugs where ground-truth file is in top 10 | Is the file findable with a small review? |
| **MRR** | mean(1/rank) over all bugs | Weighted by rank quality — rank 1 = 1.0, rank 2 = 0.5, rank 10 = 0.1 |
| **MAP** | mean of average precision over all bugs | Area under precision-recall curve; sensitive to all relevant files |

---

## 13. Quick Reference: All Hyperparameters

```python
# Architecture
nl_embedding_dim      = 512
code_embedding_dim    = 512
nl_kernels            = 128       # filters per kernel size in N_CNN
nl_kernel_sizes       = [3, 4, 5]
stmt_kernels          = 100       # filters per kernel size, Stage 1 P_CNN
stmt_kernel_sizes     = [3, 4, 5]
file_kernels          = 100       # filters per kernel size, Stage 2 P_CNN
file_kernel_sizes     = [3, 5, 7]
hidden_dim            = 256       # FC hidden layer
num_classes           = 2
dropout               = 0.5

# Input shapes
max_bug_len           = 512       # tokens in bug report
max_lines             = 768       # lines per source file
max_line_len          = 21        # tokens per line

# Training
EPOCHS                = 20
BATCH_SIZE            = 192
LEARNING_RATE         = 0.001
WEIGHT_DECAY          = 1e-5
TOP_K_CANDIDATES      = 300
EARLY_STOPPING_PAT    = 3         # epochs without val improvement

# Derived shapes
N_CNN output          = 128 × 3 = 384
P_CNN output          = 100 × 3 = 300
Fusion input          = 384 + 300 = 684
FC hidden             = 256
FC output             = 2 (logits)
```
