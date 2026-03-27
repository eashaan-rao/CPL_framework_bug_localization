# COOBA: Technical Implementation Guide

**Full Name:** CooBa — Cooperative and Adversarial Bug Localization
**Paper:** "CooBa: Cross-project Bug Localization via Adversarial Transfer Learning"
        — Ziye Zhu, Yun Li, Hanghang Tong, Yu Wang (Nanjing University / UIUC)
**Our Implementation:** `src/cooba/model.py`, `src/cooba/pipeline.py`, `src/cooba/ast_parsers.py`, `src/cooba/utils.py`

---

## 1. What Problem Does COOBA Address That TRANP-CNN Does Not?

TRANP-CNN transfers by sharing CNN weights across projects. Its weakness: **negative transfer** — if source and target projects are in very different domains (e.g., a scientific library vs. a web framework), the shared features may actively hurt target performance because the source patterns are misleading.

COOBA's answer: use **adversarial training** to explicitly separate features into two types:

- **Public (transferable) features:** patterns that are universally useful across projects — general code structure, common bug patterns.
- **Private (project-specific) features:** patterns unique to a project — its idioms, domain vocabulary, specific API usage.

The adversarial component (discriminator) *enforces* that public features carry no project identity signal. If the discriminator cannot tell "is this a source-project file or a target-project file?" from the public features, those features are genuinely project-agnostic and safe to transfer.

The "cooperative" part refers to how the two types of features work together: public features transfer, private features adapt. "Adversarial" refers to the GAN-style training between the feature extractor and the discriminator.

---

## 2. The Two Modes of Operation

Depending on the scenario, COOBA runs in one of two modes:

| Mode | When Used | Scenarios | Difference |
|------|-----------|-----------|------------|
| **within-project** | No source data | WP-small, WP-large | Single extractor, no discriminator, task loss only |
| **cross-project** | Source data available | CP-cold-start, CP-transfer | Dual extractors, discriminator, task + adversarial loss |

---

## 3. Full Architecture Overview

```
BUG REPORT (text)                    SOURCE CODE FILE
      │                                      │
  [BGE Embed]                           [BGE Embed]
  (1536-dim)                            (1536-dim)
      │                                      │
[BugReportEncoder]           ┌──────────────┴────────────────┐
  Bi-LSTM (2 layers)    [SharedExtractor]            [IndividualExtractor]
  hidden=256            CNN (public features)         GCN (private features)
  Output: 512-dim       3 kernels × 128 filters       GCNConv × 2 layers
      │                 Output: 384-dim                Output: 256-dim
      │                      │                              │
      │                      ├──── [ProjectDiscriminator] ──┤  (adversarial)
      │                      │     MLP → source/target?
      │                      │
      │                 [FeatureFusion]
      │                 concat(384 + 256) = 640
      │                 MLP → 256-dim
      │                      │
  [bug_projection]      [code_projection]
  Linear(512→256)        Linear(256→256)
      │                      │
      └──── cosine_similarity ────┘
                  │
            relevance score
            (scalar per pair)
```

---

## 4. Component 1: Bug Report Encoder (Bi-LSTM)

### Why a Bi-LSTM for bug reports instead of a CNN?

TRANP-CNN uses a CNN for bug reports (N_CNN). COOBA switches to a **Bidirectional LSTM**. The reason:

- CNN captures local n-gram patterns, which is great for detecting specific phrases.
- LSTM captures **sequential dependencies** — it maintains a hidden state that accumulates context as it reads through the text. This means the word "null" at position 3 can influence the representation of "exception" at position 47.
- Bidirectional: reads the bug text both left-to-right and right-to-left, so each token's representation is informed by all other tokens in both directions.

Bug reports often have long-range dependencies: "...when initialising the X component, if Y is true, then Z crashes with a NullPointerException" — the LSTM can connect "initialising" to "crashes" across a long span; a CNN with kernel size 5 cannot.

### Architecture Details

```python
BugReportEncoder(
    embedding_dim = 1536,   # BGE embedding input dim
    hidden_size   = 256,    # hidden state per direction
    num_layers    = 2,      # stacked LSTM layers
    bidirectional = True,   # forward + backward
    dropout_prob  = 0.5
)
```

**Input:** A single 1536-dim BGE embedding vector (mean-pooled over the bug text tokens).

**Wait — if the input is a single vector (seq_len=1), why use an LSTM at all?**
The intent was to pass a sequence of token-level embeddings. In the current implementation, the BGE model produces one mean-pooled embedding per bug report, so `seq_len=1`. The LSTM effectively acts as a linear projection (`1536 → 512`). This is a simplification vs. the original paper (which would embed each token separately). The infrastructure is in place for a future upgrade to token-level sequences.

**Output:**
```
hidden: (num_layers × 2, batch, hidden_size) = (4, batch, 256)
Take last layer's forward hidden:  hidden[-2] → (batch, 256)
Take last layer's backward hidden: hidden[-1] → (batch, 256)
Concatenate → (batch, 512)
```

**`pack_padded_sequence`:** Even with seq_len=1, the LSTM uses pack/unpack for correctness and forward compatibility with variable-length sequences. With all lengths=1, this is a no-op overhead that causes no issues.

---

## 5. Component 2: SharedExtractor (CNN for Public Features)

This is a **text CNN** applied to the code embedding — essentially the same design as N_CNN in TRANP-CNN, but applied to code instead of bug reports.

```python
SharedExtractor(
    embedding_dim = 1536,
    num_filters   = 128,
    kernel_sizes  = [3, 4, 5],
    dropout_prob  = 0.5
)
```

### Input and Shape Flow

```
code_embeddings shape: (batch, 1, 1536)   ← one mean-pooled BGE vector per file
                       (batch, seq_len=1, embed_dim=1536)

After unsqueeze: (batch, 1, 1, 1536)      ← add channel dim for Conv2d
```

### Why Conv2d on code?

Same reasoning as TRANP-CNN's N_CNN: Conv2d with kernel `(k, 1536)` extracts k-gram patterns. Since `seq_len=1` in the current implementation (same simplification as the bug encoder — one embedding per file), the convolution effectively acts as a linear projection with different kernel sizes. The infrastructure supports upgrading to chunk-level embeddings (multiple vectors per file).

### Filter Counts and Why

Each of the 3 kernel sizes runs 128 filters in parallel:

| Kernel | Filters | Output before pool | After max-pool |
|--------|---------|-------------------|----------------|
| 3      | 128     | (batch, 128, −, 1) | (batch, 128) |
| 4      | 128     | (batch, 128, −, 1) | (batch, 128) |
| 5      | 128     | (batch, 128, −, 1) | (batch, 128) |

Concatenated: **(batch, 384)** — the public feature vector.

**Why 128 filters and [3,4,5] sizes?** Inherited from the original COOBA paper (which used these for the shared CNN component). It matches N_CNN in TRANP-CNN, allowing fair comparison.

**Why is this called "public"?** This extractor has **no copy per project** — it is shared across source and target. The adversarial training forces its outputs to be project-neutral. That is why these features are "public" or "transferable" — they describe general code characteristics, not project-specific ones.

---

## 6. Component 3: IndividualExtractor (GCN for Private Features)

This is where COOBA fundamentally differs from TRANP-CNN: it uses **Abstract Syntax Trees (ASTs)** and **Graph Convolutional Networks (GCNs)** to extract project-specific structural patterns.

### Why Use ASTs and GCNs?

Code is not just text — it has a **tree structure** (syntax tree) that captures the grammar and logical structure. Two files can have very different text but similar AST shapes (both define a class with a try-catch block, for instance), or very similar text but different structures.

GCNs learn to aggregate information from neighbouring nodes in a graph. For an AST, this means each node's representation is informed by its parent, siblings, and children — capturing structural context that text-level processing misses.

### AST Parsing (from `ast_parsers.py`)

**Python files:** `ast` standard library
```python
tree = ast.parse(source_code)
# Traverse recursively
# Each AST node → a graph node
# Parent→child relation → a directed edge
```

**Java files:** `javalang` library
```python
tree = javalang.parse.parse(source_code)
# Same traversal logic
```

**Max nodes:** 500. Files with more AST nodes are truncated (first 500 nodes, edges filtered to keep only those within the first 500).

**Node features:** Each node in the AST gets the **same** BGE embedding of the whole file (mean-pooled, 1536-dim). This is a simplification — ideally each node would be embedded from its code snippet. But it is a reasonable placeholder: the structural information (which node connects to which) is captured by the GCN; the semantic content comes from the shared BGE embedding.

**Edge_index:** A `(2, num_edges)` tensor listing all parent→child connections. An empty graph (parse failure) becomes a single node with no edges.

**Output of parsing:** A `torch_geometric.data.Data` object:
```
graph_data.x          = (num_nodes, 1536)  # node features
graph_data.edge_index = (2, num_edges)     # adjacency
```

### GCN Architecture

```python
IndividualExtractor(
    node_embedding_dim = 1536,
    hidden_dim         = 256,
    output_dim         = 256,
    num_layers         = 2,
    dropout_prob       = 0.5
)
```

**Two GCN layers:**
```
Layer 1: GCNConv(1536 → 256)
         → skip BatchNorm1d if batch has only 1 node
         → ReLU + Dropout(0.5)

Layer 2: GCNConv(256 → 256)     ← output layer, no activation
```

**Why GCNConv?** GCNConv (Kipf & Welling 2016) is the standard graph convolution:
```
h_v = W × mean(h_u for all neighbours u of v, including v itself)
```
Each node aggregates its own features with its neighbours' features, weighted by the graph's degree matrix. This propagates structural information through the tree.

**Why only 2 layers?** Deeper GCNs suffer from "over-smoothing" — after many layers, all node representations converge to the same vector. For trees (sparse graphs), 2 layers means each node sees its grandparent and grandchildren. That is sufficient structural context for bug localization; deeper would wash out the local structure.

**Why hidden_dim = output_dim = 256?** Matches the fusion layer's expected private_dim. 256 is sufficient to capture structural patterns without excessive parameters.

**Why skip BatchNorm when only 1 node?** `BatchNorm1d` requires at least 2 samples to compute mean/variance. If a batch contains graphs that together have only 1 AST node, this crashes. Fixed by the `x.shape[0] > 1` guard.

**Global Mean Pooling:**
```python
return global_mean_pool(x, batch)
# x shape: (total_nodes_in_batch, 256)
# batch:   (total_nodes,) — which graph each node belongs to
# output:  (batch_size, 256) — one vector per graph
```
This aggregates all node representations into one graph-level vector by taking the mean. Max pooling would emphasise the most activated node; mean pooling treats all nodes equally — more appropriate for trees where all nodes contribute to the structure.

**Two separate instances (source and target):**
```python
self.individual_extractor_source = IndividualExtractor(...)  # learns source idioms
self.individual_extractor_target = IndividualExtractor(...)  # learns target idioms
```
These are the "individual" (private/project-specific) components. Separate parameters mean each can specialise in its project's code patterns without interfering with the other.

---

## 7. Component 4: ProjectDiscriminator (Adversarial Component)

```python
ProjectDiscriminator(
    input_dim   = 384,   # public features from SharedExtractor
    hidden_dim  = 128,
    dropout_prob = 0.5
)

Architecture:
  Linear(384 → 128) + LayerNorm(128) + ReLU + Dropout(0.5)
  Linear(128 →  64) + LayerNorm(64)  + ReLU + Dropout(0.5)
  Linear( 64 →   2)   # logits: source=0, target=1
```

### Why LayerNorm Instead of BatchNorm1d?

`BatchNorm1d` normalises across the batch dimension — it needs ≥ 2 samples to compute mean/std. When the last mini-batch has only 1 sample, it crashes.

`LayerNorm` normalises across the feature dimension for each sample independently. `LayerNorm(128)` computes mean and std of the 128 features for each sample, then normalises. This works with batch_size=1.

This switch was made during our implementation after encountering `ValueError: Expected more than 1 value per channel when training, got input size torch.Size([1, 384])` on the last batch of WP-large.

### What Does the Discriminator Do?

It is a binary classifier: given a public feature vector (384-dim), predict whether it came from the source project or the target project.

- **If the discriminator easily succeeds** → the public features carry project-identity information → the SharedExtractor has failed at being project-neutral.
- **If the discriminator performs at random chance (50%)** → public features are indistinguishable across projects → successful domain invariance.

### The Adversarial Game (GAN-style)

In each training step, two opposing objectives are optimised:

**Step 1 — Train the Discriminator (to succeed):**
```python
# Get public features from both domains (with torch.no_grad() so no gradients to model)
source_public = model(..., project_type='source')
target_public = model(..., project_type='target')

# Labels: source=0, target=1
disc_logits = discriminator(concat([source_public, target_public]))
disc_loss = CrossEntropyLoss(disc_logits, [0..., 1...])
disc_loss.backward()
optimizer_disc.step()
```

**Step 2 — Train the Generator (to fool the discriminator):**
```python
# Recompute with gradients flowing to model
source_public = model(..., project_type='source')

# Fool discriminator: label source as target (1)
fool_logits = discriminator(source_public)
adv_loss = CrossEntropyLoss(fool_logits, [1...])   # want it to think source is target

# Combined with task loss
total_loss = task_loss + λ_adv × adv_loss
total_loss.backward()
optimizer_main.step()
```

**Why only fool the discriminator with source features?** We want source features to look like target features (project-neutral). We already have explicit target training data when it is available; the adversarial signal comes from pushing source public features into the target-indistinguishable zone.

**λ_adv = 0.1:** The adversarial loss is weighted at 10% of the task loss. Too high and the model sacrifices bug localization accuracy to fool the discriminator; too low and domain adaptation is ineffective. 0.1 is from the original paper.

---

## 8. Component 5: FeatureFusion (Project-Specific MLP)

```python
FeatureFusion(
    public_dim   = 384,
    private_dim  = 256,
    output_dim   = 256,
    hidden_dim   = 384,
    dropout_prob = 0.5
)

Architecture:
  input: concat(public_384, private_256) = 640-dim
  Linear(640 → 384) + LayerNorm(384) + ReLU + Dropout(0.5)
  Linear(384 → 256) + LayerNorm(256) + ReLU + Dropout(0.5)
  Linear(256 → 256)
  output: 256-dim code vector
```

**Why two separate fusion instances (source and target)?** The fusion MLP learns how to combine public and private features optimally for each project. Source and target have different relationships between structural patterns (private) and general code features (public), so separate fusion layers allow independent calibration.

**Why hidden_dim = 384?** Intermediate between input (640) and output (256). Compresses gradually rather than jumping directly from 640 to 256.

**Why LayerNorm throughout?** Same reason as the discriminator — handles single-sample last batches without crashing.

---

## 9. Relevance Scoring

After encoding bug and code:

```python
bug_vector  = bug_projection(bug_vector)   # Linear(512 → 256)
code_vector = code_projection(code_vector) # Linear(256 → 256)

score = F.cosine_similarity(bug_vector, code_vector, dim=1)
# score ∈ [-1, 1], higher = more similar = more likely buggy file
```

**Why cosine similarity instead of a classifier (like TRANP-CNN)?** Cosine similarity directly measures directional alignment between the bug and code representations. It produces a continuous score suitable for ranking without needing class labels and cross-entropy. It also normalises for magnitude — a long bug report doesn't automatically score higher than a short one.

**Why projection layers before cosine similarity?** The bug vector comes from the LSTM (512-dim) and the code vector from fusion (256-dim) — different dimensions. Projecting both to 256 makes them comparable. The projection also adds a learnable linear transformation that can adjust the embedding space for cosine comparison.

---

## 10. Task Loss: Margin Ranking Loss

```python
task_criterion = nn.MarginRankingLoss(margin=0.4)
```

For each batch, we have positive pairs (bug, buggy-file) and negative pairs (bug, non-buggy-file). The loss enforces:

```
score(bug, positive_file) > score(bug, negative_file) + margin

L = max(0, margin - (pos_score - neg_score))
```

**Why margin = 0.4?** The margin creates a buffer zone. Without it, a model that scores positive=0.51 and negative=0.50 satisfies the ranking constraint but is fragile. A margin of 0.4 forces positive files to score at least 0.4 higher than negative files, creating more confident separations.

**Why MarginRankingLoss instead of cross-entropy?** Bug localization is fundamentally a **ranking task**: we care about the relative order (does the ground-truth file rank above the others?), not absolute class probabilities. Ranking losses directly optimise the ordering, which aligns with the evaluation metrics (Top-K, MRR, MAP).

**Handling imbalance:** For each batch, we take `n_pairs = min(len(positive_scores), len(negative_scores))` pairs. This balances the loss without class weights — each positive sample is matched with one negative sample.

**Skipping backward when no valid pairs:** If a batch happens to have only positives or only negatives (rare with imbalanced data), the task loss is a zero tensor with no grad_fn. We skip `backward()` in this case to avoid the "element 0 does not require grad" error.

---

## 11. Training Loop Details (`pipeline.py`)

### Within-Project Mode (WP-small, WP-large)
```python
for epoch in range(EPOCHS):
    for batch in target_loader:
        bug_emb, bug_len, code_emb, graph = batch tensors
        scores, public_feat = model(bug_emb, bug_len, code_emb, graph, project_type=None)

        # rank positives above negatives
        task_loss = MarginRankingLoss(pos_scores, neg_scores, ones)

        if task_loss.grad_fn is not None:
            task_loss.backward()
            optimizer.step()

    print(f"Epoch {epoch+1}/{EPOCHS} | Task={avg_task_loss:.4f} [{n_batches} batches]")
```

### Cross-Project Mode (CP-cold-start, CP-transfer)

```python
source_iter = DataLoader(source_dataset)     # 406 prefect bugs
target_iter = cycle(DataLoader(target_dataset))  # 20-25 xarray bugs, cycled

for batch in source_iter:                   # primary loop over source
    # --- Task loss on source ---
    scores, public_feat = model(source_batch, project_type='source')
    task_loss = MarginRankingLoss(...)

    # --- Adversarial step (if target data exists) ---
    target_batch = next(target_iter)        # get next target batch

    # 1. Train discriminator
    with torch.no_grad():
        src_pub = model(source_batch, 'source')
        tgt_pub = model(target_batch, 'target')
    disc_loss = CrossEntropyLoss(discriminator([src_pub, tgt_pub]), [0s, 1s])
    disc_loss.backward()
    optimizer_disc.step()

    # 2. Train generator to fool discriminator
    src_pub = model(source_batch, 'source')  # with gradients
    adv_loss = CrossEntropyLoss(discriminator(src_pub), [1s])  # fool: say source=target

    total_loss = task_loss + 0.1 × adv_loss
    total_loss.backward()
    optimizer_main.step()
```

**CP-cold-start:** `target_train_ids = []` → no target loader → `target_iter = None` → adversarial step is skipped → only task loss on source.

**CP-transfer:** Both source and target data present → full adversarial training.

---

## 12. Caching Strategy

Computing BGE embeddings and AST graphs is expensive (~1–6 seconds per file). The cache stores pre-computed (code_embeddings, graph_data) tuples:

```
data/processed/cooba_cache/{project}_{scenario}/{blob_sha}.pt
```

**Key design choice:** Cache key = blob SHA. Since a blob SHA uniquely identifies the content of a file at a specific commit, caching by SHA means:
- If two bugs share the same candidate file (same blob SHA), it is processed only once.
- The cache is correct across epochs (same content = same features).
- Cache is separate per scenario because the cache directory naming includes scenario — a design decision that prevents cross-scenario contamination but wastes disk space if blob sets overlap.

**Why `.pt` (PyTorch) format?** Saves both a tensor (code_embeddings) and a `Data` object (graph_data) in one file. Supports all PyTorch tensor types natively.

**Cache invalidation:** If the code in `ast_parsers.py` changes (e.g., removing `node_tokens`), old cached files have a different graph structure. The fix was to delete stale cache files before rerunning.

---

## 13. Evaluation (`evaluate_cooba`)

```python
model.eval()
with torch.no_grad():
    for batch in test_loader:
        scores = model(bug_emb, bug_len, code_emb, graph, project_type='target')
        # accumulate: predictions[bug_id].append((blob_sha, score))
        # ground_truths[bug_id] = [gt_blob_shas]

for bug_id in predictions:
    ranked = sorted(predictions[bug_id], by score descending)
    # compute Top-K, MRR, MAP
```

Note: `project_type='target'` is always used at evaluation (we are always evaluating on the target project test set).

---

## 14. Enhancements vs. Original COOBA Paper

| Original Paper | Our Implementation | Reason |
|----------------|-------------------|--------|
| GloVe/Word2Vec embeddings for code | BAAI/bge-code-v1 (1536-dim) | Code-aware embeddings carry more semantic content |
| Single BGE embedding per bug (approximation) | Same — mean-pooled BGE | Simplification; LSTM infrastructure ready for upgrade to token-level |
| Node-level embeddings from code context | Same node embedding for all nodes (mean-pooled file embedding) | Approximation; sufficient for structural GCN learning |
| BatchNorm1d throughout | LayerNorm in MLP components; conditional BatchNorm skip in GCN | Handles single-sample last batches without crashing |
| Multi-project discriminator | Binary discriminator (source vs. target) | Simpler; sufficient for pairwise CPL |
| Not evaluated on CPL framework | 4-scenario framework (WP-small/large, CP-cold-start/transfer) | Our research contribution |
| No FAISS pre-filtering | FAISS top-300 candidates | Same pipeline as TRANP-CNN for fair comparison |
| Python/Java only (original) | Python fully implemented; Java parser ready | Phase 1 scope: Python; Java infrastructure in place |

---

## 15. Design Decision Log (Implementation Changes)

These changes were made during development and are important for paper writing:

| Change | File | Commit Reason |
|--------|------|--------------|
| `bug_length = 1` (not word count) | `pipeline.py` | BGE produces 1 vector per bug (not per token); `pack_padded_sequence` requires length ≤ seq_len |
| Skip `backward()` when `grad_fn is None` | `pipeline.py` | Batches with no pos/neg pairs produce a constant zero loss — calling backward crashes |
| Remove `node_tokens` from `Data` objects | `ast_parsers.py` | List-of-strings attributes break `Batch.from_data_list` when mixed with graphs lacking that attribute |
| `x.shape[0] > 1` guard in GCN | `model.py` | Single-node graphs trigger `BatchNorm1d` crash |
| `LayerNorm` in FeatureFusion and Discriminator | `model.py` | `BatchNorm1d` in `nn.Sequential` cannot be guarded; `LayerNorm` works at batch_size=1 |
| Epoch-summary logging (not tqdm) | `pipeline.py` | tqdm via `tee` creates thousands of lines per epoch; one line per epoch is sufficient |
| BGE_EMBEDDING_DIM = 1536 | `pipeline.py` | Original placeholder was 512; BGE-code-v1 hidden size is 1536 |

---

## 16. Quick Reference: All Hyperparameters

```python
# Embeddings
BGE_MODEL_NAME      = 'BAAI/bge-code-v1'
BGE_EMBEDDING_DIM   = 1536

# Data limits
TOP_K_CANDIDATES    = 300     # FAISS candidates per bug
MAX_BUG_LEN         = 512     # token limit for bug reports
MAX_CODE_LEN        = 1024    # line limit for code
MAX_AST_NODES       = 500     # graph node limit

# Model architecture
bug_encoder:
  hidden_size       = 256
  num_layers        = 2
  dropout_prob      = 0.5
  output_dim        = 512  (256 × 2 bidirectional)

shared_extractor (CNN):
  num_filters       = 128
  kernel_sizes      = [3, 4, 5]
  dropout_prob      = 0.5
  output_dim        = 384  (128 × 3)

individual_extractor (GCN):
  hidden_dim        = 256
  output_dim        = 256
  num_layers        = 2
  dropout_prob      = 0.5

fusion (FeatureFusion):
  input_dim         = 640  (384 public + 256 private)
  hidden_dim        = 384
  output_dim        = 256
  dropout_prob      = 0.5

discriminator:
  input_dim         = 384  (public features)
  hidden_dim        = 128
  output_dim        = 2    (source / target)

projection:
  bug_projection    = Linear(512 → 256)
  code_projection   = Linear(256 → 256)
  similarity        = cosine

# Training
EPOCHS              = 10
BATCH_SIZE          = 32
LEARNING_RATE       = 0.001    (AdamW, main model)
DISC_LEARNING_RATE  = 0.0005   (Adam, discriminator)
WEIGHT_DECAY        = 1e-5
MARGIN              = 0.4      (MarginRankingLoss)
LAMBDA_ADV          = 0.1      (weight of adversarial loss)
```
