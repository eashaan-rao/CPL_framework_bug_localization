"""
BLAZE training and evaluation pipeline.

Entry point: run_blaze_experiment()

Training uses in-batch contrastive learning (NTXentLoss) to fine-tune a
pre-trained transformer with modal-specific ResidualAdapters.

Evaluation reranks the shared FAISS top-300 candidate pool (TOP_K_CANDIDATES),
matching the candidate-generation protocol TRANP-CNN and COOBA use. The
fine-tuned model embeds bug reports and code chunks, then ranks files by
maximum chunk similarity among the FAISS-retrieved candidates only.
"""

import copy
import itertools
import math
import os
import time

import git
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import torch.optim as optim
import transformers
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from tqdm import tqdm

from .model import BlazeEmbedding, NTXentLoss
from .data_prep import (
    BLAZE_CACHE_DIR,
    BLOB_DB_DIR,
    BUG_METADATA_DIR,
    BUG_REPORTS_PATH,
    PROJECTS_METADATA_PATH,
    REPO_BASE_PATH,
    BlazeDataset,
    DynamicCodeSplitter,
    _LANG_MAP,
    _fallback_chunk,
    create_contrastive_samples,
    get_file_language,
    get_path_to_sha_map,
    get_project_path,
    load_project_databases,
    read_blob_content,
)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
RESULT_PATH = "/home/cs21d002_eashaan/PhD/Objective1/results"

# Backbone model — same as the rest of the framework
BGE_MODEL_NAME = "BAAI/bge-code-v1"
BGE_EMBEDDING_DIM = 1536

# Model hyperparameters
RESIDUAL_LAYERS = 2
MAX_BUG_LENGTH = 512   # tokens for bug report
MAX_CODE_LENGTH = 512  # tokens for code chunk

# Loss hyperparameters (from the BLAZE paper)
TAU = 0.3
HARD_NEGATIVE_SCALE = 1.2
HARD_POSITIVE_SCALE = 1.2
POSITIVE_SCALE = 1.001

# Training hyperparameters
EPOCHS = 10
BATCH_SIZE = 4          # small: frozen transformer still holds all activations in VRAM
LEARNING_RATE = 1e-4    # higher LR is fine when only adapters are trained
WEIGHT_DECAY = 1e-2
WARMUP_RATIO = 0.1      # fraction of steps used for LR warm-up
PATIENCE = 3            # early-stopping patience (validation loss)
GRAD_ACCUM_STEPS = 2    # effective batch = BATCH_SIZE × GRAD_ACCUM_STEPS = 8

# Whether to freeze the transformer backbone and train only the adapters.
# Set True (default): trains adapters only — much lower VRAM, faster convergence.
# Set False: full fine-tuning — needs ~30 GB free VRAM for bge-code-v1.
FREEZE_TRANSFORMER = True

# Evaluation batch size (larger: inference is cheaper than training)
EVAL_BATCH_SIZE = 16

# Candidate pool size for evaluation reranking — matches TRANP-CNN/COOBA's
# TOP_K_CANDIDATES so all three models compete over the same FAISS shortlist.
TOP_K_CANDIDATES = 300

# GPU memory guard — pause BLAZE and offload model to CPU when GPU is tight,
# so COOBA shards always have headroom for their occasional spikes to ~45 GB.
GPU_FREE_THRESHOLD_MiB = 8_192    # back off when less than 8 GB is free
GPU_POLL_INTERVAL_S    = 30       # seconds between checks while waiting
GPU_CHECK_EVERY_N_BATCHES = 20    # how often to check mid-epoch


# ---------------------------------------------------------------------------
# GPU guard helpers
# ---------------------------------------------------------------------------

def _gpu_free_mib() -> float:
    """Return free GPU memory in MiB (CUDA API — no subprocess overhead)."""
    free_bytes, _ = torch.cuda.mem_get_info()
    return free_bytes / (1024 ** 2)


def _wait_for_gpu(model: torch.nn.Module, device: str,
                  threshold_mib: int = GPU_FREE_THRESHOLD_MiB) -> None:
    """
    Block until the GPU has at least threshold_mib free.
    While waiting, offloads the model to CPU so COOBA shards can use the
    memory during their allocation spikes — preventing COOBA OOM as well.
    """
    if _gpu_free_mib() >= threshold_mib:
        return  # fast path — no waiting needed

    free = _gpu_free_mib()
    print(f"\n  [GPU guard] {free:.0f} MiB free (< {threshold_mib} MiB threshold). "
          f"Offloading BLAZE model to CPU to free headroom for COOBA...")
    model.cpu()
    torch.cuda.empty_cache()

    while True:
        free = _gpu_free_mib()
        if free >= threshold_mib:
            break
        print(f"  [GPU guard] {free:.0f} MiB free — still waiting "
              f"({GPU_POLL_INTERVAL_S}s interval)...")
        time.sleep(GPU_POLL_INTERVAL_S)

    print(f"  [GPU guard] {free:.0f} MiB free — resuming, moving model back to {device}.")
    model.to(device)


# ---------------------------------------------------------------------------
# Training helpers
# ---------------------------------------------------------------------------

def _build_dataloader(samples, shuffle: bool, drop_last: bool = False) -> DataLoader:
    if not samples:
        return None
    return DataLoader(
        BlazeDataset(samples),
        batch_size=BATCH_SIZE,
        shuffle=shuffle,
        num_workers=0,
        collate_fn=_collate_fn,
        drop_last=drop_last,
    )


def _collate_fn(batch):
    """Collate list-of-dicts into dict-of-lists (texts stay as lists of strings)."""
    return {
        "bug_text": [item["bug_text"] for item in batch],
        "code_text": [item["code_text"] for item in batch],
        "issue_id": torch.tensor([item["issue_id"] for item in batch], dtype=torch.long),
    }


def _train_epoch(
    model: BlazeEmbedding,
    loss_fn: NTXentLoss,
    optimizer: optim.Optimizer,
    scheduler,
    scaler: torch.cuda.amp.GradScaler,
    source_loader,
    target_loader,
    device: str,
    epoch: int,
) -> float:
    """Run one training epoch over source and/or target loaders."""
    model.train()
    # Keep frozen backbone in eval mode so its dropout/BN stays deterministic
    if FREEZE_TRANSFORMER:
        model.transformer.eval()
    total_loss = 0.0
    step_count = 0
    optimizer.zero_grad(set_to_none=True)

    # Build an interleaved iterator: source + target (if both present)
    if source_loader and target_loader:
        loaders = itertools.chain(source_loader, target_loader)
        n_batches = len(source_loader) + len(target_loader)
        label = "Joint"
    elif source_loader:
        loaders = iter(source_loader)
        n_batches = len(source_loader)
        label = "Source-only"
    else:
        loaders = iter(target_loader)
        n_batches = len(target_loader)
        label = "Target-only"

    pbar = tqdm(loaders, total=n_batches, desc=f"Epoch {epoch+1} [{label}]")

    for batch_idx, batch in enumerate(pbar):
        # Periodic GPU memory check — offloads model to CPU if COOBA is spiking
        if batch_idx % GPU_CHECK_EVERY_N_BATCHES == 0:
            _wait_for_gpu(model, device)

        issue_ids = batch["issue_id"].to(device)

        with torch.amp.autocast(device_type="cuda"):
            embeddings = model(batch["bug_text"], batch["code_text"])
            loss = loss_fn(embeddings["report"], embeddings["code"], issue_ids)
            loss = loss / GRAD_ACCUM_STEPS

        scaler.scale(loss).backward()

        if (batch_idx + 1) % GRAD_ACCUM_STEPS == 0 or (batch_idx + 1) == n_batches:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)

        total_loss += loss.item() * GRAD_ACCUM_STEPS
        step_count += 1
        if step_count % 20 == 0:
            pbar.set_postfix({"loss": f"{total_loss / step_count:.4f}"})

    return total_loss / max(step_count, 1)


def _val_loss(
    model: BlazeEmbedding,
    loss_fn: NTXentLoss,
    source_val_loader,
    target_val_loader,
    device: str,
) -> float:
    """Compute validation loss for early stopping."""
    model.eval()
    total_loss, count = 0.0, 0
    loaders = []
    if source_val_loader:
        loaders.append(source_val_loader)
    if target_val_loader:
        loaders.append(target_val_loader)

    with torch.no_grad():
        for loader in loaders:
            for batch in loader:
                issue_ids = batch["issue_id"].to(device)
                with torch.amp.autocast(device_type="cuda"):
                    embeddings = model(batch["bug_text"], batch["code_text"])
                    loss = loss_fn(embeddings["report"], embeddings["code"], issue_ids)
                total_loss += loss.item()
                count += 1

    # No validation batches means all val sets were smaller than BATCH_SIZE;
    # return inf so early stopping does not treat this as a perfect result.
    if count == 0:
        return float("inf")
    return total_loss / count


# ---------------------------------------------------------------------------
# Evaluation helper — encode entire snapshot corpus
# ---------------------------------------------------------------------------

def _encode_snapshot_blobs(
    model: BlazeEmbedding,
    repo: git.Repo,
    path_to_sha: dict,
    project_language: str,
    device: str,
    blob_cache: dict,
) -> dict:
    """
    Encode every blob in a commit snapshot with the fine-tuned BLAZE source adapter.

    Uses dynamic chunking; file-level representation = individual chunk embeddings
    stored as a (num_chunks, dim) array.  Results are cached in `blob_cache` by
    blob_sha to avoid re-encoding across bugs that share the same snapshot.

    Returns:
        blob_sha → np.ndarray of shape (num_chunks, BGE_EMBEDDING_DIM)
    """
    sha_to_chunks: dict = {}
    shas_to_process = [
        sha for sha in path_to_sha.values() if sha not in blob_cache
    ]

    # Gather (sha, chunks) for all un-cached blobs
    sha_chunk_pairs: list = []
    for sha in shas_to_process:
        content = read_blob_content(repo, sha)
        if not content:
            blob_cache[sha] = np.zeros((1, BGE_EMBEDDING_DIM), dtype=np.float32)
            continue
        # detect language from the file path
        file_path = next(
            (p for p, s in path_to_sha.items() if s == sha), ""
        )
        lang = get_file_language(file_path, project_language)
        splitter = DynamicCodeSplitter(language=lang)
        chunks = splitter.split_text(content)
        if not chunks:
            chunks = _fallback_chunk(content, 60)
        sha_chunk_pairs.append((sha, chunks))

    if sha_chunk_pairs:
        model.eval()
        with torch.no_grad():
            # Flatten all chunks into one list for batched encoding
            flat_shas, flat_chunks, chunk_counts = [], [], []
            for sha, chunks in sha_chunk_pairs:
                flat_shas.extend([sha] * len(chunks))
                flat_chunks.extend(chunks)
                chunk_counts.append((sha, len(chunks)))

            all_embeddings = []
            for i in range(0, len(flat_chunks), EVAL_BATCH_SIZE):
                batch_texts = flat_chunks[i : i + EVAL_BATCH_SIZE]
                with torch.amp.autocast(device_type="cuda"):
                    emb = model.get_source_embedding(batch_texts)
                emb = F.normalize(emb, p=2, dim=1).cpu().numpy().astype(np.float32)
                all_embeddings.append(emb)

            all_embeddings = np.concatenate(all_embeddings, axis=0)

            # Split back per blob
            offset = 0
            for sha, count in chunk_counts:
                chunk_embs = all_embeddings[offset : offset + count]
                blob_cache[sha] = chunk_embs
                offset += count

    # Build the return dict for this snapshot
    for sha in path_to_sha.values():
        if sha in blob_cache:
            sha_to_chunks[sha] = blob_cache[sha]

    return sha_to_chunks


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate(
    model: BlazeEmbedding,
    bug_metadata_db: dict,
    blob_embedding_db: dict,  # pre-built BGE embeddings — defines the FAISS candidate pool
    bug_reports_df: pd.DataFrame,
    target_test_ids: list,
    target_repo: git.Repo,
    target_language: str,
    source_project: str,
    target_project: str,
    scenario: str,
    device: str,
) -> dict:
    """
    BLAZE evaluation reranking the shared FAISS top-K candidate pool.

    For each test bug:
      1. Encode the bug report with the fine-tuned report_adapter.
      2. Encode every file in the commit snapshot with the source_adapter.
      3. Score each file as max(cosine_sim over its chunks).
      4. Restrict to the FAISS top-TOP_K_CANDIDATES candidates (from pre-built
         BGE embeddings), then rank the restricted set by BLAZE score →
         compute Top-1/5/10, MAP, MRR over that candidate pool.

    Also writes a diagnostic CSV with FAISS rank vs. BLAZE rank (both within
    the same restricted candidate pool) for every ground-truth file.
    """
    model.eval()
    os.makedirs(os.path.join(RESULT_PATH, "blaze_ph1_diagnostics"), exist_ok=True)

    bug_texts: dict = pd.Series(
        bug_reports_df["bug_report_text"].values,
        index=bug_reports_df["bug_id"],
    ).to_dict()

    top_k_hits = {1: 0, 5: 0, 10: 0}
    average_precisions, reciprocal_ranks = [], []
    diagnostic_results = []

    # Per-experiment blob embedding cache: sha → (num_chunks, dim)
    blob_cache: dict = {}

    test_db = {bid: bug_metadata_db[bid] for bid in target_test_ids if bid in bug_metadata_db}

    for bug_id, meta in tqdm(test_db.items(), desc="Evaluating"):
        bug_text = bug_texts.get(bug_id, "")
        if not bug_text:
            reciprocal_ranks.append(0.0)
            average_precisions.append(0.0)
            continue
        commit_sha = meta.get("commit_sha", "")
        if not commit_sha:
            reciprocal_ranks.append(0.0)
            average_precisions.append(0.0)
            continue

        path_to_sha = get_path_to_sha_map(target_repo, commit_sha)
        if not path_to_sha:
            reciprocal_ranks.append(0.0)
            average_precisions.append(0.0)
            continue

        ground_truth_shas = {
            sha
            for path, sha in path_to_sha.items()
            if any(path.endswith(gt) for gt in meta.get("ground_truth_files", []))
        }
        if not ground_truth_shas:
            reciprocal_ranks.append(0.0)
            average_precisions.append(0.0)
            continue

        # --- BLAZE ranks: encode with fine-tuned model ---
        sha_to_chunk_embs = _encode_snapshot_blobs(
            model, target_repo, path_to_sha, target_language, device, blob_cache
        )

        with torch.no_grad():
            with torch.amp.autocast(device_type="cuda"):
                query_emb = model.get_report_embedding([bug_text])
        query_emb = F.normalize(query_emb, p=2, dim=1).cpu().numpy()[0]  # (dim,)

        # Score each blob: max cosine_sim over its chunks
        blaze_scores: dict = {}
        for sha, chunk_embs in sha_to_chunk_embs.items():
            sims = chunk_embs @ query_emb  # (num_chunks,) — already L2-normalised
            blaze_scores[sha] = float(sims.max())

        blaze_ranked = sorted(blaze_scores.items(), key=lambda x: x[1], reverse=True)
        blaze_shas = [sha for sha, _ in blaze_ranked]

        # --- FAISS ranks: pre-built BGE embeddings define the candidate pool ---
        snap_shas = [sha for sha in path_to_sha.values() if sha in blob_embedding_db]
        if snap_shas:
            snap_embs = np.stack(
                [blob_embedding_db[sha].astype(np.float32) for sha in snap_shas]
            )
            snap_embs_norm = snap_embs / (
                np.linalg.norm(snap_embs, axis=1, keepdims=True) + 1e-10
            )
            bug_emb_pre = meta["embedding"].astype(np.float32)
            bug_emb_pre = bug_emb_pre / (np.linalg.norm(bug_emb_pre) + 1e-10)
            faiss_sims = snap_embs_norm @ bug_emb_pre
            faiss_order = np.argsort(faiss_sims)[::-1]
            faiss_ranked_shas = [snap_shas[i] for i in faiss_order]
        else:
            faiss_ranked_shas = []

        # --- Restrict to the shared FAISS top-K candidate pool ---
        # blaze_ranked is already sorted by score, so filtering to membership
        # in the FAISS top-K preserves BLAZE's relative order within the pool
        # (equivalent to scoring only those candidates in the first place).
        faiss_topk_set = set(faiss_ranked_shas[:TOP_K_CANDIDATES])
        blaze_shas = [sha for sha in blaze_shas if sha in faiss_topk_set]

        # --- Standard metrics (BLAZE ranking, restricted to FAISS top-K) ---
        for k in top_k_hits:
            if not set(blaze_shas[:k]).isdisjoint(ground_truth_shas):
                top_k_hits[k] += 1

        found_rank = next(
            (i + 1 for i, sha in enumerate(blaze_shas) if sha in ground_truth_shas), None
        )
        reciprocal_ranks.append(1.0 / found_rank if found_rank else 0.0)

        hits, prec_at_k = 0, []
        for i, sha in enumerate(blaze_shas):
            if sha in ground_truth_shas:
                hits += 1
                prec_at_k.append(hits / (i + 1))
        average_precisions.append(np.mean(prec_at_k) if prec_at_k else 0.0)

        # --- Diagnostic rows ---
        for gt_sha in ground_truth_shas:
            faiss_rank = next(
                (r + 1 for r, s in enumerate(faiss_ranked_shas) if s == gt_sha), -1
            )
            blaze_rank = next(
                (r + 1 for r, s in enumerate(blaze_shas) if s == gt_sha), -1
            )
            rank_displacement = (faiss_rank - blaze_rank) if (faiss_rank != -1 and blaze_rank != -1) else -1
            diagnostic_results.append({
                "bug_id": bug_id,
                "gt_blob_sha": gt_sha,
                "faiss_rank": faiss_rank,
                "model_rank": blaze_rank,
                "rank_displacement": rank_displacement,
                "faiss_score": float(faiss_sims[snap_shas.index(gt_sha)]) if gt_sha in snap_shas else -1.0,
                "model_score": blaze_scores.get(gt_sha, -1.0),
                "n_candidates_ranked": len(blaze_shas),
            })

    # --- Save diagnostics ---
    if diagnostic_results:
        diag_df = pd.DataFrame(diagnostic_results)
        src_safe = source_project.replace("/", "_")
        tgt_safe = target_project.replace("/", "_")
        diag_path = os.path.join(
            RESULT_PATH, "blaze_ph1_diagnostics",
            f"{src_safe}_{tgt_safe}_{scenario}_diagnostics.csv",
        )
        try:
            diag_df.to_csv(diag_path, index=False)
            print(f"  Saved diagnostics → {diag_path}")
        except Exception as e:
            print(f"  Warning: could not save diagnostics: {e}")

    n = len(test_db)
    if n == 0:
        return {"Top-1": 0, "Top-5": 0, "Top-10": 0, "MAP": 0, "MRR": 0}

    return {
        "Top-1":  round(top_k_hits[1] / n, 4),
        "Top-5":  round(top_k_hits[5] / n, 4),
        "Top-10": round(top_k_hits[10] / n, 4),
        "MAP":    round(float(np.mean(average_precisions)), 4),
        "MRR":    round(float(np.mean(reciprocal_ranks)), 4),
    }


# ---------------------------------------------------------------------------
# Main experiment entry point
# ---------------------------------------------------------------------------

def run_blaze_experiment(
    source_project: str,
    target_project: str,
    source_train_ids: list,
    target_train_ids: list,
    target_test_ids: list,
    scenario: str,
    validation_split_ratio: float = 0.2,
) -> dict:
    """
    Train and evaluate BLAZE for one (source, target, scenario) combination.

    Mirrors the signature of run_tranp_cnn_experiment and run_cooba_experiment
    so it can be swapped in run_expts_ph1.py with no changes.

    Scenarios:
        WP-small      : target_train_ids (20%), source_train_ids = []
        WP-large      : target_train_ids (80%), source_train_ids = []
        CPL-cold-start: source_train_ids (all), target_train_ids = []
        CPL-transfer  : both provided

    Returns:
        dict with keys 'Top-1', 'Top-5', 'Top-10', 'MAP', 'MRR'.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\n{'='*60}")
    print(f"BLAZE | {source_project} → {target_project} | {scenario}")
    print(f"Device: {device}")
    print(f"{'='*60}")

    # 1. Load metadata and databases
    df_meta = pd.read_parquet(PROJECTS_METADATA_PATH)
    df_bugs = pd.read_parquet(BUG_REPORTS_PATH)

    source_bug_db, source_blob_db, source_language = load_project_databases(source_project, df_meta)
    target_bug_db, target_blob_db, target_language = load_project_databases(target_project, df_meta)

    target_repo = git.Repo(get_project_path(target_project, target_language))

    # 2. Train / validation split
    def _split(ids, ratio):
        if not ids or len(ids) * ratio < 1:
            return ids, []
        return train_test_split(ids, test_size=ratio, random_state=42)

    src_train_final, src_val_ids = _split(source_train_ids, validation_split_ratio)
    tgt_train_final, tgt_val_ids = _split(target_train_ids, validation_split_ratio)

    print(f"  Source: {len(src_train_final)} train / {len(src_val_ids)} val bugs")
    print(f"  Target: {len(tgt_train_final)} train / {len(tgt_val_ids)} val bugs")
    print(f"  Target test: {len(target_test_ids)} bugs")

    # 3. Create contrastive training / validation samples
    print("\nBuilding contrastive samples...")
    src_train_samples = (
        create_contrastive_samples(
            source_project, source_language, source_bug_db, df_bugs, src_train_final
        )
        if src_train_final else []
    )
    src_val_samples = (
        create_contrastive_samples(
            source_project, source_language, source_bug_db, df_bugs, src_val_ids
        )
        if src_val_ids else []
    )
    tgt_train_samples = (
        create_contrastive_samples(
            target_project, target_language, target_bug_db, df_bugs, tgt_train_final
        )
        if tgt_train_final else []
    )
    tgt_val_samples = (
        create_contrastive_samples(
            target_project, target_language, target_bug_db, df_bugs, tgt_val_ids
        )
        if tgt_val_ids else []
    )

    print(f"  Source train samples: {len(src_train_samples)}")
    print(f"  Target train samples: {len(tgt_train_samples)}")

    if not src_train_samples and not tgt_train_samples:
        print("WARNING: No training samples found. Skipping training; returning zeros.")
        return {"Top-1": 0, "Top-5": 0, "Top-10": 0, "MAP": 0, "MRR": 0}

    # 4. Build DataLoaders
    # Training: drop_last=True so every batch has BATCH_SIZE items (needed for
    # stable NTXent hard-mining). Validation: drop_last=False to use all samples.
    source_train_loader = _build_dataloader(src_train_samples, shuffle=True,  drop_last=True)
    source_val_loader   = _build_dataloader(src_val_samples,   shuffle=False, drop_last=False)
    target_train_loader = _build_dataloader(tgt_train_samples, shuffle=True,  drop_last=True)
    target_val_loader   = _build_dataloader(tgt_val_samples,   shuffle=False, drop_last=False)

    # 5. Initialize model, optimizer, scheduler, loss
    torch.cuda.empty_cache()
    model = BlazeEmbedding(
        model_name=BGE_MODEL_NAME,
        dimension=BGE_EMBEDDING_DIM,
        max_length=MAX_CODE_LENGTH,
        residual_layers=RESIDUAL_LAYERS,
    ).to(device)

    # Freeze transformer backbone to stay within VRAM budget.
    # Only adapters and pooler parameters accumulate gradients.
    if FREEZE_TRANSFORMER:
        for param in model.transformer.parameters():
            param.requires_grad = False
        model.transformer.eval()  # disable dropout in frozen layers
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total    = sum(p.numel() for p in model.parameters())
        print(f"  Transformer frozen. Trainable params: {trainable:,} / {total:,}")

    loss_fn = NTXentLoss(
        tau=TAU,
        hard_negative_scale=HARD_NEGATIVE_SCALE,
        hard_positive_scale=HARD_POSITIVE_SCALE,
        positive_scale=POSITIVE_SCALE,
    )

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.AdamW(trainable_params, lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

    total_train_batches = (
        (len(source_train_loader) if source_train_loader else 0)
        + (len(target_train_loader) if target_train_loader else 0)
    )
    # Use ceil so that even when total_train_batches < GRAD_ACCUM_STEPS we get
    # at least 1 optimizer step per epoch (the training loop always steps at
    # end-of-epoch via the `(batch_idx+1)==n_batches` guard).
    optimizer_steps_per_epoch = max(1, math.ceil(total_train_batches / GRAD_ACCUM_STEPS))
    total_steps = max(1, optimizer_steps_per_epoch * EPOCHS)
    warmup_steps = max(1, int(total_steps * WARMUP_RATIO))

    scheduler = transformers.get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps
    )
    scaler = torch.amp.GradScaler("cuda")

    # 6. Training loop with early stopping on validation loss
    torch.backends.cudnn.benchmark = True
    best_val_loss = float("inf")
    patience_counter = 0
    best_state = None
    best_epoch = -1

    print(f"\n--- Training (max {EPOCHS} epochs) ---")
    for epoch in range(EPOCHS):
        _wait_for_gpu(model, device)  # block epoch start if GPU is tight
        train_loss = _train_epoch(
            model, loss_fn, optimizer, scheduler, scaler,
            source_train_loader, target_train_loader, device, epoch,
        )
        print(f"  Epoch {epoch+1}: train_loss={train_loss:.4f}")

        if source_val_loader or target_val_loader:
            val_loss = _val_loss(model, loss_fn, source_val_loader, target_val_loader, device)
            print(f"  Epoch {epoch+1}: val_loss={val_loss:.4f}")

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                best_state = copy.deepcopy(model.state_dict())
                best_epoch = epoch + 1
                print(f"  -> New best val_loss={best_val_loss:.4f} (epoch {best_epoch})")
            else:
                patience_counter += 1
                print(f"  -> No improvement ({patience_counter}/{PATIENCE})")
                if patience_counter >= PATIENCE:
                    print(f"  Early stopping at epoch {epoch+1}.")
                    break
        else:
            # No validation data — save the last state as best
            best_state = copy.deepcopy(model.state_dict())
            best_epoch = epoch + 1

    if best_state is not None:
        print(f"\nRestoring model from epoch {best_epoch} (val_loss={best_val_loss:.4f})")
        model.load_state_dict(best_state)

    # 7. Evaluation
    if not target_test_ids:
        print("WARNING: No test IDs. Returning zeros.")
        return {"Top-1": 0, "Top-5": 0, "Top-10": 0, "MAP": 0, "MRR": 0}

    print("\n--- Evaluating on target test set ---")
    _wait_for_gpu(model, device)  # ensure headroom before full-snapshot embedding pass
    metrics = evaluate(
        model=model,
        bug_metadata_db=target_bug_db,
        blob_embedding_db=target_blob_db,
        bug_reports_df=df_bugs,
        target_test_ids=target_test_ids,
        target_repo=target_repo,
        target_language=target_language,
        source_project=source_project,
        target_project=target_project,
        scenario=scenario,
        device=device,
    )

    print(f"\nFinal metrics: {metrics}")
    return metrics
