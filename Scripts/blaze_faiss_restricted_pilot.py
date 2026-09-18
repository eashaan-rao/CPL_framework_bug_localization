"""
Pilot experiment: does restricting BLAZE to the shared FAISS top-300 candidate
pool (like TRANP-CNN and COOBA already are) change its results?

Context: BLAZE's production evaluate() (src/blaze/pipeline.py) ranks the ENTIRE
file snapshot, not the FAISS top-300 shortlist the paper describes all three
models as sharing. This script trains one BLAZE model on one pair/scenario
exactly as the production pipeline does, then evaluates that SAME trained
model two ways:

  1. full-corpus   — mirrors current production evaluate() (ranks all files)
  2. faiss-top300   — ranks only the files inside the FAISS top-300 shortlist
                       (same candidate-generation logic TRANP-CNN/COOBA use:
                       TOP_K_CANDIDATES = 300)

Pair chosen: ansible/ansible -> jupyterlab/jupyterlab, scenario CP-transfer.
jupyterlab's FAISS Recall@300 is 47% (results/faiss_recall_results/
semantic_search_jupyterlab_jupyterlab_results.csv) — the single project this
concern is most visible on, and the pair already discussed in the paper as
Case 2. This is a genuinely fresh training run (not a re-use of any saved
checkpoint), so results will differ slightly from the production CSV row for
this pair by ordinary run-to-run variance (~0.05-0.09 MRR, documented in
main.tex's threats section) — the meaningful comparison is full-corpus vs.
faiss-top300 WITHIN this run, using identical trained weights.

All outputs are written under results/blaze_faiss_restricted_pilot/ and do
NOT touch any production result file or diagnostics directory.

Run inside the obj1 virtualenv:
    python Scripts/blaze_faiss_restricted_pilot.py
"""

import copy
import json
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import git
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import torch.optim as optim
import transformers
from sklearn.model_selection import train_test_split

from src.blaze.model import BlazeEmbedding, NTXentLoss
from src.blaze.data_prep import (
    BUG_REPORTS_PATH,
    PROJECTS_METADATA_PATH,
    create_contrastive_samples,
    get_path_to_sha_map,
    get_project_path,
    load_project_databases,
)
from src.blaze.pipeline import (
    BATCH_SIZE,
    BGE_EMBEDDING_DIM,
    BGE_MODEL_NAME,
    EPOCHS,
    EVAL_BATCH_SIZE,
    FREEZE_TRANSFORMER,
    GRAD_ACCUM_STEPS,
    HARD_NEGATIVE_SCALE,
    HARD_POSITIVE_SCALE,
    LEARNING_RATE,
    PATIENCE,
    POSITIVE_SCALE,
    RESIDUAL_LAYERS,
    TAU,
    WARMUP_RATIO,
    WEIGHT_DECAY,
    _build_dataloader,
    _encode_snapshot_blobs,
    _train_epoch,
    _val_loss,
    _wait_for_gpu,
)

# ── Pilot configuration ──────────────────────────────────────────────────
SOURCE_PROJECT = "ansible/ansible"
TARGET_PROJECT = "jupyterlab/jupyterlab"
SCENARIO = "CP-transfer"
FAISS_TOP_K = 300

OUT_DIR = f"/home/cs21d002_eashaan/PhD/Objective1/results/blaze_faiss_restricted_pilot/{SOURCE_PROJECT.replace('/', '_')}_{TARGET_PROJECT.replace('/', '_')}_{SCENARIO}"
os.makedirs(OUT_DIR, exist_ok=True)

REFERENCE_CSV = "/home/cs21d002_eashaan/PhD/Objective1/results/paper_results_complete_corrected.csv"


def load_bug_ids_for_project(project_name, df_bugs):
    return df_bugs[df_bugs["repo_name"] == project_name]["bug_id"].unique().tolist()


def get_data_splits(target_bug_ids):
    train_pool, test_set = train_test_split(target_bug_ids, test_size=0.20, random_state=42)
    return {"train_pool": train_pool, "test_set": test_set}


def evaluate_dual(model, bug_metadata_db, blob_embedding_db, bug_reports_df,
                   target_test_ids, target_repo, target_language, device,
                   faiss_top_k=300):
    """
    Evaluate one trained model twice: full-corpus ranking and FAISS-top-K-
    restricted ranking. Returns (metrics_full, diag_full, metrics_restricted,
    diag_restricted).
    """
    model.eval()
    bug_texts = pd.Series(
        bug_reports_df["bug_report_text"].values, index=bug_reports_df["bug_id"]
    ).to_dict()

    top_k_hits_full = {1: 0, 5: 0, 10: 0}
    top_k_hits_res = {1: 0, 5: 0, 10: 0}
    ap_full, rr_full = [], []
    ap_res, rr_res = [], []
    diag_full, diag_res = [], []

    test_db = {bid: bug_metadata_db[bid] for bid in target_test_ids if bid in bug_metadata_db}
    blob_cache = {}

    for bug_id, meta in test_db.items():
        bug_text = bug_texts.get(bug_id, "")
        commit_sha = meta.get("commit_sha", "")
        if not bug_text or not commit_sha:
            for lst in (rr_full, ap_full, rr_res, ap_res):
                lst.append(0.0)
            continue

        path_to_sha = get_path_to_sha_map(target_repo, commit_sha)
        if not path_to_sha:
            for lst in (rr_full, ap_full, rr_res, ap_res):
                lst.append(0.0)
            continue

        ground_truth_shas = {
            sha for path, sha in path_to_sha.items()
            if any(path.endswith(gt) for gt in meta.get("ground_truth_files", []))
        }
        if not ground_truth_shas:
            for lst in (rr_full, ap_full, rr_res, ap_res):
                lst.append(0.0)
            continue

        sha_to_chunk_embs = _encode_snapshot_blobs(
            model, target_repo, path_to_sha, target_language, device, blob_cache
        )

        with torch.no_grad():
            with torch.amp.autocast(device_type="cuda"):
                query_emb = model.get_report_embedding([bug_text])
        query_emb = F.normalize(query_emb, p=2, dim=1).cpu().numpy()[0]

        blaze_scores = {}
        for sha, chunk_embs in sha_to_chunk_embs.items():
            sims = chunk_embs @ query_emb
            blaze_scores[sha] = float(sims.max())
        blaze_ranked = sorted(blaze_scores.items(), key=lambda x: x[1], reverse=True)
        blaze_shas_full = [sha for sha, _ in blaze_ranked]

        # FAISS ranking from pre-built BGE embeddings — identical candidate-
        # generation signal TRANP-CNN/COOBA use.
        snap_shas = [sha for sha in path_to_sha.values() if sha in blob_embedding_db]
        if snap_shas:
            snap_embs = np.stack([blob_embedding_db[sha].astype(np.float32) for sha in snap_shas])
            snap_embs_norm = snap_embs / (np.linalg.norm(snap_embs, axis=1, keepdims=True) + 1e-10)
            bug_emb_pre = meta["embedding"].astype(np.float32)
            bug_emb_pre = bug_emb_pre / (np.linalg.norm(bug_emb_pre) + 1e-10)
            faiss_sims = snap_embs_norm @ bug_emb_pre
            faiss_order = np.argsort(faiss_sims)[::-1]
            faiss_ranked_shas = [snap_shas[i] for i in faiss_order]
        else:
            faiss_ranked_shas = []

        faiss_topk_set = set(faiss_ranked_shas[:faiss_top_k])
        blaze_shas_restricted = [sha for sha in blaze_shas_full if sha in faiss_topk_set]

        def score_ranking(blaze_shas, top_k_hits, rr_list, ap_list, diag_list):
            for k in top_k_hits:
                if not set(blaze_shas[:k]).isdisjoint(ground_truth_shas):
                    top_k_hits[k] += 1
            found_rank = next(
                (i + 1 for i, sha in enumerate(blaze_shas) if sha in ground_truth_shas), None
            )
            rr_list.append(1.0 / found_rank if found_rank else 0.0)
            hits, prec_at_k = 0, []
            for i, sha in enumerate(blaze_shas):
                if sha in ground_truth_shas:
                    hits += 1
                    prec_at_k.append(hits / (i + 1))
            ap_list.append(np.mean(prec_at_k) if prec_at_k else 0.0)

            for gt_sha in ground_truth_shas:
                faiss_rank = next(
                    (r + 1 for r, s in enumerate(faiss_ranked_shas) if s == gt_sha), -1
                )
                model_rank = next(
                    (r + 1 for r, s in enumerate(blaze_shas) if s == gt_sha), -1
                )
                diag_list.append({
                    "bug_id": bug_id,
                    "gt_blob_sha": gt_sha,
                    "faiss_rank": faiss_rank,
                    "model_rank": model_rank,
                    "n_candidates_ranked": len(blaze_shas),
                    "faiss_score": float(faiss_sims[snap_shas.index(gt_sha)]) if gt_sha in snap_shas else -1.0,
                    "model_score": blaze_scores.get(gt_sha, -1.0),
                })

        score_ranking(blaze_shas_full, top_k_hits_full, rr_full, ap_full, diag_full)
        score_ranking(blaze_shas_restricted, top_k_hits_res, rr_res, ap_res, diag_res)

    n = len(test_db)

    def summarize(top_k_hits, rr, ap):
        if n == 0:
            return {"Top-1": 0, "Top-5": 0, "Top-10": 0, "MAP": 0, "MRR": 0, "n_test": 0}
        return {
            "Top-1":  round(top_k_hits[1] / n, 4),
            "Top-5":  round(top_k_hits[5] / n, 4),
            "Top-10": round(top_k_hits[10] / n, 4),
            "MAP":    round(float(np.mean(ap)), 4),
            "MRR":    round(float(np.mean(rr)), 4),
            "n_test": n,
        }

    return (
        summarize(top_k_hits_full, rr_full, ap_full), pd.DataFrame(diag_full),
        summarize(top_k_hits_res, rr_res, ap_res), pd.DataFrame(diag_res),
    )


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"BLAZE FAISS-restriction pilot | {SOURCE_PROJECT} -> {TARGET_PROJECT} | {SCENARIO}")
    print(f"Device: {device}\nOutput dir: {OUT_DIR}")

    df_meta = pd.read_parquet(PROJECTS_METADATA_PATH)
    df_bugs = pd.read_parquet(BUG_REPORTS_PATH)

    source_bug_ids_all = load_bug_ids_for_project(SOURCE_PROJECT, df_bugs)
    target_bug_ids_all = load_bug_ids_for_project(TARGET_PROJECT, df_bugs)
    splits = get_data_splits(target_bug_ids_all)

    source_train_ids = source_bug_ids_all
    target_train_ids = train_test_split(splits["train_pool"], train_size=0.25, random_state=42)[0]
    target_test_ids = splits["test_set"]
    print(f"  Source train bugs: {len(source_train_ids)}  Target train bugs: {len(target_train_ids)}  "
          f"Target test bugs: {len(target_test_ids)}")

    source_bug_db, source_blob_db, source_language = load_project_databases(SOURCE_PROJECT, df_meta)
    target_bug_db, target_blob_db, target_language = load_project_databases(TARGET_PROJECT, df_meta)
    target_repo = git.Repo(get_project_path(TARGET_PROJECT, target_language))

    validation_split_ratio = 0.2

    def _split(ids, ratio):
        if not ids or len(ids) * ratio < 1:
            return ids, []
        return train_test_split(ids, test_size=ratio, random_state=42)

    src_train_final, src_val_ids = _split(source_train_ids, validation_split_ratio)
    tgt_train_final, tgt_val_ids = _split(target_train_ids, validation_split_ratio)
    print(f"  Source: {len(src_train_final)} train / {len(src_val_ids)} val bugs")
    print(f"  Target: {len(tgt_train_final)} train / {len(tgt_val_ids)} val bugs")

    print("\nBuilding contrastive samples...")
    src_train_samples = create_contrastive_samples(
        SOURCE_PROJECT, source_language, source_bug_db, df_bugs, src_train_final
    ) if src_train_final else []
    src_val_samples = create_contrastive_samples(
        SOURCE_PROJECT, source_language, source_bug_db, df_bugs, src_val_ids
    ) if src_val_ids else []
    tgt_train_samples = create_contrastive_samples(
        TARGET_PROJECT, target_language, target_bug_db, df_bugs, tgt_train_final
    ) if tgt_train_final else []
    tgt_val_samples = create_contrastive_samples(
        TARGET_PROJECT, target_language, target_bug_db, df_bugs, tgt_val_ids
    ) if tgt_val_ids else []
    print(f"  Source train samples: {len(src_train_samples)}  Target train samples: {len(tgt_train_samples)}")

    source_train_loader = _build_dataloader(src_train_samples, shuffle=True, drop_last=True)
    source_val_loader = _build_dataloader(src_val_samples, shuffle=False, drop_last=False)
    target_train_loader = _build_dataloader(tgt_train_samples, shuffle=True, drop_last=True)
    target_val_loader = _build_dataloader(tgt_val_samples, shuffle=False, drop_last=False)

    torch.cuda.empty_cache()
    model = BlazeEmbedding(
        model_name=BGE_MODEL_NAME, dimension=BGE_EMBEDDING_DIM,
        max_length=512, residual_layers=RESIDUAL_LAYERS,
    ).to(device)

    if FREEZE_TRANSFORMER:
        for param in model.transformer.parameters():
            param.requires_grad = False
        model.transformer.eval()
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in model.parameters())
        print(f"  Transformer frozen. Trainable params: {trainable:,} / {total:,}")

    loss_fn = NTXentLoss(
        tau=TAU, hard_negative_scale=HARD_NEGATIVE_SCALE,
        hard_positive_scale=HARD_POSITIVE_SCALE, positive_scale=POSITIVE_SCALE,
    )
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.AdamW(trainable_params, lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

    total_train_batches = (
        (len(source_train_loader) if source_train_loader else 0)
        + (len(target_train_loader) if target_train_loader else 0)
    )
    optimizer_steps_per_epoch = max(1, math.ceil(total_train_batches / GRAD_ACCUM_STEPS))
    total_steps = max(1, optimizer_steps_per_epoch * EPOCHS)
    warmup_steps = max(1, int(total_steps * WARMUP_RATIO))
    scheduler = transformers.get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps
    )
    scaler = torch.amp.GradScaler("cuda")

    torch.backends.cudnn.benchmark = True
    best_val_loss = float("inf")
    patience_counter = 0
    best_state = None
    best_epoch = -1

    print(f"\n--- Training (max {EPOCHS} epochs) ---")
    for epoch in range(EPOCHS):
        _wait_for_gpu(model, device)
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
            best_state = copy.deepcopy(model.state_dict())
            best_epoch = epoch + 1

    if best_state is not None:
        print(f"\nRestoring model from epoch {best_epoch} (val_loss={best_val_loss:.4f})")
        model.load_state_dict(best_state)

    print("\n--- Evaluating (full-corpus AND FAISS-top-300, same trained weights) ---")
    _wait_for_gpu(model, device)
    metrics_full, diag_full, metrics_res, diag_res = evaluate_dual(
        model=model, bug_metadata_db=target_bug_db, blob_embedding_db=target_blob_db,
        bug_reports_df=df_bugs, target_test_ids=target_test_ids, target_repo=target_repo,
        target_language=target_language, device=device, faiss_top_k=FAISS_TOP_K,
    )

    diag_full.to_csv(os.path.join(OUT_DIR, "full_corpus_diagnostics.csv"), index=False)
    diag_res.to_csv(os.path.join(OUT_DIR, "faiss_top300_diagnostics.csv"), index=False)

    # Reference: existing production number for this exact pair/scenario/model
    reference = None
    if os.path.exists(REFERENCE_CSV):
        ref_df = pd.read_csv(REFERENCE_CSV)
        row = ref_df[
            (ref_df.model_name == "BLAZE") & (ref_df.source_project == SOURCE_PROJECT)
            & (ref_df.target_project == TARGET_PROJECT) & (ref_df.scenario == SCENARIO)
        ]
        if len(row):
            reference = row.iloc[0][["top-1", "top-5", "top-10", "MAP", "MRR"]].to_dict()

    summary = {
        "pair": f"{SOURCE_PROJECT} -> {TARGET_PROJECT}",
        "scenario": SCENARIO,
        "faiss_top_k": FAISS_TOP_K,
        "best_epoch": best_epoch,
        "production_reference_full_corpus": reference,
        "this_run_full_corpus": metrics_full,
        "this_run_faiss_top300_restricted": metrics_res,
    }
    with open(os.path.join(OUT_DIR, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    print("\n" + "=" * 70)
    print("RESULT")
    print("=" * 70)
    print(f"Production reference (original full run, this exact pair/scenario): {reference}")
    print(f"This run — full corpus (fresh training, sanity check):              {metrics_full}")
    print(f"This run — FAISS top-{FAISS_TOP_K} restricted (same weights):          {metrics_res}")
    print(f"\nSaved: {OUT_DIR}/")


if __name__ == "__main__":
    main()
