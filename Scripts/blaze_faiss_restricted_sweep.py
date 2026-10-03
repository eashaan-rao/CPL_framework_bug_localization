"""
Sweep version of blaze_faiss_restricted_pilot.py: runs the same train-once /
evaluate-twice (full-corpus vs. FAISS-top-300, same weights) experiment across
all 63 paper pairs, CP-transfer scenario only.

NOTE ON SCOPE: this only covers CP-transfer. The full-corpus-ranking issue in
src/blaze/pipeline.py's evaluate() affects WP-small, WP-large, and
CP-cold-start identically (evaluate() doesn't branch on scenario) — CP-transfer
is just the scenario this sweep starts with because it drives most of the
paper's headline RQ1 claims. Table 7 will have BLAZE on two different
evaluation methodologies (three scenarios full-corpus, one FAISS-300-restricted)
until/unless the other three scenarios get the same treatment.

Safe to leave running unattended:
  - Writes each pair's result to results/blaze_faiss_restricted_sweep/
    cp_transfer_summary.csv the moment it finishes (not buffered to the end).
  - Skips any (source, target) pair already present in that CSV, so it is
    resumable after an interruption — just rerun the script.
  - Per-pair diagnostics saved to results/blaze_faiss_restricted_sweep/
    diagnostics/<src>_<tgt>_full.csv and ..._faiss_top300.csv.

Run inside the obj1 virtualenv (expect ~45-50 min/pair based on the pilot run
on ansible->jupyterlab; 63 pairs is ~two days of continuous GPU time):
    python Scripts/blaze_faiss_restricted_sweep.py
"""

import copy
import json
import math
import os
import sys
import time

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
    BGE_EMBEDDING_DIM,
    BGE_MODEL_NAME,
    EPOCHS,
    FREEZE_TRANSFORMER,
    GPU_CHECK_EVERY_N_EVAL_ITEMS,
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

# Reuse the exact 63-pair paper set from the production runner, without
# executing its main().
from Scripts.run_expts_ph1 import PAPER_PAIRS  # noqa: E402

SCENARIO = "CP-transfer"
FAISS_TOP_K = 300

OUT_DIR = "/home/cs21d002_eashaan/PhD/Objective1/results/blaze_faiss_restricted_sweep"
DIAG_DIR = os.path.join(OUT_DIR, "diagnostics")
SUMMARY_CSV = os.path.join(OUT_DIR, "cp_transfer_summary.csv")
REFERENCE_CSV = "/home/cs21d002_eashaan/PhD/Objective1/results/paper_results_complete_corrected.csv"
os.makedirs(DIAG_DIR, exist_ok=True)

SUMMARY_COLUMNS = [
    "source_project", "target_project", "scenario", "best_epoch", "wall_time_sec",
    "ref_top1", "ref_top5", "ref_top10", "ref_MAP", "ref_MRR",
    "full_top1", "full_top5", "full_top10", "full_MAP", "full_MRR",
    "res_top1", "res_top5", "res_top10", "res_MAP", "res_MRR",
    "n_test", "n_gt_files", "n_gt_files_in_top300",
]


def already_done(source_project, target_project):
    if not os.path.exists(SUMMARY_CSV):
        return False
    df = pd.read_csv(SUMMARY_CSV)
    return ((df.source_project == source_project) & (df.target_project == target_project)).any()


def append_summary_row(row: dict):
    df = pd.DataFrame([row])
    write_header = not os.path.exists(SUMMARY_CSV)
    df.to_csv(SUMMARY_CSV, mode="a", header=write_header, index=False, columns=SUMMARY_COLUMNS)


def load_bug_ids_for_project(project_name, df_bugs):
    return df_bugs[df_bugs["repo_name"] == project_name]["bug_id"].unique().tolist()


def get_data_splits(target_bug_ids):
    train_pool, test_set = train_test_split(target_bug_ids, test_size=0.20, random_state=42)
    return {"train_pool": train_pool, "test_set": test_set}


def evaluate_dual(model, bug_metadata_db, blob_embedding_db, bug_reports_df,
                   target_test_ids, target_repo, target_language, device,
                   faiss_top_k=300):
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

    for bug_idx, (bug_id, meta) in enumerate(test_db.items()):
        if bug_idx % GPU_CHECK_EVERY_N_EVAL_ITEMS == 0:
            _wait_for_gpu(model, device)  # full-snapshot encoding is the spikiest step
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
                    "bug_id": bug_id, "gt_blob_sha": gt_sha, "faiss_rank": faiss_rank,
                    "model_rank": model_rank, "n_candidates_ranked": len(blaze_shas),
                    "faiss_score": float(faiss_sims[snap_shas.index(gt_sha)]) if gt_sha in snap_shas else -1.0,
                    "model_score": blaze_scores.get(gt_sha, -1.0),
                })

        score_ranking(blaze_shas_full, top_k_hits_full, rr_full, ap_full, diag_full)
        score_ranking(blaze_shas_restricted, top_k_hits_res, rr_res, ap_res, diag_res)

    n = len(test_db)

    def summarize(top_k_hits, rr, ap):
        if n == 0:
            return {"Top-1": 0, "Top-5": 0, "Top-10": 0, "MAP": 0, "MRR": 0}
        return {
            "Top-1": round(top_k_hits[1] / n, 4), "Top-5": round(top_k_hits[5] / n, 4),
            "Top-10": round(top_k_hits[10] / n, 4), "MAP": round(float(np.mean(ap)), 4),
            "MRR": round(float(np.mean(rr)), 4),
        }

    return (
        summarize(top_k_hits_full, rr_full, ap_full), pd.DataFrame(diag_full),
        summarize(top_k_hits_res, rr_res, ap_res), pd.DataFrame(diag_res), n,
    )


def run_pair(source_project, target_project, df_meta, df_bugs, ref_df, device):
    t0 = time.time()
    print(f"\n{'='*70}\nBLAZE FAISS pilot | {source_project} -> {target_project} | {SCENARIO}\n{'='*70}")

    source_bug_ids_all = load_bug_ids_for_project(source_project, df_bugs)
    target_bug_ids_all = load_bug_ids_for_project(target_project, df_bugs)
    splits = get_data_splits(target_bug_ids_all)

    source_train_ids = source_bug_ids_all
    target_train_ids = train_test_split(splits["train_pool"], train_size=0.25, random_state=42)[0]
    target_test_ids = splits["test_set"]
    print(f"  Source train: {len(source_train_ids)}  Target train: {len(target_train_ids)}  "
          f"Target test: {len(target_test_ids)}")

    source_bug_db, source_blob_db, source_language = load_project_databases(source_project, df_meta)
    target_bug_db, target_blob_db, target_language = load_project_databases(target_project, df_meta)
    target_repo = git.Repo(get_project_path(target_project, target_language))

    def _split(ids, ratio):
        if not ids or len(ids) * ratio < 1:
            return ids, []
        return train_test_split(ids, test_size=ratio, random_state=42)

    src_train_final, src_val_ids = _split(source_train_ids, 0.2)
    tgt_train_final, tgt_val_ids = _split(target_train_ids, 0.2)

    src_train_samples = create_contrastive_samples(
        source_project, source_language, source_bug_db, df_bugs, src_train_final
    ) if src_train_final else []
    src_val_samples = create_contrastive_samples(
        source_project, source_language, source_bug_db, df_bugs, src_val_ids
    ) if src_val_ids else []
    tgt_train_samples = create_contrastive_samples(
        target_project, target_language, target_bug_db, df_bugs, tgt_train_final
    ) if tgt_train_final else []
    tgt_val_samples = create_contrastive_samples(
        target_project, target_language, target_bug_db, df_bugs, tgt_val_ids
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

    if not src_train_samples and not tgt_train_samples:
        print("  WARNING: no training samples, skipping pair.")
        return None

    for epoch in range(EPOCHS):
        _wait_for_gpu(model, device)
        train_loss = _train_epoch(
            model, loss_fn, optimizer, scheduler, scaler,
            source_train_loader, target_train_loader, device, epoch,
        )
        if source_val_loader or target_val_loader:
            val_loss = _val_loss(model, loss_fn, source_val_loader, target_val_loader, device)
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                best_state = copy.deepcopy(model.state_dict())
                best_epoch = epoch + 1
            else:
                patience_counter += 1
                if patience_counter >= PATIENCE:
                    print(f"  Early stopping at epoch {epoch+1} (train_loss={train_loss:.4f}).")
                    break
        else:
            best_state = copy.deepcopy(model.state_dict())
            best_epoch = epoch + 1
        print(f"  Epoch {epoch+1}: train_loss={train_loss:.4f}"
              + (f" val_loss={val_loss:.4f}" if (source_val_loader or target_val_loader) else ""))

    if best_state is not None:
        model.load_state_dict(best_state)

    if not target_test_ids:
        print("  WARNING: no test IDs, skipping pair.")
        return None

    _wait_for_gpu(model, device)
    metrics_full, diag_full, metrics_res, diag_res, n_test = evaluate_dual(
        model=model, bug_metadata_db=target_bug_db, blob_embedding_db=target_blob_db,
        bug_reports_df=df_bugs, target_test_ids=target_test_ids, target_repo=target_repo,
        target_language=target_language, device=device, faiss_top_k=FAISS_TOP_K,
    )

    src_safe = source_project.replace("/", "_")
    tgt_safe = target_project.replace("/", "_")
    diag_full.to_csv(os.path.join(DIAG_DIR, f"{src_safe}_{tgt_safe}_full.csv"), index=False)
    diag_res.to_csv(os.path.join(DIAG_DIR, f"{src_safe}_{tgt_safe}_faiss_top300.csv"), index=False)

    ref_row = ref_df[
        (ref_df.model_name == "BLAZE") & (ref_df.source_project == source_project)
        & (ref_df.target_project == target_project) & (ref_df.scenario == SCENARIO)
    ]
    ref = ref_row.iloc[0][["top-1", "top-5", "top-10", "MAP", "MRR"]].to_dict() if len(ref_row) else {}

    n_gt = len(diag_res)
    n_gt_in_pool = int((diag_res.model_rank != -1).sum()) if n_gt else 0

    wall_time = time.time() - t0
    row = {
        "source_project": source_project, "target_project": target_project, "scenario": SCENARIO,
        "best_epoch": best_epoch, "wall_time_sec": round(wall_time, 1),
        "ref_top1": ref.get("top-1"), "ref_top5": ref.get("top-5"), "ref_top10": ref.get("top-10"),
        "ref_MAP": ref.get("MAP"), "ref_MRR": ref.get("MRR"),
        "full_top1": metrics_full["Top-1"], "full_top5": metrics_full["Top-5"], "full_top10": metrics_full["Top-10"],
        "full_MAP": metrics_full["MAP"], "full_MRR": metrics_full["MRR"],
        "res_top1": metrics_res["Top-1"], "res_top5": metrics_res["Top-5"], "res_top10": metrics_res["Top-10"],
        "res_MAP": metrics_res["MAP"], "res_MRR": metrics_res["MRR"],
        "n_test": n_test, "n_gt_files": n_gt, "n_gt_files_in_top300": n_gt_in_pool,
    }
    print(f"  Done in {wall_time/60:.1f} min. full={metrics_full}  restricted={metrics_res}  "
          f"gt_files_in_top300={n_gt_in_pool}/{n_gt}")
    return row


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    print(f"Pairs to sweep: {len(PAPER_PAIRS)}  Scenario: {SCENARIO}  FAISS top-K: {FAISS_TOP_K}")
    print(f"Summary CSV: {SUMMARY_CSV}\n")

    df_meta = pd.read_parquet(PROJECTS_METADATA_PATH)
    df_bugs = pd.read_parquet(BUG_REPORTS_PATH)
    ref_df = pd.read_csv(REFERENCE_CSV)

    sweep_start = time.time()
    done_count, skip_count = 0, 0
    for i, (source_project, target_project) in enumerate(sorted(PAPER_PAIRS), 1):
        print(f"\n[{i}/{len(PAPER_PAIRS)}] {source_project} -> {target_project}")
        if already_done(source_project, target_project):
            print("  Already done, skipping.")
            skip_count += 1
            continue

        try:
            row = run_pair(source_project, target_project, df_meta, df_bugs, ref_df, device)
        except Exception as e:
            print(f"  ERROR on {source_project} -> {target_project}: {e}")
            continue

        if row is not None:
            append_summary_row(row)
            done_count += 1

        elapsed = time.time() - sweep_start
        remaining = len(PAPER_PAIRS) - skip_count - done_count
        rate = elapsed / done_count if done_count else None
        eta = f"{rate * remaining / 3600:.1f} hrs" if rate else "n/a"
        print(f"  Progress: {done_count} done this run, {skip_count} already done, "
              f"{remaining} remaining. Elapsed {elapsed/3600:.2f} hrs. ETA for rest: {eta}")

    print(f"\nSweep finished. {done_count} run, {skip_count} skipped (already done).")


if __name__ == "__main__":
    main()
