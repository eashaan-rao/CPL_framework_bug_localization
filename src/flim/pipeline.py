"""
FLIM Pipeline — faithful academic replication.

Two-layer architecture
──────────────────────
LAYER 1  (shared retrieval, identical for all models)
    BGE embeddings + FAISS → top-300 candidate files per bug report

LAYER 2  (FLIM reranker, faithful to paper)
    a. Extract Python/Java functions from each candidate file via AST
    b. Fine-tune CodeBERT (microsoft/codebert-base) as a bi-encoder on
       (bug_report, function_code, label) pairs from the training bugs
       — random chunk sampling ensures no information is discarded
    c. After fine-tuning, encode every (bug_summary, bug_description,
       function) with the fine-tuned CodeBERT using full chunking + mean-pool
    d. Compute the 4 FLIM semantic features per (bug_report, file) pair:
         f1  max  cosine_sim(summary_emb,     func_emb_i)  over all functions
         f2  mean cosine_sim(summary_emb,     func_emb_i)  over all functions
         f3  max  cosine_sim(description_emb, func_emb_i)  over all functions
         f4  mean cosine_sim(description_emb, func_emb_i)  over all functions
    e. Train FLIMRanker (Adaptive LTR) on the 4 features
    f. Predict and evaluate on test bugs

No-truncation policy
────────────────────
All texts (bug reports AND function bodies) are chunked (CHUNK_SIZE=450,
CHUNK_OVERLAP=50) and embedded chunk-by-chunk; chunk embeddings are
mean-pooled.  During fine-tuning, random chunk sampling is used as
data augmentation so the model is exposed to every part of long texts.
"""

import ast
import os
import pickle
import random
import re

import faiss
import git
import numpy as np
import pandas as pd
from tqdm import tqdm

from .encoder import CodeBERTEncoder
from .model   import FLIMRanker

# ─── Paths ────────────────────────────────────────────────────────────────────
REPO_BASE_PATH         = "/home/cs21d002_eashaan/PhD/Objective1/data/repos"
BUG_METADATA_DIR       = "/home/cs21d002_eashaan/PhD/Objective1/data/processed/embedding_dbs"
BLOB_DB_DIR            = "/home/cs21d002_eashaan/PhD/Objective1/data/processed/embedding_dbs"
PROJECTS_METADATA_PATH = "/home/cs21d002_eashaan/PhD/Objective1/data/processed/project_metadata.parquet"
BUG_REPORTS_PATH       = "/home/cs21d002_eashaan/PhD/Objective1/data/processed/bug_reports_clean.parquet"

# ─── Experiment settings ──────────────────────────────────────────────────────
TOP_K_CANDIDATES    = 300

# Fine-tuning pair sampling
MAX_FUNCS_POS       = 5    # max functions taken from each GT (positive) file
MAX_NEG_FILES       = 15   # max non-GT files sampled per bug for negatives
MAX_FUNCS_NEG       = 3    # max functions taken from each negative file
MAX_FUNCS_PER_FILE  = 20   # cap when computing features (keeps cost bounded)

# CodeBERT fine-tuning hyper-parameters
FINETUNE_EPOCHS          = 3
FINETUNE_BATCH_SIZE      = 32
FINETUNE_LR              = 2e-5
FINETUNE_MAX_STEPS_EPOCH = 5000   # hard cap per epoch (for very large datasets)

FLIM_FEATURES = ['f1', 'f2', 'f3', 'f4']


# ─────────────────────────────────────────────────────────────────────────────
#  Data loading helpers
# ─────────────────────────────────────────────────────────────────────────────

def _load_project_databases(project_name: str, meta_df: pd.DataFrame):
    language  = meta_df[meta_df['repo_name'] == project_name]['language'].iloc[0]
    safe      = project_name.replace('/', '_')
    with open(os.path.join(BLOB_DB_DIR,      safe + '_blob_embeddings32.pkl'), 'rb') as f:
        blob_db = pickle.load(f)
    with open(os.path.join(BUG_METADATA_DIR, safe + '_bug_metadata32.pkl'),    'rb') as f:
        bug_db  = pickle.load(f)
    return bug_db, blob_db, language


def _path_to_sha_map(repo: git.Repo, commit_sha: str) -> dict:
    try:
        return {b.path: b.hexsha
                for b in repo.commit(commit_sha).tree.traverse()
                if b.type == 'blob'}
    except Exception:
        return {}


# Module-level cache: blob_sha → file content string.
# Safe because git SHA1 is content-addressed (same SHA = same bytes everywhere).
_blob_content_cache: dict[str, str] = {}
_function_cache:     dict[str, list[str]] = {}


def _blob_content(repo: git.Repo, blob_sha: str) -> str:
    if blob_sha in _blob_content_cache:
        return _blob_content_cache[blob_sha]
    try:
        from git import Blob
        from git.util import hex_to_bin
        content = Blob(repo, hex_to_bin(blob_sha)).data_stream.read().decode('utf-8', 'ignore')
    except Exception:
        content = ''
    _blob_content_cache[blob_sha] = content
    return content


# ─────────────────────────────────────────────────────────────────────────────
#  Function extraction
# ─────────────────────────────────────────────────────────────────────────────

def _extract_python_functions(code: str) -> list[str]:
    """
    Extract top-level and class-method function bodies from Python source.
    Uses Python's built-in `ast` module for accuracy.
    Falls back to the whole file if parsing fails.
    """
    funcs = []
    try:
        tree = ast.parse(code)
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                seg = ast.get_source_segment(code, node)
                if seg and len(seg.strip()) > 10:
                    funcs.append(seg.strip())
    except SyntaxError:
        pass
    return funcs or [code]


def _extract_java_functions(code: str) -> list[str]:
    """
    Heuristic Java method extractor: finds method signatures then follows
    matched braces to extract each method body.
    Falls back to the whole file if nothing is found.
    """
    # Simplified method signature pattern (covers most practical cases)
    sig_re = re.compile(
        r'(?:(?:public|protected|private|static|final|abstract|'
        r'synchronized|native|default|strictfp)\s+)*'
        r'(?:[\w<>\[\],\s]+\s+)?'          # return type (may contain generics)
        r'\b\w+\s*\([^)]*\)'               # method name + params
        r'(?:\s*throws\s+[\w,\s]+)?'       # optional throws clause
        r'\s*\{',                           # opening brace
        re.MULTILINE,
    )
    funcs = []
    for match in sig_re.finditer(code):
        depth = 0
        i     = match.end() - 1  # position of '{'
        while i < len(code):
            if   code[i] == '{': depth += 1
            elif code[i] == '}':
                depth -= 1
                if depth == 0:
                    body = code[match.start(): i + 1].strip()
                    if len(body) > 20:
                        funcs.append(body)
                    break
            i += 1
    return funcs or [code]


def _get_functions(repo: git.Repo, blob_sha: str, language: str) -> list[str]:
    """
    Return function/method strings for a blob.
    Results are cached by blob_sha (content-addressed, immutable).
    """
    cache_key = f"{blob_sha}_{language}"
    if cache_key in _function_cache:
        return _function_cache[cache_key]

    code  = _blob_content(repo, blob_sha)
    lang  = language.lower()

    if not code:
        funcs = ['']
    elif lang == 'python':
        funcs = _extract_python_functions(code)
    elif lang == 'java':
        funcs = _extract_java_functions(code)
    else:
        funcs = [code]

    _function_cache[cache_key] = funcs
    return funcs


def _split_bug_report(text: str) -> tuple[str, str]:
    """
    Split bug_report_text into (summary, description).

    Summary     = first non-empty line (the title / one-liner).
    Description = everything after the first non-empty line.
    If the text has only one line, description equals the full text.
    """
    lines           = text.split('\n')
    summary_lines   = []
    desc_lines      = []
    found_summary   = False

    for line in lines:
        if not found_summary:
            if line.strip():
                summary_lines.append(line.strip())
                found_summary = True
        else:
            desc_lines.append(line)

    summary     = ' '.join(summary_lines) if summary_lines else text[:300]
    description = '\n'.join(desc_lines).strip() if desc_lines else text
    return summary, description


# ─────────────────────────────────────────────────────────────────────────────
#  LAYER 1 — FAISS candidate retrieval (uses BGE embeddings)
# ─────────────────────────────────────────────────────────────────────────────

def _faiss_candidates(
    bug_ids:        list,
    bug_metadata_db: dict,
    blob_embedding_db: dict,
    repo:           git.Repo,
    is_training:    bool = True,
    top_k:          int  = TOP_K_CANDIDATES,
) -> dict:
    """
    Returns {bug_id: [(blob_sha, file_path, is_ground_truth), …]}.
    Ground-truth files are force-included for training splits.
    """
    valid = [b for b in bug_ids if b in bug_metadata_db]
    bugs_by_commit: dict[str, list] = {}
    for bug_id in valid:
        c = bug_metadata_db[bug_id]['commit_sha']
        bugs_by_commit.setdefault(c, []).append(bug_id)

    result: dict = {}

    for commit_sha, commit_bugs in tqdm(
        bugs_by_commit.items(), desc="FAISS retrieval", leave=False
    ):
        p2s = _path_to_sha_map(repo, commit_sha)
        if not p2s:
            continue
        s2p = {s: p for p, s in p2s.items()}

        snap = [s for s in p2s.values() if s in blob_embedding_db]
        if not snap:
            continue

        embs = np.array([blob_embedding_db[s] for s in snap], dtype='float32')
        faiss.normalize_L2(embs)
        index = faiss.IndexFlatIP(embs.shape[1])
        index.add(embs)

        for bug_id in commit_bugs:
            info = bug_metadata_db[bug_id]
            q    = info['embedding'].astype('float32').reshape(1, -1).copy()
            faiss.normalize_L2(q)

            k       = min(top_k, len(snap))
            _, idxs = index.search(q, k)

            gt_paths = set(info['ground_truth_files'])
            gt_shas  = {s for p, s in p2s.items()
                        if any(p.endswith(g) for g in gt_paths)}

            cands: list[tuple] = []
            seen:  set         = set()
            for idx in idxs[0]:
                if idx < 0:
                    continue
                sha = snap[idx]
                seen.add(sha)
                cands.append((sha, s2p.get(sha, ''), sha in gt_shas))

            if is_training:
                for sha in gt_shas:
                    if sha not in seen and sha in blob_embedding_db:
                        cands.append((sha, s2p.get(sha, ''), True))

            result[bug_id] = cands

    return result


# ─────────────────────────────────────────────────────────────────────────────
#  LAYER 2a — Fine-tuning pair construction
# ─────────────────────────────────────────────────────────────────────────────

def _build_finetune_pairs(
    candidates:     dict,
    bug_texts_map:  dict,
    repo:           git.Repo,
    language:       str,
) -> tuple[list, list, list]:
    """
    Build (bug_text, function_text, label) triples.

    Positive (label=1): bug_text × functions from ground-truth files
                        (up to MAX_FUNCS_POS per file)
    Negative (label=0): bug_text × functions from sampled non-GT files
                        (MAX_NEG_FILES files × MAX_FUNCS_NEG functions each)
    """
    bugs_out: list[str] = []
    func_out: list[str] = []
    lbl_out:  list[int] = []

    for bug_id, cand_list in tqdm(candidates.items(),
                                   desc="Building fine-tune pairs", leave=False):
        bug_text = bug_texts_map.get(str(bug_id), '')
        if not bug_text:
            continue

        gt_cands  = [(sha, path) for sha, path, is_gt in cand_list if is_gt]
        neg_cands = [(sha, path) for sha, path, is_gt in cand_list if not is_gt]

        # Positives
        for sha, _ in gt_cands:
            funcs = _get_functions(repo, sha, language)
            for func in funcs[:MAX_FUNCS_POS]:
                if func.strip():
                    bugs_out.append(bug_text)
                    func_out.append(func)
                    lbl_out.append(1)

        # Negatives (randomly sampled to limit data size and keep class balance)
        sampled = random.sample(neg_cands, min(MAX_NEG_FILES, len(neg_cands)))
        for sha, _ in sampled:
            funcs  = _get_functions(repo, sha, language)
            chosen = random.sample(funcs, min(MAX_FUNCS_NEG, len(funcs)))
            for func in chosen:
                if func.strip():
                    bugs_out.append(bug_text)
                    func_out.append(func)
                    lbl_out.append(0)

    return bugs_out, func_out, lbl_out


# ─────────────────────────────────────────────────────────────────────────────
#  LAYER 2b — FLIM feature computation (uses fine-tuned CodeBERT)
# ─────────────────────────────────────────────────────────────────────────────

def _cosine_sims(query: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    """
    Compute cosine similarity between a query vector and each row of matrix.
    Both are L2-normalised before the dot product to handle any scale.
    """
    q_norm = query / (np.linalg.norm(query) + 1e-8)
    m_norm = matrix / (np.linalg.norm(matrix, axis=1, keepdims=True) + 1e-8)
    return (m_norm @ q_norm).astype('float32')


def _compute_flim_features(
    candidates:      dict,
    bug_texts_map:   dict,
    repo:            git.Repo,
    language:        str,
    encoder:         CodeBERTEncoder,
) -> pd.DataFrame:
    """
    Compute the 4 FLIM semantic features per (bug_id, blob_sha) pair.

    Features (paper-faithful):
      f1  max  cosine_sim(summary_emb,     func_emb_i)  over functions in file
      f2  mean cosine_sim(summary_emb,     func_emb_i)  over functions in file
      f3  max  cosine_sim(description_emb, func_emb_i)  over functions in file
      f4  mean cosine_sim(description_emb, func_emb_i)  over functions in file

    All embeddings: fine-tuned CodeBERT, full chunking (no truncation),
    chunk embeddings mean-pooled.
    """
    records = []

    for bug_id, cand_list in tqdm(candidates.items(),
                                   desc="Computing FLIM features", leave=False):
        bug_text = bug_texts_map.get(str(bug_id), '')
        if not bug_text:
            continue

        summary, description = _split_bug_report(bug_text)

        # Encode bug parts once per bug (full text, no truncation)
        sum_emb  = encoder.encode(summary)      # (768,)
        desc_emb = encoder.encode(description)  # (768,)

        for sha, path, is_gt in cand_list:
            label = 1.0 if is_gt else 0.0
            funcs = _get_functions(repo, sha, language)[:MAX_FUNCS_PER_FILE]

            if not funcs or not any(f.strip() for f in funcs):
                records.append({
                    'bug_id': str(bug_id), 'blob_sha': sha,
                    'used_in_fix': label,
                    'f1': 0.0, 'f2': 0.0, 'f3': 0.0, 'f4': 0.0,
                })
                continue

            # Encode all functions of this file in a single batched call
            func_embs = encoder.encode_batch(funcs)  # (n_funcs, 768)

            sum_sims  = _cosine_sims(sum_emb,  func_embs)   # (n_funcs,)
            desc_sims = _cosine_sims(desc_emb, func_embs)   # (n_funcs,)

            records.append({
                'bug_id':      str(bug_id),
                'blob_sha':    sha,
                'used_in_fix': label,
                'f1': float(sum_sims.max()),
                'f2': float(sum_sims.mean()),
                'f3': float(desc_sims.max()),
                'f4': float(desc_sims.mean()),
            })

    if not records:
        return pd.DataFrame()

    return pd.DataFrame(records).set_index(['bug_id', 'blob_sha'])


# ─────────────────────────────────────────────────────────────────────────────
#  Evaluation metrics
# ─────────────────────────────────────────────────────────────────────────────

def _compute_metrics(df: pd.DataFrame) -> dict:
    top1 = top5 = top10 = 0
    ap_list: list[float] = []
    rr_list: list[float] = []
    n_bugs = 0

    for _, group in df.groupby(level=0, sort=False):
        ranked = group.sort_values('result', ascending=False)
        labels = ranked['used_in_fix'].values
        n_bugs += 1

        if labels[:1].sum()  > 0: top1  += 1
        if labels[:5].sum()  > 0: top5  += 1
        if labels[:10].sum() > 0: top10 += 1

        prec, n_rel = [], 0
        for i, lbl in enumerate(labels):
            if lbl == 1:
                n_rel += 1
                prec.append(n_rel / (i + 1))
        ap_list.append(float(np.mean(prec)) if prec else 0.0)

        hits = np.where(labels == 1)[0]
        rr_list.append(1.0 / (hits[0] + 1) if len(hits) else 0.0)

    if n_bugs == 0:
        return {'Top-1': 0.0, 'Top-5': 0.0, 'Top-10': 0.0, 'MAP': 0.0, 'MRR': 0.0}

    return {
        'Top-1':  top1  / n_bugs,
        'Top-5':  top5  / n_bugs,
        'Top-10': top10 / n_bugs,
        'MAP':    float(np.mean(ap_list)),
        'MRR':    float(np.mean(rr_list)),
    }


# ─────────────────────────────────────────────────────────────────────────────
#  Main entry point
# ─────────────────────────────────────────────────────────────────────────────

def run_flim_experiment(
    source_project:   str,
    target_project:   str,
    source_train_ids: list,
    target_train_ids: list,
    target_test_ids:  list,
    scenario:         str,
) -> dict:
    """
    Run one FLIM experiment for a given source→target pair and scenario.

    Scenarios
    ---------
    WP-small      : source_train_ids=[],        target_train_ids=~10 % of target
    WP-large      : source_train_ids=[],        target_train_ids=~80 % of target
    CP-cold-start : source_train_ids=all source, target_train_ids=[]
    CP-transfer   : source_train_ids=all source, target_train_ids=~20 % of target

    Returns
    -------
    dict  {'Top-1', 'Top-5', 'Top-10', 'MAP', 'MRR'}  — all floats in [0, 1]
    """
    print(f"\n[FLIM] {source_project} → {target_project} | {scenario}")

    # ── Load shared metadata ─────────────────────────────────────────────────
    meta_df     = pd.read_parquet(PROJECTS_METADATA_PATH)
    bug_reports = pd.read_parquet(BUG_REPORTS_PATH)
    bug_texts   = dict(zip(
        bug_reports['bug_id'].astype(str),
        bug_reports['bug_report_text'].fillna(''),
    ))

    src_bug_db, src_blob_db, src_lang = _load_project_databases(source_project, meta_df)
    tgt_bug_db, tgt_blob_db, tgt_lang = _load_project_databases(target_project, meta_df)

    src_repo = git.Repo(os.path.join(
        REPO_BASE_PATH, src_lang.lower(), source_project.replace('/', '_')
    ))
    tgt_repo = git.Repo(os.path.join(
        REPO_BASE_PATH, tgt_lang.lower(), target_project.replace('/', '_')
    ))

    # ────────────────────────────────────────────────────────────────────────
    #  LAYER 1 — FAISS top-300 candidates  (BGE embeddings, no CodeBERT)
    # ────────────────────────────────────────────────────────────────────────
    src_train_cands, tgt_train_cands = {}, {}

    if source_train_ids:
        print(f"[FLIM]   FAISS: {len(source_train_ids)} source train bugs …")
        src_train_cands = _faiss_candidates(
            source_train_ids, src_bug_db, src_blob_db, src_repo, is_training=True,
        )

    if target_train_ids:
        print(f"[FLIM]   FAISS: {len(target_train_ids)} target train bugs …")
        tgt_train_cands = _faiss_candidates(
            target_train_ids, tgt_bug_db, tgt_blob_db, tgt_repo, is_training=True,
        )

    print(f"[FLIM]   FAISS: {len(target_test_ids)} target test bugs …")
    tgt_test_cands = _faiss_candidates(
        target_test_ids, tgt_bug_db, tgt_blob_db, tgt_repo, is_training=False,
    )

    # ────────────────────────────────────────────────────────────────────────
    #  LAYER 2a — Build CodeBERT fine-tuning pairs
    # ────────────────────────────────────────────────────────────────────────
    all_bug_texts: list[str] = []
    all_func_texts: list[str] = []
    all_labels: list[int]    = []

    if src_train_cands:
        print(f"[FLIM]   Building source fine-tune pairs …")
        b, f, l = _build_finetune_pairs(
            src_train_cands, bug_texts, src_repo, src_lang,
        )
        all_bug_texts.extend(b)
        all_func_texts.extend(f)
        all_labels.extend(l)

    if tgt_train_cands:
        print(f"[FLIM]   Building target fine-tune pairs …")
        b, f, l = _build_finetune_pairs(
            tgt_train_cands, bug_texts, tgt_repo, tgt_lang,
        )
        all_bug_texts.extend(b)
        all_func_texts.extend(f)
        all_labels.extend(l)

    # ────────────────────────────────────────────────────────────────────────
    #  LAYER 2b — Fine-tune CodeBERT  (the trainable component of FLIM)
    # ────────────────────────────────────────────────────────────────────────
    encoder = CodeBERTEncoder()

    if all_bug_texts:
        encoder.fine_tune(
            all_bug_texts, all_func_texts, all_labels,
            epochs              = FINETUNE_EPOCHS,
            batch_size          = FINETUNE_BATCH_SIZE,
            lr                  = FINETUNE_LR,
            max_steps_per_epoch = FINETUNE_MAX_STEPS_EPOCH,
        )
    else:
        print("[FLIM]   No training pairs — using pre-trained CodeBERT without fine-tuning.")

    # ────────────────────────────────────────────────────────────────────────
    #  LAYER 2c — Compute 4 FLIM features with fine-tuned CodeBERT
    # ────────────────────────────────────────────────────────────────────────
    train_dfs = []

    if src_train_cands:
        print(f"[FLIM]   Feature extraction: source train …")
        df = _compute_flim_features(
            src_train_cands, bug_texts, src_repo, src_lang, encoder,
        )
        if not df.empty:
            train_dfs.append(df)

    if tgt_train_cands:
        print(f"[FLIM]   Feature extraction: target train …")
        df = _compute_flim_features(
            tgt_train_cands, bug_texts, tgt_repo, tgt_lang, encoder,
        )
        if not df.empty:
            train_dfs.append(df)

    if not train_dfs:
        print("[FLIM]   No training features — returning zero metrics.")
        return {'Top-1': 0.0, 'Top-5': 0.0, 'Top-10': 0.0, 'MAP': 0.0, 'MRR': 0.0}

    train_df = pd.concat(train_dfs)
    n_train  = train_df.index.get_level_values(0).nunique()
    print(f"[FLIM]   Train set: {n_train} bugs, {len(train_df)} (bug, file) pairs")

    # ────────────────────────────────────────────────────────────────────────
    #  LAYER 2d — Train Adaptive LTR on the 4 FLIM features
    # ────────────────────────────────────────────────────────────────────────
    ranker = FLIMRanker(columns=FLIM_FEATURES, cv_folds=2)
    ranker.fit(train_df)
    print(f"[FLIM]   LTR: weight_method={ranker.weight_method_name} | "
          f"use_regressor={ranker.use_regressor}")

    # ────────────────────────────────────────────────────────────────────────
    #  LAYER 2e — Test feature extraction + evaluation
    # ────────────────────────────────────────────────────────────────────────
    print(f"[FLIM]   Feature extraction: test ({len(target_test_ids)} bugs) …")
    test_df = _compute_flim_features(
        tgt_test_cands, bug_texts, tgt_repo, tgt_lang, encoder,
    )

    if test_df.empty:
        print("[FLIM]   No test features — returning zero metrics.")
        return {'Top-1': 0.0, 'Top-5': 0.0, 'Top-10': 0.0, 'MAP': 0.0, 'MRR': 0.0}

    test_df          = test_df.copy()
    test_df['result'] = ranker.predict_scores(test_df)
    metrics          = _compute_metrics(test_df)

    print(f"[FLIM]   Top-1={metrics['Top-1']:.3f}  Top-5={metrics['Top-5']:.3f}  "
          f"Top-10={metrics['Top-10']:.3f}  MAP={metrics['MAP']:.3f}  "
          f"MRR={metrics['MRR']:.3f}")
    return metrics
