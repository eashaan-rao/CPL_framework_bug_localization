import numpy as np
import pandas as pd
from tqdm import tqdm
import os
import pickle
import re
import torch
from transformers import AutoModel, AutoTokenizer

# ── GloVe constants ──────────────────────────────────────────────────────────
GLOVE_DIM = 300

# Module-level cache so GloVe is only loaded once per process
_GLOVE_CACHE = None

def load_glove(glove_path):
    """
    Load GloVe 840B 300d vectors from text file into a {word: np.float32 array} dict.
    Saves/loads a fast numpy binary cache alongside the text file for subsequent runs.
    """
    global _GLOVE_CACHE
    if _GLOVE_CACHE is not None:
        return _GLOVE_CACHE

    npy_path = glove_path.replace('.txt', '_cache.npy')
    words_path = glove_path.replace('.txt', '_words.pkl')

    if os.path.exists(npy_path) and os.path.exists(words_path):
        print("Loading GloVe from binary cache...")
        matrix = np.load(npy_path)
        with open(words_path, 'rb') as f:
            words = pickle.load(f)
        glove = {w: matrix[i] for i, w in enumerate(words)}
    else:
        print(f"Loading GloVe from {glove_path} (first time, building binary cache)...")
        words, vecs = [], []
        with open(glove_path, 'r', encoding='utf-8', errors='ignore') as f:
            for line in tqdm(f, desc="Reading GloVe"):
                parts = line.rstrip().split(' ')
                if len(parts) != GLOVE_DIM + 1:
                    continue
                words.append(parts[0])
                vecs.append(np.array(parts[1:], dtype='float32'))
        matrix = np.stack(vecs)
        np.save(npy_path, matrix)
        with open(words_path, 'wb') as f:
            pickle.dump(words, f)
        glove = {w: matrix[i] for i, w in enumerate(words)}
        print(f"GloVe loaded: {len(glove):,} vectors")

    _GLOVE_CACHE = glove
    return glove


def _split_identifier(token):
    """Split a camelCase or underscore_separated identifier into lowercase subtokens."""
    parts = token.split('_')
    result = []
    for part in parts:
        # split camelCase: 'FunctionDef' → ['Function', 'Def']
        words = re.findall('[A-Z][a-z]*|[a-z]+|[A-Z]+(?=[A-Z]|$)', part)
        result.extend([w.lower() for w in words] if words else [part.lower()])
    return [r for r in result if r]


def glove_lookup(token, glove_dict):
    """
    Look up a token in GloVe.  For OOV tokens (code identifiers, AST node type names)
    splits on underscores and camelCase and averages sub-token vectors.
    Returns a zero vector if nothing matches.
    """
    t = token.lower()
    if t in glove_dict:
        return glove_dict[t]
    subtokens = _split_identifier(token)
    vecs = [glove_dict[s] for s in subtokens if s in glove_dict]
    if vecs:
        return np.mean(vecs, axis=0).astype('float32')
    return np.zeros(GLOVE_DIM, dtype='float32')


def text_to_glove_sequence(text, glove_dict, max_len=512):
    """
    Tokenise plain text, look up each word in GloVe, pad/truncate to max_len.
    Returns:
        seq   : np.float32 array of shape (max_len, GLOVE_DIM)
        length: int — actual number of tokens (clamped to [1, max_len])
    """
    tokens = re.findall(r'[a-zA-Z]+', text)[:max_len]
    length = max(len(tokens), 1)
    seq = np.zeros((max_len, GLOVE_DIM), dtype='float32')
    for i, tok in enumerate(tokens):
        seq[i] = glove_lookup(tok, glove_dict)
    return seq, length

def get_bge_embeddings(texts, model, tokenizer, device='cuda', max_length= 512, batch_size=32):
    '''
    Get BGE embeddings for a list of texts.
    Args:
        texts (list): List of text strings to embed
        model: BGE model
        tokenizer: BGE tokenizer
        device: Device to run on
        max_length: Maximum sequence length
        batch_size: Batch size for processing
    Returns:
        np.ndarray: Embeddings of shape (num_texts, embedding_dim)
    '''
    model = model.to(device)
    model.eval()
    all_embeddings = []

    for i in tqdm(range(0, len(texts), batch_size), desc="Computing BGE Embeddings"):
        batch_texts = texts[i:i + batch_size]

        # Tokenize batch
        inputs = tokenizer(
            batch_texts,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors='pt'
        ).to(device)

        # Get embeddings
        with torch.no_grad():
            outputs = model(**inputs)
            # Use mean pooling over sequence
            embeddings = outputs.last_hidden_state.mean(dim=1)
            # Alternative: Use CLS token
            # embeddings = outputs.last_hidden_state[:, 0, :]
        
        all_embeddings.append(embeddings.cpu().numpy())

    return np.vstack(all_embeddings)

def prepare_bug_embeddings(bug_reports_df, model_name="BAAI/bge-code-v1", device='cuda'):
    '''
    Prepare BGE embeddings for bug reports.
    Args:
        bug_reports_df = Dataframe with 'bug_id' and 'bug_report_text' columns
        model_name: Name of the BGE model
        device: Device to run on
    Returns:
        dict: Mapping from bug_id to embeddings
    '''
    print(f"Loading BGE model: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)

    # Extract texts
    bug_texts = bug_reports_df['bug_report_text'].tolist()
    bug_ids = bug_reports_df['bug_id'].tolist()

    # Get embeddings
    embeddings = get_bge_embeddings(bug_texts, model, tokenizer, device)

    # Create mapping
    bug_embeddings = {
        bug_id: embedding
        for bug_id, embedding in zip(bug_ids, embeddings)
    }
    return bug_embeddings

def prepare_code_embeddings(code_files, model_name="BAAI/bge-code-v1", device='cuda', max_length=1024):
    '''
    Prepare BGE embeddings for code files.
    Args:
        code_files = List of (file_id, code_content) tuples
        model_name: Name of the BGE model
        device: Device to run on
        max_length: Maximum code length
    Returns:
        dict: Mapping from file_id to embeddings
    '''
    print(f"Loading BGE model for code: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)

    file_ids = [f[0] for f in code_files]
    code_contents = [f[1] for f in code_files]

    # Get embeddings
    embeddings = get_bge_embeddings(code_contents, model, tokenizer, device, max_length)
    # Create mapping
    code_embeddings = {
        file_id: embedding
        for file_id, embedding in zip(file_ids, embeddings)
    }
    return code_embeddings

def preprocess_code_to_ast_embeddings(code_string, ast_parser, bge_model, bge_tokenizer, 
                                      device='cuda', max_nodes=500):
    '''
    Convert code string to AST graph with BGE embeddings as node features.
    Args:
        code_string: Source code as string
        ast_parser: AST parser for the language
        bge_model: BGE model for embeddings
        bge_tokenizer: BGE tokenizer
        device: Device to run on
        max_nodes = Maximum number of nodes in graph
    Returns:
        torch_geometric.data.Data: Graph with BGE embeddings as node features
    '''
    # Parse code to AST
    graph_data = ast_parser.parse(code_string)

    if graph_data is None:
        # Return empty graph if parsing fails
        return torch.geometric.data.Data(
            x=torch.zeros(1, 1536), # BGE embedding dimension
            edge_index=torch.tensor([[], []], dtype=torch.long)
        )
    
    # Get code embedding
    inputs = bge_tokenizer(
        code_string,
        padding=True,
        truncation=True,
        max_length=1024,
        return_tensors='pt'
    ).to(device)

    with torch.no_grad():
        outputs = bge_model(**inputs)
        code_embedding = outputs.last_hidden_state.mean(dim=1).cpu()

    # Limit number of nodes
    num_nodes = min(graph_data.x.shape[0], max_nodes)

    # Create node embeddings (use same embedding for all nodes as a simple approach)
    # In a more sophisticated verion, you could embed each node's code snippet separately
    node_embeddings = code_embedding.expand(num_nodes, -1)

    # Update graph data
    graph_data.x = node_embeddings

    # Truncate edges if needed
    if num_nodes < graph_data.x.shape[0]:
        mask = (graph_data.edge_index[0] < num_nodes) & (graph_data.edge_index[1] < num_nodes)
        graph_data.edge_index = graph_data.edge_index[:, mask]

    return graph_data

def save_embeddings(embeddings_dict, filepath):
    '''
    Save embeddings dictionary to file.
    Args:
        embeddings_dict: Dictionary of embeddings
        filepath: Path to save file
    '''
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, 'wb') as f:
        pickle.dump(embeddings_dict, f)
    print(f"Embeddings saved to {filepath}")

def load_embeddings(filepath):
    '''
    Load embeddings dictionary from file.
    Args:
        filepath: Path to saved file
    Returns: 
        dict: Dictionary of embeddings
    '''
    with open(filepath, 'rb') as f:
        embeddings_dict = pickle.load(f)
    print(f"Embeddings loaded from {filepath}")
    return embeddings_dict

def compute_similarity_matrix(bug_embeddings, code_embeddings):
    '''
    Compute cosine similarity matrix between bug and code embeddings.
    Args:
        bug_embeddings: Array of shape (num_bugs, embedding_dim)
        code_embeddings: Array of shape (num_codes, embedding_dim)
    Returns:
        np.ndarray: Similarity matric of shape (num_bugs, num_codes)
    '''
    # Normalize embeddings
    bug_norm = bug_embeddings / np.linalg.norm(bug_embeddings, axis=1, keepdims=True)
    code_norm = code_embeddings / np.linalg.norm(code_embeddings, axis=1, keepdims=True)

    # Compute cosine similarity
    similarity = np.dot(bug_norm, code_norm.T)
    return similarity

def create_triplets_for_margin_loss(samples, max_triplets_per_bug=10):
    '''
    Create triplets (bug, positive_code, negative_code) for margin ranking loss.
    Args:
        samples: List of (bug_id, code_id, label) tuples
        max_triplets_per_bug: Maximum triplets to create per bug
    Returns:
        list: List of (bug_id, pos_code_id, neg_code_id) triplets
    '''
    # Group by bug_id
    bug_samples = {}
    for bug_id, code_id, label in samples:
        if bug_id not in bug_samples:
            bug_samples[bug_id] = {'positive': [], 'negative':[]}

        if label == 1:
            bug_samples[bug_id]['positive'].append(code_id)
        else:
            bug_samples[bug_id]['negative'].append(code_id)
    
    # Create triplets
    triplets = []
    for bug_id, codes in bug_samples.items():
        pos_codes = codes['positive']
        neg_codes = codes['negative']
        
        if not pos_codes or not neg_codes:
            continue

        # Create all possible combinations (limited by max_triplets_per_bug)
        count = 0
        for pos_code in pos_codes:
            for neg_code in neg_codes:
                triplets.append((bug_id, pos_code, neg_code))
                count += 1
                if count >= max_triplets_per_bug:
                    break
            if count >- max_triplets_per_bug:
                break
    
    return triplets

def calculate_metrics(predictions, ground_truth, k_values=[1, 5, 10]):
    '''
    Calculate retrieval metrics.
    Args:
        predictions: Dict mapping bug_id to list of (code_id, score) tuples
        ground_truth: Dict mapping bug_id to list of relevant code_ids
        k_values: List of k values for Top-K accuracy

    Returns:
        dict: Dictionary of metrics
    '''
    metrics = {}

    # Initialize counters
    top_k_hits = {k: 0 for k in k_values}
    mrr_scores = []
    map_scores = []

    for bug_id, pred_list in predictions.items():
        # Sort by score
        sorted_preds = sorted(pred_list, key=lambda x: x[1], reverse=True)
        ranked_codes = [code_id for code_id, _ in sorted_preds]
        relevant_codes = set(ground_truth.get(bug_id, []))

        if not relevant_codes: continue

        # Top-K accuracy
        for k in k_values:
            if len(set(ranked_codes[:k]) & relevant_codes) > 0:
                top_k_hits[k] += 1
        
        # MRR (Mean Reciprocal Rank)
        for i, code_id in enumerate(ranked_codes):
            if code_id in relevant_codes:
                mrr_scores.append(1.0 / (i + 1))
                break
            else:
                mrr_scores.append(0.0)

        # MAP (Mean Average Precision)
        precisions = []
        hits = 0
        for i, code_id in enumerate(ranked_codes):
            if code_id in relevant_codes:
                hits += 1
                precisions.append(hits / (i + 1))
        if precisions:
            map_scores.append(np.mean(precisions))
        else:
            map_scores.append(0.0)
    
    # Calculate final metrics
    num_queries = len(predictions)

    for k in k_values:
        metrics[f'Top-{k}'] = top_k_hits[k] / num_queries if num_queries > 0 else 0
    
    metrics['MRR'] = np.mean(mrr_scores) if mrr_scores else 0
    metrics['MAP'] = np.mean(map_scores) if map_scores else 0

    return metrics

def save_preprocessors(vocab, embedding_matrix, path):
    '''
    Saves preprocessor data (compatibility function.)
    '''
    data = {
        'vocabulary': vocab,
        'embedding_matrix': embedding_matrix,
        'embedding_type': 'BGE'
    }
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'wb') as f:
        pickle.dump(data, f)
    print(f"Preprocessors saved to {path}")

def load_preprocessors(path):
    '''
    Loads preprocessor data (compatibility function)
    '''
    if os.path.exists(path):
        with open(path, 'rb') as f:
            data = pickle.load(f)
        return data.get('vocabulary', {}), data.get('embedding_matrix', np.zeros((1, 768)))
    else:
        print(f"Preprocessor file not found at {path}, returning defaults")
        return {}, np.zeros((1, 768))






