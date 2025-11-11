import numpy as np
import pandas as pd
from tqdm import tqdm
import os
import pickle

# Define special tokens
PAD_TOKEN = '<PAD>'
UNK_TOKEN = '<UNK>' 

def build_vocabulary(bug_reports_df, ast_parsers, all_code_files, min_freq=3):
    '''
    Builds a vocabulary from all bug reports and code file AST tokens.
    
    Args:
        bug_reports_df (pd.DataFrame): DataFrame containing 'bug_report_text.
        ast_parsers (dict): {'java': JavaASTParser, 'python': PythonASTParser}
        all_code_files (list of tuples): [(code_string, language), ...]
        min_freq (int): Minimum frequency for a token to be included in vocab.
        
    Returns:
        dict: A dictionary mapping tokens (str) to integer IDs.
    '''
    print("Building vocabulary...")
    token_counts = {}

    # 1. Process bug reports
    print("Processing bug reports...")
    for text in tqdm(bug_reports_df['bug_report_text']):
        for token in str(text).split():
            token_counts[token] = token_counts.get(token, 0) + 1

    # 2. Process all code files
    print("Processing code files...")
    for code_string, language in tqdm(all_code_files):
        parser = ast_parsers.get(language)
        if not parser:
            continue  # Skip unsupported languages
        
        # We need a way to get tokens without building the full graph
        # let's assume a helper function in the parsers
        try:
            tokens = parser.get_tokens(code_string)
            for token in tokens:
                token_counts[token] = token_counts.get(token, 0) + 1
        except Exception:
            continue # Skip files that fail to parse
    
    # 3. Filter by frequency and create vocabulary
    print(f"Total unique tokens found : {len(token_counts)}")
    vocab = {PAD_TOKEN: 0, UNK_TOKEN: 1}
    vocab_idx = 2
    for token, count in token_counts.items():
        if count >= min_freq:
            vocab[token] = vocab_idx
            vocab_idx += 1

    print(f"Final vocabulary size (min_freq = {min_freq}): {len(vocab)} ")
    return vocab
    
def load_glove_embeddings(glove_file_path, vocabulary, embedding_dim):
    '''
    Loads GloVe embeddings for the words in our vocabulary.
    
    Args:
        glove_file_path (str): Path to the GloVe.txt file.
        vocabulary (dict): The word-to-ID mapping.
        embedding_dim (int): The embedding dimension (e.g. 300).
    
    Returns:
        np.ndarray: An embedding matrix of shape (vocab_size, embedding_dim).
    '''
    print("Loading GloVe embeddings...")
    embeddings = {}
    with open(glove_file_path, 'r', encoding='utf-8') as f:
        for line in tqdm(f):
            parts = line.strip()
            word = parts[0]
            if word in vocabulary:
                embeddings[word] = np.array(parts[1:], dtype=np.float32)
    
    print(f"Found embeddings for {len(embeddings)} / {len(vocabulary)} words.")

    vocab_size = len(vocabulary)
    embedding_matrix = np.random.rand(vocab_size, embedding_dim).astype(np.float32) * 0.05
    embedding_matrix[vocabulary[PAD_TOKEN]] = np.zeros(embedding_dim, dtype=np.float32)

    for word, idx in vocabulary.items():
        if word in embeddings:
            embedding_matrix[idx] = embeddings[word]
    
    return embedding_matrix

def save_preprocessors(vocab, embedding_matrix, path):
    '''
    Saves the vocabulary and embedding matrix.
    
    Args:
        vocab (dict): The word-to-ID mapping.
        embedding_matrix (np.ndarray): The embedding matrix.
        save_path (str): Path to save the preprocessors.
    '''
    with open(path, 'wb') as f:
        pickle.dump({'vocabulary': vocab, 'embedding_matrix': embedding_matrix}, f)
    print(f"Preprocessors saved to {path}")

def load_preprocessors(path):
    '''
    Loads the vocabulary and embedding matrix.
    
    Args:
        path (str): Path to the saved preprocessors.
    Returns:
        tuple: (vocabulary (dict), embedding_matrix (np.ndarray))
    '''
    with open(path, 'rb') as f:
        data = pickle.load(f)
    print(f"Preprocessors loaded from {path}")
    return data['vocabulary'], data['embedding_matrix']