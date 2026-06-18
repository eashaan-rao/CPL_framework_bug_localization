import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.checkpoint as ckpt
from torch.utils.data import DataLoader, Dataset

class N_CNN(nn.Module):
    '''
    The CNN for Natural Language (Bug Reports). It takes pre-computed embeddings as input.
    Correctly handles a pre-computed 2D vector (batch, embed_dim).
    Based on Kim's CNN for text classification.
    '''
    
    def __init__(self, vocab_size, embeding_dim, num_kernels, kernel_sizes, dropout):
        super(N_CNN, self).__init__()
        # 1. Add an embedding layer for the bug report vocabular
        self.embedding = nn.Embedding(vocab_size, embeding_dim, padding_idx=0)

        # 2. Define convolutions that will slide over the sequence of word embeddings
        self.convs = nn.ModuleList(
            # The input channel is 1. The convolution happens over the sequence of embeddings.
            [nn.Conv2d(1, num_kernels, (k, embeding_dim)) for k in kernel_sizes]
        )
        self.dropout = nn.Dropout(dropout)
        # The output dimension will be the total number of filters
        self.output_dim = num_kernels * len(kernel_sizes)

    def forward(self, x):
        '''
        x is now a 2D tensor of token IDs: (batch_size, sequence_length)
        '''
        # Conver token IDs to a sequence of embeddings
        # x is a 2D tensor of token IDs: (batch_size, sequence_length)
        embedded = self.embedding(x)  # -> (batch, seq_len, embed_dim)
        
        # Add the "channel" dimension for Conv2d
        embedded = embedded.unsqueeze(1) # -> (batch, 1, seq_len, embed_dim)
        
        # Apply convolutions, then global max-pooling
        pooled = []
        for conv in self.convs:
            conv_out = F.relu(conv(embedded), inplace=True) # inplace ReLU
            conv_out = conv_out.squeeze(3) # (batch, num_kernels, seq_len)
            # Global max pooling
            pooled_out = F.adaptive_max_pool1d(conv_out, 1).squeeze(2)
            pooled.append(pooled_out)
        cat = torch.cat(pooled, dim=1)
        
        # convolved = [F.relu(conv(embedded)).squeeze(3) for conv in self.convs]
        # pooled = [F.max_pool1d(item, item.size(2)).squeeze(2) for item in convolved]
        
        # # Concatenate features and apply dropout
        # cat = torch.cat(pooled, 1)
        return self.dropout(cat)
    
class P_CNN_Parallelized(nn.Module):
    '''
    Optimized P_CNN with chunked parallel processing to avoid sequential bottleneck.
    '''
    def __init__(self, vocab_size, embedding_dim, stmt_kernels, stmt_kernel_sizes, file_kernels, file_kernels_sizes, dropout,
                 chunk_size):
        super(P_CNN_Parallelized, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.chunk_size = chunk_size # Process statements in chunks

        # Stage 1: Within-statement Convolutions
        self.statement_convs = nn.ModuleList([
            nn.Conv2d(1, stmt_kernels, (k, embedding_dim)) for k in stmt_kernel_sizes
        ])

        # Stage 2: Between-statement convolutions
        stmt_output_dim = stmt_kernels * len(stmt_kernel_sizes)
        self.file_convs = nn.ModuleList([
            nn.Conv1d(stmt_output_dim, file_kernels, k, padding=k//2) for k in file_kernels_sizes
        ])
        self.dropout = nn.Dropout(dropout)
        self.output_dim = file_kernels * len(file_kernels_sizes)

    def forward(self, x):
        '''
        x: (batch_size, num_statements, statement_length)
        '''
        batch_size, num_stmts, stmt_len = x.shape
        # Embed all tokens at once
        x_embed = self.embedding(x) # (batch, num_stmts, stmt_len, embed_dim)

        # Key optimization: Process statements in manageable chunks, instead of creating one massive 
        # (99200, 1, 14, 512) tensor, we process in chunks of size (5000, 1, 14, 512)

        stmt_vectors_list = []
        total_stmts = batch_size * num_stmts

        # Flatten for processing
        x_embed_flat = x_embed.view(total_stmts, stmt_len, -1)

        # Process in chunks
        for chunk_start in range(0, total_stmts, self.chunk_size):
            chunk_end = min(chunk_start + self.chunk_size, total_stmts)

            # Get chunk and add channel dimension
            chunk = x_embed_flat[chunk_start:chunk_end].unsqueeze(1) # (chunk_size, 1, stmt_len, embed_dim)

            # Stage 1: Statement-level features (parallel within chunk)
            stmt_pooled = []
            for conv in self.statement_convs:
                conv_out = F.relu(conv(chunk), inplace=True).squeeze(3)
                pooled_out = F.adaptive_max_pool1d(conv_out, 1).squeeze(2)
                stmt_pooled.append(pooled_out)

            chunk_vectors = torch.cat(stmt_pooled, dim=1)
            stmt_vectors_list.append(chunk_vectors)

        # Concatenate all chunks
        stmt_vectors = torch.cat(stmt_vectors_list, dim=0)

        # Reshape back to file structure
        stmt_vectors = stmt_vectors.view(batch_size, num_stmts, -1)
        x_file = stmt_vectors.permute(0, 2, 1) # (batch, features, num_stmts)

        # Stage 2: File-level features
        file_pooled = []
        for conv in self.file_convs:
            conv_out = F.relu(conv(x_file), inplace=True)
            pooled_out = F.adaptive_max_pool1d(conv_out, 1).squeeze(2)
            file_pooled.append(pooled_out)

        final_vector = torch.cat(file_pooled, dim=1)
        return self.dropout(final_vector)

    
class P_CNN(nn.Module):
    '''
    The specialized, hierarchical CNN for Programming Language (Source Code). It implements the "within-statement" and
    "between-statement" concept from the paper. It also takes pre-computed embeddings as input.
    '''
    
    def __init__(self, vocab_size, embedding_dim, stmt_kernels, stmt_kernel_sizes, file_kernels, file_kernel_sizes, dropout):
        super(P_CNN, self).__init__()

        # This CNN learns its own embeddings for code tokens.
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        
        # Stage 1: "Within-Statement" Feature Extractor
        # This is a standad TextCNN that processes each "statement" (or line) of code.
        self.statement_convs = nn.ModuleList(
            [nn.Conv2d(1, stmt_kernels, (k, embedding_dim)) for k in stmt_kernel_sizes]
        )

        # Stage 2: "Between-Statement" Feature Extractor
        # This CNN operates on the sequence of statement vectors produced by Stage 1.
        stmt_output_dim = stmt_kernels * len(stmt_kernel_sizes)
        self.file_convs = nn.ModuleList(
            [nn.Conv1d(stmt_output_dim, file_kernels, k, padding=k//2) for k in file_kernel_sizes]
        )
        self.dropout = nn.Dropout(dropout)
        # The output dimension will be the total number of filters
        self.output_dim = file_kernels * len(file_kernel_sizes)

    def forward(self, x):
        '''
        x input shape: (batch_size, num_statements, statement_length)
        '''
        batch_size, num_stmts, stmt_len= x.shape
        # print(f"\nP_CNN Input: batch={batch_size}, statements={num_stmts}, tokens={stmt_len}")
        # print(f"Total sub-batches to process: {batch_size * num_stmts}")

        # To process efficiently, flatten all statements into a single large batch
        x_embed = self.embedding(x) # Shape: (batch_size, num_stmts, stmt_len, embed_dim)
        x_flat = x_embed.view(batch_size * num_stmts, 1, stmt_len, -1)

        # Stage 1: Statement-level features with fused operations
        stmt_pooled = []
        for conv in self.statement_convs:
            conv_out = F.relu(conv(x_flat), inplace=True).squeeze(3)
            pooled_out = F.adaptive_max_pool1d(conv_out, 1).squeeze(2)
            stmt_pooled.append(pooled_out)

        # stmt_features = [F.relu(conv(x_flat)).squeeze(3) for conv in self.statement_convs]
        # stmt_pooled = [F.max_pool1d(item, item.size(2)).squeeze(2) for item in stmt_features]
        stmt_vectors = torch.cat(stmt_pooled, dim=1)

        # Reshape back to the file-level processing
        stmt_vectors = stmt_vectors.view(batch_size, num_stmts, -1)
        x_file = stmt_vectors.permute(0, 2, 1)
        
        # Stage 2: File-level features with fused operations
        file_pooled = []
        for conv in self.file_convs:
            conv_out = F.relu(conv(x_file), inplace=True)
            pooled_out = F.adaptive_max_pool1d(conv_out, 1).squeeze(2)
            file_pooled.append(pooled_out)

        # file_features = [F.relu(conv(x_reshaped)) for conv in self.file_convs]
        # file_pooled = [F.max_pool1d(item, item.size(2)).squeeze(2) for item in file_features]
        
        final_vector = torch.cat(file_pooled, dim=1)
        final_vector = self.dropout(final_vector)
        return final_vector

class P_CNN_FullyParallel(nn.Module):
    '''
    Alternative: Used grouped convolutions for true parallelism. This processes all statements
    simulataneously using PyTorch's native batching
    '''
    def __init__(self, vocab_size, embedding_dim, stmt_kernels, stmt_kernel_sizes, file_kernels,
                 file_kernel_sizes, dropout):
        super(P_CNN_FullyParallel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)

        # Stage 1: Use 1D convolutions instead of 2D for better parallelism
        self.statement_convs = nn.ModuleList(
            [nn.Conv1d(embedding_dim, stmt_kernels, k, padding=k//2) for k in stmt_kernel_sizes]
        )

        # Stage 2: Between-Statement convolutions
        stmt_output_dim = stmt_kernels * len(stmt_kernel_sizes)
        self.file_convs = nn.ModuleList([
            nn.Conv1d(stmt_output_dim, file_kernels, k, padding=k//2) for k in file_kernel_sizes
        ])
        self.dropout = nn.Dropout(dropout)
        self.output_dim = file_kernels * len(file_kernel_sizes)
    
    def _stmt_conv(self, x_flat):
        '''Statement-level convolutions. Extracted for gradient checkpointing.'''
        stmt_features = []
        for conv in self.statement_convs:
            conv_out = F.relu(conv(x_flat), inplace=True)
            pooled = F.adaptive_max_pool1d(conv_out, 1).squeeze(2)
            stmt_features.append(pooled)
        return torch.cat(stmt_features, dim=1)

    def forward(self, x):
        '''
        x: (batch_size, num_statements, statement_length)
        '''
        batch_size, num_stmts, stmt_len = x.shape

        x_embed = self.embedding(x)
        x_flat = x_embed.view(batch_size * num_stmts, stmt_len, -1).transpose(1, 2)

        # Checkpoint the statement convolutions: avoids storing large intermediate
        # conv activations (~400-600 MB per batch) at the cost of one extra forward
        # through _stmt_conv during backward. use_reentrant=False = modern API,
        # compatible with torch.compile.
        stmt_vectors = ckpt.checkpoint(self._stmt_conv, x_flat, use_reentrant=False)

        x_file = stmt_vectors.view(batch_size, num_stmts, -1).transpose(1, 2)

        file_features = []
        for conv in self.file_convs:
            conv_out = F.relu(conv(x_file), inplace=True)
            pooled = F.adaptive_max_pool1d(conv_out, 1).squeeze(2)
            file_features.append(pooled)

        return self.dropout(torch.cat(file_features, dim=1))


class TRANPCNN(nn.Module):
    '''
    The main TRANP-CNN model.
    It combines the shared feature extractors with the project-specific prediction layers.
    '''
    def __init__(self, params):
        super(TRANPCNN, self).__init__()

        # Transferable Feature Extraction Layer (with Weight Sharing)
        # We create ONE instance of each CNN. These will be shared for source and target projects.
        self.n_cnn = N_CNN(
            params['vocab_size'], params['nl_embedding_dim'],
            params['nl_kernels'], params['nl_kernel_sizes'], params['dropout']
        )

        # self.p_cnn = P_CNN(
        #     params['vocab_size'], params['code_embedding_dim'],
        #     params['stmt_kernels'], params['stmt_kernel_sizes'],
        #     params['file_kernels'], params['file_kernel_sizes'], params['dropout']
        # )

        # Conv1d fully parallel: processes all statements in one GPU call instead of 30 Python-loop
        # chunks × 3 conv calls. Same representational capacity, ~90 kernel launches → 3.
        self.p_cnn = P_CNN_FullyParallel(
            params['vocab_size'], params['code_embedding_dim'],
            params['stmt_kernels'], params['stmt_kernel_sizes'],
            params['file_kernels'], params['file_kernel_sizes'],
            params['dropout']
        )

        # Option1 (slower): Chunk processing to avoid large intermediate tensors
        # self.p_cnn = P_CNN_Parallelized(
        #     params['vocab_size'], params['code_embedding_dim'],
        #     params['stmt_kernels'], params['stmt_kernel_sizes'],
        #     params['file_kernels'], params['file_kernel_sizes'], params['dropout'], chunk_size=5000
        # )
        
        # Project_Specific Prediction Layer
        combined_dim = self.n_cnn.output_dim + self.p_cnn.output_dim
        # hidden_dim = params['hidden_dim']

        # We create TWO separate FCNs, one for the source and one for the target project
        self.fc_source = self._create_fcn(combined_dim, params['hidden_dim'], params['num_classes'], params['dropout'])
        self.fc_target = self._create_fcn(combined_dim, params['hidden_dim'], params['num_classes'], params['dropout'])

    def _create_fcn(self, input_dim, hidden_dim, num_classes, dropout):
        '''Helper function to create a standard two-layer FCN.'''
        return nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes)
        )
    
    def forward(self, bug_report_ids, source_code_ids, project_type):
        nl_features = self.n_cnn(bug_report_ids)
        pl_features = self.p_cnn(source_code_ids)
        features = torch.cat([nl_features, pl_features], dim=1)

        if project_type == 'source':
            logits = self.fc_source(features)
        elif project_type == 'target':
            logits = self.fc_target(features)
        else:
            raise ValueError("project_type must be 'source' or 'target'")

        return logits

    def forward_joint(self, bug_ids_s, code_ids_s, bug_ids_t, code_ids_t):
        '''
        Fused forward pass for joint source+target training (CP-transfer scenario).
        Runs the shared CNN layers once on the concatenated batch instead of twice,
        then routes each half to its project-specific head.
        '''
        n_source = bug_ids_s.size(0)
        all_nl = self.n_cnn(torch.cat([bug_ids_s, bug_ids_t], dim=0))
        all_pl = self.p_cnn(torch.cat([code_ids_s, code_ids_t], dim=0))
        all_features = torch.cat([all_nl, all_pl], dim=1)
        logits_s = self.fc_source(all_features[:n_source])
        logits_t = self.fc_target(all_features[n_source:])
        return logits_s, logits_t
