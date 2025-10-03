import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
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
        convolved = [F.relu(conv(embedded)).squeeze(3) for conv in self.convs]
        pooled = [F.max_pool1d(item, item.size(2)).squeeze(2) for item in convolved]
        
        # Concatenate features and apply dropout
        cat = torch.cat(pooled, 1)
        return self.dropout(cat)
    
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
            [nn.Conv1d(stmt_output_dim, file_kernels, k, padding="same") for k in file_kernel_sizes]
        )
        self.dropout = nn.Dropout(dropout)
        # The output dimension will be the total number of filters
        self.output_dim = file_kernels * len(file_kernel_sizes)

    def forward(self, x):
        '''
        x input shape: (batch_size, num_statements, statement_length)
        '''
        batch_size, num_stmts, stmt_len= x.shape

        # To process efficiently, flatten all statements into a single large batch
        x_embed = self.embedding(x) # Shape: (batch_size, num_stmts, stmt_len, embed_dim)
        x_flat = x_embed.view(batch_size * num_stmts, 1, stmt_len, -1)

        # Get a vector for each statement
        stmt_features = [F.relu(conv(x_flat)).squeeze(3) for conv in self.statement_convs]
        stmt_pooled = [F.max_pool1d(item, item.size(2)).squeeze(2) for item in stmt_features]
        stmt_vectors = torch.cat(stmt_pooled, 1)

        # Reshape back to the file structure
        x_reshaped = stmt_vectors.view(batch_size, num_stmts, -1).permute(0, 2, 1)
        
        file_features = [F.relu(conv(x_reshaped)) for conv in self.file_convs]
        file_pooled = [F.max_pool1d(item, item.size(2)).squeeze(2) for item in file_features]
        
        final_vector = torch.cat(file_pooled, 1)
        final_vector = self.dropout(final_vector)
        return final_vector

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

        self.p_cnn = P_CNN(
            params['vocab_size'], params['code_embedding_dim'],
            params['stmt_kernels'], params['stmt_kernel_sizes'],
            params['file_kernels'], params['file_kernel_sizes'], params['dropout']
        )

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
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes)
        )
    
    def forward(self, bug_report_ids, source_code_ids, project_type):
        # Pass inputs through the shared feature extractors
        nl_features = self.n_cnn(bug_report_ids)
        pl_features = self.p_cnn(source_code_ids)

        # print("Shape of nl_features:", nl_features.shape)
        # print("Shape of pl_features:", pl_features.shape)

        # Concatenate the features
        features = torch.cat([nl_features, pl_features], dim=1)

        # Route to the correct project-specific prediction head
        if project_type == 'source':
            logits = self.fc_source(features)
        elif project_type == 'target':
            logits = self.fc_target(features)
        else:
            raise ValueError("project_type must be 'source' or 'target'")
        
        return logits
