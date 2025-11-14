import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool
from torch_geometric.data import Data

# 1. Shared Bug Report Encoder
class BugReportEncoder(nn.Module):
    '''
    Encodes bug reports using a Bidirectional LSTM. This module is shared across both source and 
    target projects.
    '''
    def __init__(self, embedding_dim, hidden_size, num_layers, dropout_prob=0.5):
        super(BugReportEncoder, self).__init__()
        # For BGE embeddings, we don't need a learnable embedding layer.
        # The embeddings come pre-computed from the BGE model
        
        self.lstm = nn.LSTM(embedding_dim, hidden_size, num_layers, bidirectional=True, 
                            batch_first=True, dropout=dropout_prob if num_layers > 1 else 0)
        self.dropout = nn.Dropout(dropout_prob)

    def forward(self, embedded_sequence, lengths):
        '''
        Args: 
        embedded_sequence (Tensor): Batch of padded embedded bug reports from BGE
                                    Shape: (batch_size, seq_len, embedding_dim)
        lengths (Tensor): Actual lengths of bug reports before padding
                          Shape: (batch_size, )
        Returns:
            Tensor: Bug Report representation. Shape: (batch_size, hidden_size * 2)
        '''
        # Pack padded sequence to handle variable lenghts efficiently
        packed_embedded = nn.utils.rnn.pack_padded_sequence(
            embedded_sequence, lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        # LSTM forward pass
        _, (hidden, _) = self.lstm(packed_embedded)

        # Concatenate the final hidden states from both directions.
        # hidden is of shape (num_layers * 2, batch_size, hidden_size)
        # We take the last layer's forward and backward states
        forward_hidden = hidden[-2, :, :]
        backward_hidden = hidden[-1, :, :]

        bug_vector = torch.cat((forward_hidden, backward_hidden), dim=1)
        return self.dropout(bug_vector)
            
# 2. Cooperative Code File Processing Module
class SharedExtractor(nn.Module):
    '''
    Extracts public (transferable) features from code using CNNs. 
    This acts as the "generator" in the adversarial setup.
    Adapted for BGE
    '''
    # CNN-based shared feature extractor for public features
    def __init__(self, embedding_dim, num_filters, kernel_sizes, dropout_prob=0.5):
        super(SharedExtractor, self).__init__()
        # For processing sequences of BGE embeddings
        self.convs = nn.ModuleList([
            nn.Conv2d(in_channels=1, out_channel=num_filters, 
                      kernel_sizes=(k, embedding_dim), padding=(k//2, 0)) for k in kernel_sizes
        ])
        self.dropout = nn.Dropout(dropout_prob)
        self.output_dim = num_filters * len(kernel_sizes)

    def forward(self, embedded_code_tokens):
        '''
        Args:
            embedded_code_tokens (Tensor): Code embeddings from BGE.
                                           Shape: (batch_size, seq_len, embedding_dim)
        Returns:
            Tensor: The extracted public feature vector.
                    Shape: (batch_size, num_filters * len(kernel_sizes))
        '''
        # Add a channel dimension for Conv2D: (batch_size, 1, seq_len, embedding_dim)
        x = embedded_code_tokens.unsqueeze(1)

        # Apply convolutions and pooling
        conv_outputs = []
        for conv in self.convs:
            # Apply convolution: (batch_size, num_filters, new_seq_len, 1)
            conv_out = F.relu(conv(x))
            # Remove last dimension: (batch_size, num_filters, new_seq_len)
            conv_out = conv_out.squeeze(3)
            # Max pool over sequence: (batch_size, num_filters)
            pooled = F.max_pool1d(conv_out, kernel_size=conv_out.size(2))
            conv_outputs.append(pooled.squeeze(2))

        # Concatenate all filter outputs
        public_features = torch.cat(conv_outputs, dim=1)
        return self.dropout(public_features)
    
class IndividualExtractor(nn.Module):
    '''
    Extracts private (project-specific) features from a code file's AST using a 
    Graph Convolutional Network (GCN).
    '''
    # GCN-based individual feature extractor for private features
    def __init__(self, node_embedding_dim, hidden_dim, output_dim, num_layers=2, dropout_prob=0.5):
        super(IndividualExtractor, self).__init__()

        self.convs = nn.ModuleList()
        self.batch_norms = nn.ModuleList()

        # First layer
        self.convs.append(GCNConv(node_embedding_dim, hidden_dim))
        self.batch_norms.append(nn.BatchNorm1d(hidden_dim))
        
        # Hidden layers
        for _ in range(num_layers - 2):
            self.convs.append(GCNConv(hidden_dim, hidden_dim))
            self.batch_norms.append(nn.BatchNorm1d(hidden_dim))
        
        # Output layer
        if num_layers > 1:
            self.convs.append(GCNConv(hidden_dim, output_dim))
            self.batch_norms.append(nn.BatchNorm1d(hidden_dim))

        self.dropout = nn.Dropout(dropout_prob)
        self.output_dim = output_dim

    def forward(self, graph_data):
        '''
        Args:
            graph_data (torch_geometric.data.Data): A graph object containing:
                - x:  node features (embeddings)
                - edge_index: edge connectivity
                - batch: batch assignment for nodes 

        Returns:
            Tensor: The extracted private feature vector for each graph in the batch.
                Shape: (batch_size, output_dim)
        '''
        # implementation of GCN layers
        x, edge_index, batch = graph_data.x, graph_data.edge_index, graph_data.batch
        # Apply GCN layers with batch norm and dropout
        for i, (conv, bn) in enumerate(zip(self.convs[:-1], self.batch_norms[:-1])):
            x = self.conv(x, edge_index)
            x = bn(x)
            x = F.relu(x)
            x = self.dropout(x)
        
        # Final layer (no activation)
        if len(self.convs) > 0:
            x = self.convs[-1](x, edge_index)
            if len(self.batch_norms) >  len(self.convs) - 1:
                x = self.batch_norms[-1](x)

        # Global mean pooling to get graph-level representation
        return global_mean_pool(x, batch)
    
class ProjectDiscriminator(nn.Module):
    '''
    A discriminator to distinguish between source and target project features. Used for adversarial
    training to ensure public features are project-agnostic
    '''
    def __init__(self, input_dim, hidden_dim=128, dropout_prob=0.5):
        super(ProjectDiscriminator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_prob),
            nn.Linear(hidden_dim, hidden_dim // 2), 
            nn.BatchNorm1d(hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout_prob),
            nn.LogSoftmax(hidden_dim // 2, 2) # 2 classes: source(0) vs target(1)
        )

    def forward(self, public_features):
        '''
        Args:
            public_features: Features from the shared extractor
        Returns:
            Logits for source/target classification
        '''
        return self.model(public_features)
    
class FeatureFusion(nn.Module):
    '''
    Project-specific MLP to fuse public and private code features.
    '''
    def __init__(self, public_dim, private_dim, output_dim, hidden_dim=None, dropout_prob=0.5):
        super(FeatureFusion, self).__init__()
        input_dim = public_dim + private_dim
        if hidden_dim is None:
            hidden_dim = (input_dim + output_dim) // 2
        
        self.model = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_prob),
            nn.Linear(output_dim, output_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_prob),
            nn.Linear(output_dim, output_dim)
        )
    
    def forward(self, combined_features):
        return self.model(combined_features)

# 3. Full COOBA Model
class COOBA(nn.Module):
    '''
    The complete COOBA model, integrating all components.
    Adapted for BGE Embeddings instead of GloVe.
    '''
    def __init__(self, bug_embedding_dim, code_embedding_dim, bug_encoder_params, 
                 shared_extractor_params, individual_extractor_params, fusion_params,
                 mode='cross-project'):
        super(COOBA, self).__init__()

        self.mode = mode # 'cross-project' or 'within-project'
        self.bug_embedding_dim = bug_embedding_dim
        self.code_embedding_dim = code_embedding_dim

        # Shared components
        self.bug_report_encoder = BugReportEncoder(bug_embedding_dim, **bug_encoder_params)
        self.shared_code_extractor = SharedExtractor(code_embedding_dim, **shared_extractor_params)

        # Project-specific components
        if mode == 'cross-project':
            # Separate extractors for source and target
            self.individual_extractor_source = IndividualExtractor(
                code_embedding_dim, **individual_extractor_params
            )
            self.individual_extractor_target = IndividualExtractor(
                code_embedding_dim, ** individual_extractor_params
            )

            # Calculate fusion input dimension
            public_feat_dim = self.shared_code_extractor.output_dim
            private_feat_dim = individual_extractor_params['output_dim']

            # Separate fusion layers for source and target
            self.fusion_source = FeatureFusion(
                public_feat_dim, private_feat_dim, **fusion_params
            )
            self.fusion_target = FeatureFusion(
                public_feat_dim, private_feat_dim, **fusion_params
            )
        else:
            # Single extractor for within-project
            self.individual_extractor = IndividualExtractor(
                code_embedding_dim, **individual_extractor_params
            )
            public_feat_dim = self.shared_code_extractor.output_dim
            private_feat_dim = individual_extractor_params['output_dim']

            self.fusion = FeatureFusion(
                public_feat_dim, private_feat_dim, **fusion_params
            )
        
        # Output dimension from bug encoder
        self.bug_output_dim = bug_encoder_params['hidden_size'] * 2 # bidirectional
        self.code_output_dim = fusion_params['output_dim']

    def forward(self, bug_report_embeddings, bug_lengths, code_embeddings,
                 code_graph, project_type=None):
        '''
        Performs a forward pass of COOBA model.
        Args:
            bug_report embeddings: Pre-computed BGE embeddings for bug reports
                            Shape: (batch_size, seq_len, bug_embedding_dim)
            bug_lengths: Actual lengths of bug reports
            code_embeddings: Pre-computed BGE embeddings for code
                            Shape: (batch_size, seq_len, code_embedding_dim)
            code_graph: PyG Data object for code AST with BGE node features.
            project_type: 'source' or 'target' (required for cross-project mode)
        Returns:
            scores = Relevance scores between bugs and code
            public_features: Public features (for adversarial training)
        '''
        # 1. Process Bug Report
        bug_vector = self.bug_report_encoder(bug_report_embeddings, bug_lengths)

        # 2. Extract public features from code
        public_features = self.shared_code_extractor(code_embeddings)

        # 3. Extract private features and fuse
        if self.mode == 'cross-project':
            if project_type == 'source':
                private_features = self.individual_extractor_source(code_graph)
                combined = torch.cat([public_features, private_features], dim=1)
                code_vector = self.fusion_target(combined)
            elif project_type == 'target':
                private_features = self.individual_extractor_source(code_graph)
                combined = torch.cat([public_features, private_features], dim=1)
                code_vector = self.fusion_target(combined)
            else:
                raise ValueError(f"project_type must be 'source' or 'target', got {project_type}")
        else:
            # Within-project mode
            private_features = self.individual_extractor(code_graph)
            combined = torch.cat([public_features, private_features], dim=1)
            code_vector = self.fusion(combined)

        # 4. Compute relevance score
        # Ensure dimensions match for similarity computation
        if bug_vector.shape[1] != code_vector.shape[1]:
            # Project to common dimension if needed
            min_dim = min(bug_vector.shape[1], code_vector.shape[1])
            if not hasattr(self, 'bug_projection'):
                self.bug_projection = nn.Linear(bug_vector.shape[1], min_dim).to(bug_vector.device)
                self.code_projection = nn.Linear(code_vector.shape[1], min_dim).to(code_vector.device)
            bug_vector = self.bug_projection(bug_vector)
            code_vector = self.code_projection(code_vector)
        
        # Compute cosine similarity
        scores = F.cosine_similarity(bug_vector, code_vector, dim=1)

        return scores, public_features