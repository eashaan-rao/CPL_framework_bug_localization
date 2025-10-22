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
        # This embedding layer will be handled outside this class, as it's shared with the code processing module
        
        self.lstm = nn.LSTM(self, embedding_dim, hidden_size, num_layers, bidirectional=True, 
                            batch_first=True, dropout=dropout_prob if num_layers > 1 else 0)

    def forward(self, embedded_sequence, lengths):
        '''
        Args: 
        embedded_sequence (Tensor): Batch of padded embedded bug reports
                                    Shape: (batch_size, seq_len, embedding_dim)
        lengths (Tensor): A Single vector representation for each bug report.
                          Shape: (batch_size, hidden_size * 2)
        '''
        # Pack padded sequence to handle variable lenghts
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

        return torch.cat((forward_hidden, backward_hidden), dim=1)
            
# 2. Cooperative Code File Processing Module
class SharedExtractor(nn.Module):
    '''
    Extracts public (transferable) features from code using CNNs. This acts as the "generator" in the adversarial
    setup.
    '''
    # CNN-based shared feature extractor for public features
    def __init__(self, embedding_dim, num_filters, kernel_sizes):
        super(SharedExtractor, self).__init__()
        self.convs = nn.ModuleList([
            nn.Conv2d(in_channels=1, out_channel=num_filters, kernel_sizes=(k, embedding_dim)) for k in kernel_sizes
        ])

    def forward(self, embedded_code_tokens):
        '''
        Args:
            embedded_code_tokens (Tensor): A matrix of token embeddings for a code file.
                                           Shape: (batch_size, num_tokens, embedding_dim)
        Returns:
            Tensor: The extracted public feature vector.
                    Shape: (batch_size, num_filters * len(kernel_sizes))
        '''
        # Add a channel dimension for Conv2D: (batch_size, 1, num_tokens, embedding_dim)
        x = embedded_code_tokens.unsqueeze(1)

        # Apply convolutions and pooling
        x = [F.relu(conv(x)).squeeze(3) for conv in self.convs]
        # x is a list of tensors of shape (batch_size, num_filters, num_tokens - k + 1)

        x= [F.max_pool1d(line, line.size(2)).squeeze(2) for line in x]
        # x is a list of tensors of shape (batch_size, num_filters)

        return torch.cat(x, 1)
    
class IndividualExtractor(nn.Module):
    '''
    Extracts private (project-specific) features from a code file's AST using a 
    Graph Convolutional Network (GCN).
    '''
    # GCN-based individual feature extractor for private features
    def __init__(self, node_feature_dim, hidden_dim, output_dim, dropout_prob=0.5):
        super(IndividualExtractor, self).__init__()
        self.conv1 = GCNConv(node_feature_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, output_dim)
        self.dropout = nn.Dropout(dropout_prob)

    def forward(self, graph_data: Data):
        '''
        Args:
         graph_data (torch_geometric.data.Data): A graph object containing x (node features), edge_index, 
         and batch index.

        Returns:
        Tensor: The extracted private feature vector for each graph in the batch.
                Shape: (batch_size, output_dim)
        '''
        # implementation of GCN layers
        x, edge_index = graph_data.x, graph_data.edge_index
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.conv2(x, edge_index)

        # Global mean polling to get a single vector for the entire graph
        # Note: This requires torch-scatter. If not available, another pooling can be used.
        return global_mean_pool(x, graph_data.batch)
    
class ProjectDiscriminator(nn.Module):
    '''
    A simple MLP to discriminate the origin of public features. Tries to guess if the code is from the
    source or target project.
    '''
    # Simple MLP to distinguish projects
    def __init__(self, input_dim, hidden_dim=128):
        super(ProjectDiscriminator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim,2), # 2 classes: for source(0) vs target(1)
            nn.LogSoftmax(dim=1)
        )
    def forward(self, public_features):
        return self.model(public_features)
    
class FeatureFusion(nn.Module):
    '''
    Project-specific MLP to fuse public and private code features.
    '''
    def __init__(self, input_dim, output_dim):
        super(FeatureFusion, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, output_dim),
            nn.ReLU(),
            nn.Linear(output_dim, output_dim),
            nn.ReLU(),
        )
    
    def forward(self, combined_features):
        return self.model(combined_features)


# 3. Full COOBA Model
class COOBA(nn.Module):
    '''
    The complete COOBA model, integrating all components.
    '''
    def __init__(self, embedding_matrix, bug_encoder_params, shared_extractor_params,
                 individual_extractor_params, fusion_params):
        super(COOBA, self).__init__()

        vocab_size, embedding_dim = embedding_matrix.shape
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.embedding.weight.data.copy_(torch.from_numpy(embedding_matrix))
        self.embedding.weight.requires_grad = False # Freeze embeddings as per std practice

        # Shared components
        self.bug_report_encoder = BugReportEncoder(embedding_dim, **bug_encoder_params)
        self.shared_code_extractor = SharedExtractor(embedding_dim, **shared_extractor_params)

        # Project-specific components
        # We need two separate instances for source and target projects
        self.individual_extractor_source = IndividualExtractor(embedding_dim, **individual_extractor_params)
        self.individual_extractor_target = IndividualExtractor(embedding_dim, **individual_extractor_params)

        # Calculate fusion input dimension
        public_feat_dim = shared_extractor_params['num_filters'] * len(shared_extractor_params['kernel_sizes'])
        private_feat_dim = individual_extractor_params['output_dim']
        fusion_input_dim = public_feat_dim + private_feat_dim

        self.fusion_source = FeatureFusion(fusion_input_dim, **fusion_params)
        self.fusion_target = FeatureFusion(fusion_input_dim, **fusion_params)

    def forward(self, bug_report, code_file_graph, project_type):
        '''
        Performs a forward pass to get the relevance score.
        Args:
            bug_report (tuple): A tuple containing padded token IDs and lengths.
            code_file_graph (Data): A PyG Data object for the code file's AST.
            project_type (str): Either 'source' or 'target'
        Returns:
            Tensor: The relevance score between the bug report and code file.
        '''
        # 1. Process Bug Report
        bug_tokens, bug_lengths = bug_report
        bug_embedded = self.embedding(bug_tokens)
        bug_vector = self.bug_report_encoder(bug_embedded, bug_lengths)

        # 2. Process Code File
        # Embed the node tokens of the graph
        code_file_graph.x = self.embedding(code_file_graph.x)

        # a) Extract public features.
        # Note: The CNN expects a sequence. We can use the node features in order
        public_features = self.shared_code_extractor(code_file_graph.x.unsqueeze(0))

        # b) Extract private features and fuse
        if project_type == 'source':
            private_features = self.individual_extractor_source(code_file_graph)
            combined = torch.cat((public_features, private_features), dim=1)
            code_vector = self.fusion_source(combined)
        elif project_type == 'target':
            private_features = self.individual_extractor_target(code_file_graph)
            combined = torch.cat((public_features, private_features), dim=1)
            code_vector = self.fusion_target(combined)
        else:
            raise ValueError("project_type must be 'source' or 'target'")
        
        # 3. Relevance Prediction (using cosine similarity for stability)
        score = F.cosine_similarity(bug_vector, code_vector, dim=1)

        return score