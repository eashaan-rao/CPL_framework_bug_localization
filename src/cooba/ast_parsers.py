import ast
import javalang
from torch_geometric.data import Data
import torch
import numpy as np
import re

# Constants for special tokens (compatibility)
PAD_TOKEN = '<PAD>'
UNK_TOKEN = '<UNK>'
GLOVE_DIM = 300  # matches utils.GLOVE_DIM

class BaseASTParser:
    '''
    Base class for AST parsers.
    Uses GloVe word embeddings as per-node features (faithful to COOBA paper).
    '''
    def __init__(self, max_nodes=None, glove_dict=None):
        '''
        Args:
            max_nodes (int|None): Maximum nodes to keep. None = no cap (true AST).
            glove_dict (dict): GloVe {word: np.array(300)} lookup, set after init
        '''
        self.max_nodes = max_nodes  # None means no truncation — true Python AST
        self.glove_dict = glove_dict  # injected by pipeline after GloVe is loaded
        self.node_counter = 0

    def get_tokens(self, code_string):
        '''
        Extract tokens from code for compatibility with existing code.

        Args:
            code_string (str): Source code
        Returns:
            list: List of tokens
        '''
        # Simple tokenization by splitting on whitespace and punctuation
        tokens = re.findall(r'\w+', code_string)
        return tokens
    
    def parse(self, code_string):
        '''
        Parses a code string into a graph structure.  Muse be implemented by subclass.

        Args:
            code_string (str): Source code to parse
        Returns:
            torch_geometric.data.Data: Graph representation of AST
        '''
        raise NotImplementedError
    
    def _build_graph(self, root_node, traverse_func):
        '''
        Helper function to build a graph from a traversed AST.
        Args:
            root_node: Root of the AST
            traverse_func: Function to traverse the AST
        Returns:
            torch_geometric.data.Data: Graph data object
        '''
        self.node_counter = 0
        nodes = []
        edges = []

        # Traverse the AST
        traverse_func(root_node, nodes, edges)

        # Limit number of nodes only if max_nodes is set
        if self.max_nodes is not None and len(nodes) > self.max_nodes:
            nodes = nodes[:self.max_nodes]
            edges = [e for e in edges if e['source'] < self.max_nodes and e['target'] < self.max_nodes]
        
        num_nodes = len(nodes)
        if num_nodes == 0:
            return Data(
                x=torch.zeros(1, GLOVE_DIM),
                edge_index=torch.tensor([[], []], dtype=torch.long)
            )

        # Build edge index tensor
        edge_index = [[], []]
        for edge in edges:
            edge_index[0].append(edge['source'])
            edge_index[1].append(edge['target'])

        # Per-node GloVe features: each node gets the GloVe embedding of its token
        # (different for every node — faithful to COOBA paper's GloVe usage)
        if self.glove_dict is not None:
            from .utils import glove_lookup
            node_features = torch.stack([
                torch.tensor(glove_lookup(node['token'], self.glove_dict), dtype=torch.float32)
                for node in nodes
            ])
        else:
            node_features = torch.zeros(num_nodes, GLOVE_DIM)

        return Data(
            x=node_features,
            edge_index=torch.tensor(edge_index, dtype=torch.long)
        )

class PythonASTParser(BaseASTParser):
    '''
    Parses Python code into an AST graph using the 'ast' module.
    '''    
    def parse(self, code_string):
        '''
        Parses a Python code string.
        
        Args:
            code_string (str): Python source code
        Returns:
            torch_geometric.data.Data: Graph representation of AST
        '''
        try:
            # Remove any leading/trailing whitespace
            code_string = code_string.strip()

            # Try to parse the code
            tree = ast.parse(code_string)
            return self._build_graph(tree, self._traverse_python_ast)
        
        except SyntaxError:
            # Try to parse an expression if statement parsing fails
            try:
                tree = ast.parse(code_string, mode='eval')
                return self._build_graph(tree, self._traverse_python_ast)
            except:
                pass
        
        except Exception as e:
            pass

        # Return minimal graph on failure
        return Data(
            x=torch.zeros(1, GLOVE_DIM),
            edge_index=torch.tensor([[], []], dtype=torch.long)
        )

    def _traverse_python_ast(self, node, nodes, edges, parent_id=None):
        '''
        Recursively traverses the Python AST.
        Args:
            node: Current AST node
            nodes: List to store node information
            edges: List to store edge information
            parent_id: ID of parent node
        '''
        if self.max_nodes is not None and self.node_counter >= self.max_nodes:
            return

        current_id = self.node_counter
        self.node_counter += 1

        # Get node type and value
        node_type = type(node).__name__
        node_token = node_type

        # Extract more specific information for certain node types
        if hasattr(node, 'id'):
            node_token = f"{node_type}:{node.id}"
        elif hasattr(node, 'name'):
            node_token = f"{node_type}:{node.name}"
        elif hasattr(node, 'value') and isinstance(node.value, (str, int, float)):
            node_token = f"{node_type}:{str(node.value)[:20]}"  # Convert first, then limit length
        
        nodes.append({'id':current_id, 'token': node_token, 'type': node_type})

        if parent_id is not None:
            edges.append({'source': parent_id, 'target': current_id})

        # Traverse children
        for child in ast.iter_child_nodes(node):
            if self.max_nodes is None or self.node_counter < self.max_nodes:
                self._traverse_python_ast(child, nodes, edges, parent_id=current_id)

class JavaASTParser(BaseASTParser):
    '''
    Parses Java code into an AST graph using the 'javalang' library.
    '''
    def parse(self, code_string):
        '''
        Parses a Java code string.
        Args:
            code_string (str): Java source code
        Returns:
            torch_geometric.data.Data: Graph representation of AST
        '''
        try:
            # Try to parse as complete compilation unit
            tree = javalang.parse.parse(code_string)
            return self._build_graph(tree, self._traverse_java_ast)
        
        except javalang.parser.JavaSyntaxError:
            # Try to parse as method as statement
            try:
                # Wrap in a simple class structure
                wrapped_code = f"""
                public class TempClass {{
                    {code_string}
                }}
                """
                tree = javalang.parse.parse(wrapped_code)
                return self._build_graph(tree, self._traverse_java_ast)
            except:
                pass

        except javalang.tokenizer.LexerError:
            pass
        except Exception as e:
            pass 

        # Return minimal graph on failure
        return Data(
            x=torch.zeros(1, GLOVE_DIM),
            edge_index=torch.tensor([[], []], dtype=torch.long)
        )

    def _get_java_node_token(self, node):
        '''
        Extracts a meaningful token from a javalang node.
        Args:
            node: Javalang AST node
        Returns:
            str: Token representation of the node
        '''
        if node is None:
            return UNK_TOKEN
        
        node_type = type(node).__name__

        # Extract specific information based on node type
        if hasattr(node, 'name') and node.name:
            return f"{node_type}:{node.name}"
        elif hasattr(node, 'value') and node.value:
            value_str = str(node.value)[:20] # Limit length
            return f"{node_type}:{value_str}"
        elif hasattr(node, 'member') and node.member:
            return f"{node_type}:{node.member}"
        elif hasattr(node, 'type') and hasattr(node.type, 'name'):
            return f"{node_type}:{node.type.name}"

        return node_type 
    
    def _traverse_java_ast(self, node, nodes, edges, parent_id=None):
        '''
        Recursively traverses the Java AST.
        Args:
            node: Current AST node
            nodes: List to store node information
            edges: List to store edge information
            parent_id: ID of parent node
        '''
        if node is None or (self.max_nodes is not None and self.node_counter >= self.max_nodes):
            return
        
        current_id = self.node_counter
        self.node_counter += 1

        node_token = self._get_java_node_token(node)
        node_type = type(node).__name__

        nodes.append({'id':current_id, 'token':node_token, 'type': node_type})

        if parent_id is not None:
            edges.append({'source':parent_id, 'target':current_id})
        
        # Traverse children based on node type
        if isinstance(node, (javalang.ast.Node)):
            # Get all attributes that might contain child nodes
            for attr_name in node.attrs:
                attr_value = getattr(node, attr_name, None)
                
                if attr_value is None:
                    continue
                
                # Handle different types of attributes
                if isinstance(attr_value, list):
                    for item in attr_value:
                        if self.max_nodes is None or self.node_counter < self.max_nodes:
                            self._traverse_java_ast(item, nodes, edges, parent_id=current_id)
                elif isinstance(attr_value, javalang.ast.Node):
                    if self.max_nodes is None or self.node_counter < self.max_nodes:
                        self._traverse_java_ast(attr_value, nodes, edges, parent_id=current_id)
        elif isinstance(node, list):
            for item in node:
                self._traverse_java_ast(item, nodes, edges, parent_id=current_id)

def get_parser_for_language(language):
    '''
    Get the appropriate AST parser for a given programming language.
    Args:
        language(str): Programming language name
    Returns:
        BaseASTParser: AST parser instance
    '''
    language = language.lower()

    parsers = {
        'python': PythonASTParser,
        'java': JavaASTParser
    }
    parser_class = parsers.get(language, BaseASTParser)
    return parser_class()