import ast
import javalang
from torch_geometric.data import Data
import torch

class BaseASTParser:
    '''
    Base class for AST parsers.
    '''
    def __init__(self, vocabulary):
        '''
        Args: vocabulary(dict)L A dictionary mapping tokens to integer IDs.
        '''
        self.vocabulary = vocabulary
        self.node_counter = 0

    def get_token_id(self, token):
        '''
        Gets the integer ID for a token, return unk_id if not found.
        '''
        return self.vocabulary.get(token, self.vocabulary.get('<UNK>', 0))
    
    def parse(self, code_string):
        '''
        Parses a code string into a graph structure.  Muse be implemented by subclass.
        '''
        raise NotImplementedError
    
    def _build_graph(self, root_node, traverse_func):
        '''
        Helper function to build a graph from a traversed AST.
        '''
        self.node_counter = 0
        nodes = []
        edges = []

        traverse_func(root_node, nodes, edges)

        # Convert to PyG Data format
        node_features = [self.get_token_id(node['token']) for node in nodes]

        # Create edge index
        edge_index = [[], []]
        for edge in edges:
            edge_index[0].append(edge['source'])
            edge_index[1].append(edge['target'])

        
        return Data(x=torch.tensor(node_features, dtype=torch.long),
                    edge_index=torch.tensor(edge_index, dtype=torch.long))

class PythonASTParser(BaseASTParser):
    '''
    Parses Python code into an AST graph using the 'ast' module.
    '''    
    def parse(self, code_string):
        '''
        Parses a Python code string.
        '''
        try:
            tree = ast.parse(code_string)
            return self._build_graph(tree, self._traverse_python_ast)
        except SyntaxError:
            return None # Handle parsing errors gracefully
        
    def _traverse_python_ast(self, node, nodes, edges, parent_id=None):
        '''
        Recursively traverses the Python AST.
        '''
        current_id = self.node_counter
        self.node_counter += 1

        # Use the class name of the node as its token
        node_token = type(node).__name__
        nodes.append({'id':current_id, 'token':node_token})

        if parent_id is not None:
            edges.append({'source':parent_id, 'target':current_id})

        for child in ast.iter_child_nodes(node):
            self._traverse_python_ast(child, nodes, edges, parent_id=current_id)

class JavaASTParser(BaseASTParser):
    '''
    Parses Java code into an AST graph using the 'javalang' library.
    '''
    def parse(self, code_string):
        '''
        Parses a Java code string.
        '''
        try:
            tree = javalang.parse.parse(code_string)
            # The root is typically a CompilationUnit
            return self._build_graph(tree, self._traverse_java_ast)
        except javalang.tokenizer.LexerError:
            return None # Handle parsing errors
        
    def _get_java_node_token(self, node):
        '''
        Extracts a meaningful token from a javalang node.
        '''
        token = type(node).__name__
        # For some nodes, we might want more specific info
        if hasattr(node, 'name') and node.name:
            token = node.name
        elif hasattr(node, 'value') and node.value:
            token = node.value
        return str(token) # Ensure it's a string
    
    def _traverse_java_ast(self, node, nodes, edges, parent_id=None):
        '''
        Recursively traverses the Java AST.
        'javalang' nodes are tuples or objects with attributes.
        '''
        if node is None:
            return
        
        current_id = self.node_counter
        self.node_counter += 1

        node_token = self._get_java_node_token(node)
        nodes.append({'id':current_id, 'token':node_token})

        if parent_id is not None:
            edges.append({'source':parent_id, 'target':current_id})
        
        # Recursively traverse children
        if isinstance(node, (javalang.ast.Node, javalang.tokenizer.Token)):
            for child in node:
                self._traverse_java_ast(child, nodes, edges, parent_id=current_id)
        elif isinstance(node, list):
            for item in node:
                self._traverse_java_ast(item, nodes, edges, parent_id=current_id)