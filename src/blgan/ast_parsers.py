"""
AST parsers for BL-GAN bug localization.

Parses Python and Java source code into graph structures where node features
are stored as raw string tokens (not BGE embeddings). A separate
encode_ast_with_vocab() function converts these raw graphs to integer-ID
tensors using a learned vocabulary — keeping the cached graphs vocab-independent.
"""

import ast
import re
from typing import Dict, List, Optional

import javalang
import torch
from torch_geometric.data import Data

# Special tokens
PAD_TOKEN = '<PAD>'
UNK_TOKEN = '<UNK>'


# ---------------------------------------------------------------------------
# Vocabulary-based encoding
# ---------------------------------------------------------------------------

def encode_ast_with_vocab(raw_graph: Data, vocab: dict, max_nodes: int) -> Data:
    """Convert a raw AST graph (with string node_tokens) into a vocab-indexed Data.

    Args:
        raw_graph : Data object with .node_tokens (List[str]) and .edge_index.
        vocab     : {token_str: int_id} with '<UNK>' → 1.
        max_nodes : Maximum number of nodes to retain (truncates if needed).

    Returns:
        Data(x=LongTensor(N,), edge_index=LongTensor(2, E))
        where x contains vocab IDs for each node token.
        If the graph is empty, returns a single UNK node with no edges.
    """
    node_tokens: List[str] = getattr(raw_graph, 'node_tokens', [])
    edge_index: torch.Tensor = raw_graph.edge_index

    if not node_tokens:
        return Data(
            x=torch.tensor([vocab.get(UNK_TOKEN, 1)], dtype=torch.long),
            edge_index=torch.tensor([[], []], dtype=torch.long),
        )

    # Truncate nodes
    if len(node_tokens) > max_nodes:
        node_tokens = node_tokens[:max_nodes]
        # Keep only edges whose both endpoints survive truncation
        if edge_index.numel() > 0:
            mask = (edge_index[0] < max_nodes) & (edge_index[1] < max_nodes)
            edge_index = edge_index[:, mask]

    unk_id = vocab.get(UNK_TOKEN, 1)
    token_ids = [vocab.get(t, unk_id) for t in node_tokens]

    return Data(
        x=torch.tensor(token_ids, dtype=torch.long),
        edge_index=edge_index,
    )


# ---------------------------------------------------------------------------
# Base parser
# ---------------------------------------------------------------------------

class BaseASTParser:
    """Base class for BL-GAN AST parsers.

    Stores raw node tokens as strings. Vocabulary encoding is performed
    separately via encode_ast_with_vocab().
    """

    def __init__(self, max_nodes: int = 500):
        self.max_nodes = max_nodes
        self.node_counter = 0

    def get_tokens(self, code_string: str) -> List[str]:
        """Simple word-level tokenisation of a code string."""
        return re.findall(r'\w+', code_string)

    def parse(self, code_string: str) -> Data:
        """Parse source code into a raw graph with string node tokens.

        Must be overridden by subclasses.

        Returns:
            Data(node_tokens=List[str], edge_index=LongTensor(2, E))
        """
        raise NotImplementedError

    def _build_graph(self, root_node, traverse_func) -> Data:
        """Build a raw AST graph by calling traverse_func on root_node.

        Node features are stored as a list of string tokens in
        Data.node_tokens.  Vocabulary encoding is deferred to
        encode_ast_with_vocab().

        Args:
            root_node    : Root of the AST.
            traverse_func: Callable(root_node, nodes, edges) that fills
                           nodes (list of {'id', 'token'}) and
                           edges (list of {'source', 'target'}).

        Returns:
            Data(node_tokens=List[str], edge_index=LongTensor(2, E))
        """
        self.node_counter = 0
        nodes: list = []
        edges: list = []

        traverse_func(root_node, nodes, edges)

        # Truncate nodes
        if len(nodes) > self.max_nodes:
            nodes = nodes[:self.max_nodes]
            edges = [
                e for e in edges
                if e['source'] < self.max_nodes and e['target'] < self.max_nodes
            ]

        if not nodes:
            return Data(
                node_tokens=[],
                edge_index=torch.tensor([[], []], dtype=torch.long),
            )

        node_tokens = [n['token'] for n in nodes]

        src_ids = [e['source'] for e in edges]
        tgt_ids = [e['target'] for e in edges]
        edge_index = torch.tensor([src_ids, tgt_ids], dtype=torch.long)

        return Data(node_tokens=node_tokens, edge_index=edge_index)


# ---------------------------------------------------------------------------
# Python AST parser
# ---------------------------------------------------------------------------

class PythonASTParser(BaseASTParser):
    """Parses Python source code into a raw BL-GAN AST graph."""

    def parse(self, code_string: str) -> Data:
        """Parse Python source code.

        Args:
            code_string: Python source code.
        Returns:
            Data(node_tokens=List[str], edge_index=LongTensor(2, E))
        """
        try:
            code_string = code_string.strip()
            tree = ast.parse(code_string)
            return self._build_graph(tree, self._traverse_python_ast)
        except SyntaxError:
            try:
                tree = ast.parse(code_string, mode='eval')
                return self._build_graph(tree, self._traverse_python_ast)
            except Exception:
                pass
        except Exception:
            pass

        return Data(
            node_tokens=[],
            edge_index=torch.tensor([[], []], dtype=torch.long),
        )

    def _traverse_python_ast(self, node, nodes: list, edges: list, parent_id: Optional[int] = None):
        """Recursively traverse a Python AST and collect node tokens and edges."""
        if self.node_counter >= self.max_nodes:
            return

        current_id = self.node_counter
        self.node_counter += 1

        node_type = type(node).__name__
        node_token = node_type

        # Enrich token with identifier/name/value when available
        if hasattr(node, 'id'):
            node_token = f"{node_type}:{node.id}"
        elif hasattr(node, 'name'):
            node_token = f"{node_type}:{node.name}"
        elif hasattr(node, 'value') and isinstance(node.value, (str, int, float)):
            val_str = str(node.value)[:20]
            node_token = f"{node_type}:{val_str}"

        nodes.append({'id': current_id, 'token': node_token})

        if parent_id is not None:
            edges.append({'source': parent_id, 'target': current_id})

        for child in ast.iter_child_nodes(node):
            if self.node_counter < self.max_nodes:
                self._traverse_python_ast(child, nodes, edges, parent_id=current_id)


# ---------------------------------------------------------------------------
# Java AST parser
# ---------------------------------------------------------------------------

class JavaASTParser(BaseASTParser):
    """Parses Java source code into a raw BL-GAN AST graph."""

    def parse(self, code_string: str) -> Data:
        """Parse Java source code.

        Args:
            code_string: Java source code.
        Returns:
            Data(node_tokens=List[str], edge_index=LongTensor(2, E))
        """
        try:
            tree = javalang.parse.parse(code_string)
            return self._build_graph(tree, self._traverse_java_ast)
        except javalang.parser.JavaSyntaxError:
            try:
                wrapped = f"public class TempClass {{ {code_string} }}"
                tree = javalang.parse.parse(wrapped)
                return self._build_graph(tree, self._traverse_java_ast)
            except Exception:
                pass
        except javalang.tokenizer.LexerError:
            pass
        except Exception:
            pass

        return Data(
            node_tokens=[],
            edge_index=torch.tensor([[], []], dtype=torch.long),
        )

    def _get_java_node_token(self, node) -> str:
        """Extract a meaningful string token from a javalang AST node."""
        if node is None:
            return UNK_TOKEN

        node_type = type(node).__name__

        if hasattr(node, 'name') and node.name:
            return f"{node_type}:{node.name}"
        elif hasattr(node, 'value') and node.value:
            return f"{node_type}:{str(node.value)[:20]}"
        elif hasattr(node, 'member') and node.member:
            return f"{node_type}:{node.member}"
        elif hasattr(node, 'type') and hasattr(node.type, 'name'):
            return f"{node_type}:{node.type.name}"

        return node_type

    def _traverse_java_ast(self, node, nodes: list, edges: list, parent_id: Optional[int] = None):
        """Recursively traverse a Java AST and collect node tokens and edges."""
        if node is None or self.node_counter >= self.max_nodes:
            return

        current_id = self.node_counter
        self.node_counter += 1

        node_token = self._get_java_node_token(node)
        nodes.append({'id': current_id, 'token': node_token})

        if parent_id is not None:
            edges.append({'source': parent_id, 'target': current_id})

        if isinstance(node, javalang.ast.Node):
            for attr_name in node.attrs:
                attr_value = getattr(node, attr_name, None)
                if attr_value is None:
                    continue
                if isinstance(attr_value, list):
                    for item in attr_value:
                        if self.node_counter < self.max_nodes:
                            self._traverse_java_ast(item, nodes, edges, parent_id=current_id)
                elif isinstance(attr_value, javalang.ast.Node):
                    if self.node_counter < self.max_nodes:
                        self._traverse_java_ast(attr_value, nodes, edges, parent_id=current_id)
        elif isinstance(node, list):
            for item in node:
                if self.node_counter < self.max_nodes:
                    self._traverse_java_ast(item, nodes, edges, parent_id=current_id)


# ---------------------------------------------------------------------------
# Language dispatch
# ---------------------------------------------------------------------------

def get_parser_for_language(language: str) -> BaseASTParser:
    """Return the BL-GAN AST parser for the given programming language.

    Args:
        language: Programming language name (case-insensitive).
    Returns:
        Appropriate BaseASTParser subclass instance.
    """
    parsers = {
        'python': PythonASTParser,
        'java': JavaASTParser,
    }
    parser_class = parsers.get(language.lower(), BaseASTParser)
    return parser_class()
