#!/usr/bin/env python3
"""
Compare PII Detection Rules and Calculate IoU

This script processes JSONL attribute files from two different folders,
applies different rules to each for PII detection, and calculates the
Intersection over Union (IoU) to measure how well they overlap.

Example usage:
python pii_rule_comparison.py \
    --ref-folder s3://bucket/workspace/attributes/model_a \
    --hyp-folder s3://bucket/workspace/attributes/model_b \
    --ref-rule "gpt_4_1_contains_pii:any" \
    --hyp-rule "gpt_4_1_contains_email_addresses:any" \
    --output-file iou_results.json
"""

import argparse
import boto3
import gzip
import json
import logging
import os
import re
import sys
from collections import defaultdict
from enum import Enum, auto
from pathlib import Path
from typing import Dict, List, Set, Tuple, Union, Any, Callable
import zstandard as zstd

# Initialize logger
logging.basicConfig(
    level=logging.INFO, 
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Define token types for the rule expression parser
class TokenType(Enum):
    RULE = auto()
    AND = auto()
    OR = auto()
    NOT = auto()
    LPAREN = auto()
    RPAREN = auto()
    EOF = auto()

class Token:
    """Token for rule expression parsing"""
    def __init__(self, type, value=None):
        self.type = type
        self.value = value
        
    def __repr__(self):
        if self.value:
            return f"Token({self.type}, {self.value})"
        return f"Token({self.type})"

class ExpressionNode:
    """Base class for expression tree nodes"""
    pass

class RuleNode(ExpressionNode):
    """Leaf node representing a single rule"""
    def __init__(self, attribute_name, rule_type):
        self.attribute_name = attribute_name
        self.rule_type = rule_type
        
    def __repr__(self):
        return f"Rule({self.attribute_name}:{self.rule_type})"

class NotNode(ExpressionNode):
    """Unary NOT operation node"""
    def __init__(self, operand):
        self.operand = operand
        
    def __repr__(self):
        return f"NOT({self.operand})"

class BinaryNode(ExpressionNode):
    """Binary operation (AND/OR) node"""
    def __init__(self, left, right, operator):
        self.left = left
        self.right = right
        self.operator = operator
        
    def __repr__(self):
        return f"{self.operator}({self.left}, {self.right})"

def parse_args():
    parser = argparse.ArgumentParser(description="Compare PII detection rules and calculate IoU")
    parser.add_argument("--ref-folder", required=True, help="Reference attribute folder path (local or s3://)")
    parser.add_argument("--hyp-folder", required=True, help="Hypothesis attribute folder path (local or s3://)")
    parser.add_argument("--ref-rule", required=True, 
                        help="""Reference rule expression. Can be a simple rule in format 'attribute_name:rule_type',
                        where rule_type is 'any' or 'all'. Or a boolean expression like
                        'not rule1:any and rule2:all' or '(rule1:any or rule2:any) and not rule3:all'""")
    parser.add_argument("--hyp-rule", required=True, 
                        help="""Hypothesis rule expression. Can be a simple rule in format 'attribute_name:rule_type',
                        where rule_type is 'any' or 'all'. Or a boolean expression like
                        'not rule1:any and rule2:all' or '(rule1:any or rule2:any) and not rule3:all'""")
    parser.add_argument("--output-file", default="iou_results.json", help="Output JSON file to save results")
    parser.add_argument("--aws-profile", help="AWS profile for S3 access")
    parser.add_argument("--recursive", action="store_true", help="Recursively process folder structure")
    return parser.parse_args()

def parse_s3_path(s3_path):
    """Parse S3 path into bucket and prefix."""
    parts = s3_path.replace("s3://", "").split("/", 1)
    bucket = parts[0]
    prefix = parts[1] if len(parts) > 1 else ""
    return bucket, prefix

def get_s3_bytes(s3_client, s3_path):
    """Get bytes from S3 object."""
    bucket, key = parse_s3_path(s3_path)
    response = s3_client.get_object(Bucket=bucket, Key=key)
    return response["Body"].read()

def list_jsonl_files(path, s3_client=None, recursive=False):
    """List all JSONL files in the given path, locally or in S3."""
    jsonl_files = []
    
    if path.startswith("s3://"):
        bucket, prefix = parse_s3_path(path)
        prefix = prefix.rstrip("/") + "/"
        
        # List objects in S3 bucket with given prefix
        paginator = s3_client.get_paginator("list_objects_v2")
        for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
            if "Contents" in page:
                for obj in page["Contents"]:
                    key = obj["Key"]
                    if (key.endswith(".jsonl") or key.endswith(".json") or 
                        key.endswith(".jsonl.gz") or key.endswith(".jsonl.zst") or 
                        key.endswith(".jsonl.ztd") or key.endswith(".jsonl.zstd")):
                        jsonl_files.append(f"s3://{bucket}/{key}")
    else:
        # Local file system
        path_obj = Path(path)
        if recursive:
            for file_path in path_obj.rglob("*"):
                if (file_path.name.endswith(".jsonl") or file_path.name.endswith(".json") or
                    file_path.name.endswith(".jsonl.gz") or file_path.name.endswith(".jsonl.zst") or
                    file_path.name.endswith(".jsonl.ztd") or file_path.name.endswith(".jsonl.zstd")):
                    jsonl_files.append(str(file_path))
        else:
            for file_path in path_obj.glob("*"):
                if (file_path.name.endswith(".jsonl") or file_path.name.endswith(".json") or
                    file_path.name.endswith(".jsonl.gz") or file_path.name.endswith(".jsonl.zst") or
                    file_path.name.endswith(".jsonl.ztd") or file_path.name.endswith(".jsonl.zstd")):
                    jsonl_files.append(str(file_path))
    
    return jsonl_files

def load_jsonl_file(file_path, s3_client=None):
    """Load and decompress a JSONL file, either from local or S3."""
    try:
        # Get file content
        if file_path.startswith("s3://"):
            if s3_client is None:
                raise ValueError("S3 client is required for S3 paths")
            raw_data = get_s3_bytes(s3_client, file_path)
        else:
            with open(file_path, "rb") as f:
                raw_data = f.read()
        
        # Decompress if needed
        if file_path.endswith(".gz"):
            decompressed = gzip.decompress(raw_data)
        elif file_path.endswith((".zst", ".ztd", ".zstd")):
            dctx = zstd.ZstdDecompressor()
            decompressed = dctx.decompress(raw_data)
        else:
            decompressed = raw_data
        
        # Parse JSON lines
        lines = decompressed.decode("utf-8").strip().split("\n")
        return [json.loads(line) for line in lines if line.strip()]
    
    except Exception as e:
        logger.error(f"Error loading file {file_path}: {e}")
        return []

def apply_rule(doc, rule):
    """
    Apply a rule to determine if a document meets the PII criteria.
    
    Args:
        doc: The document JSON object
        rule: Either a tuple (attribute_name, rule_type) for simple rules,
              or an ExpressionNode for complex boolean expressions
        
    Returns:
        True if the document matches the rule, False otherwise
    """
    # Handle simple rule
    if not is_complex_expression(rule):
        return apply_simple_rule(doc, rule[0], rule[1])
    
    # Handle complex expression
    return evaluate_expression(doc, rule)

def apply_simple_rule(doc, attribute_name, rule_type):
    """
    Apply a simple rule to determine if a document meets the PII criteria.
    
    Args:
        doc: The document JSON object
        attribute_name: The attribute field to check (e.g., "gpt_4_1_contains_pii")
        rule_type: 'any' for any true value, 'all' for all true values
        
    Returns:
        True if the document matches the rule, False otherwise
    """
    if "attributes" not in doc or not doc["attributes"]:
        return False
    
    attributes = doc["attributes"]
    if attribute_name not in attributes or not attributes[attribute_name]:
        return False
    
    # Extract the boolean values from the attribute spans
    # Each span is formatted as [start_pos, end_pos, value]
    values = [span[2] for span in attributes[attribute_name] if len(span) >= 3 and span[2] is not None]
    
    if not values:
        return False
    
    if rule_type == "any":
        return any(values)
    elif rule_type == "all":
        return all(values)
    else:
        raise ValueError(f"Unknown rule type: {rule_type}")

def evaluate_expression(doc, expr):
    """
    Evaluate a boolean expression on a document.
    
    Args:
        doc: The document JSON object
        expr: An ExpressionNode representing a boolean expression
        
    Returns:
        True if the document matches the expression, False otherwise
    """
    if isinstance(expr, RuleNode):
        # Base case: evaluate a leaf rule node
        return apply_simple_rule(doc, expr.attribute_name, expr.rule_type)
    
    elif isinstance(expr, NotNode):
        # NOT operator
        return not evaluate_expression(doc, expr.operand)
    
    elif isinstance(expr, BinaryNode):
        # Binary operators (AND/OR)
        if expr.operator == "AND":
            # Short-circuit AND evaluation
            return evaluate_expression(doc, expr.left) and evaluate_expression(doc, expr.right)
        elif expr.operator == "OR":
            # Short-circuit OR evaluation
            return evaluate_expression(doc, expr.left) or evaluate_expression(doc, expr.right)
    
    # Should not reach here if the expression tree is well-formed
    raise ValueError(f"Invalid expression node type: {type(expr)}")

def tokenize_expression(expression):
    """
    Tokenize a rule expression string into a list of tokens.
    
    Args:
        expression: A string containing a boolean rule expression
                   (e.g., "not rule1:any and rule2:all")
                   
    Returns:
        A list of Token objects
    """
    tokens = []
    i = 0
    expression = expression.strip()
    
    while i < len(expression):
        char = expression[i]
        
        # Skip whitespace
        if char.isspace():
            i += 1
            continue
            
        # Handle parentheses
        elif char == '(':
            tokens.append(Token(TokenType.LPAREN))
            i += 1
        elif char == ')':
            tokens.append(Token(TokenType.RPAREN))
            i += 1
            
        # Handle operators
        elif i + 2 < len(expression) and expression[i:i+3].lower() == 'and':
            # Check if it's a standalone 'and' and not part of a word
            if (i == 0 or expression[i-1].isspace() or expression[i-1] in "()") and \
               (i+3 >= len(expression) or expression[i+3].isspace() or expression[i+3] in "()"):
                tokens.append(Token(TokenType.AND))
                i += 3
            else:
                # It's part of an attribute name
                rule_start = i
                while i < len(expression) and not expression[i].isspace() and expression[i] not in "()":
                    if i + 1 < len(expression) and expression[i] == ':':
                        break
                    i += 1
                
                # Process rule if we found a colon
                if i < len(expression) and expression[i] == ':':
                    rule_end = i
                    i += 1  # Skip the colon
                    
                    # Find the rule type
                    type_start = i
                    while i < len(expression) and not expression[i].isspace() and expression[i] not in "()":
                        i += 1
                    
                    rule_name = expression[rule_start:rule_end]
                    rule_type = expression[type_start:i]
                    
                    tokens.append(Token(TokenType.RULE, (rule_name, rule_type)))
                else:
                    raise ValueError(f"Invalid rule format at position {rule_start}")
        
        elif i + 1 < len(expression) and expression[i:i+2].lower() == 'or':
            # Check if it's a standalone 'or' and not part of a word
            if (i == 0 or expression[i-1].isspace() or expression[i-1] in "()") and \
               (i+2 >= len(expression) or expression[i+2].isspace() or expression[i+2] in "()"):
                tokens.append(Token(TokenType.OR))
                i += 2
            else:
                # Part of an attribute name
                rule_start = i
                while i < len(expression) and not expression[i].isspace() and expression[i] not in "()":
                    if i + 1 < len(expression) and expression[i] == ':':
                        break
                    i += 1
                
                # Process rule if we found a colon
                if i < len(expression) and expression[i] == ':':
                    rule_end = i
                    i += 1  # Skip the colon
                    
                    # Find the rule type
                    type_start = i
                    while i < len(expression) and not expression[i].isspace() and expression[i] not in "()":
                        i += 1
                    
                    rule_name = expression[rule_start:rule_end]
                    rule_type = expression[type_start:i]
                    
                    tokens.append(Token(TokenType.RULE, (rule_name, rule_type)))
                else:
                    raise ValueError(f"Invalid rule format at position {rule_start}")
        
        elif i + 2 < len(expression) and expression[i:i+3].lower() == 'not':
            # Check if it's a standalone 'not' and not part of a word
            if (i == 0 or expression[i-1].isspace() or expression[i-1] in "()") and \
               (i+3 >= len(expression) or expression[i+3].isspace() or expression[i+3] in "()"):
                tokens.append(Token(TokenType.NOT))
                i += 3
            else:
                # Part of an attribute name
                rule_start = i
                while i < len(expression) and not expression[i].isspace() and expression[i] not in "()":
                    if i + 1 < len(expression) and expression[i] == ':':
                        break
                    i += 1
                
                # Process rule if we found a colon
                if i < len(expression) and expression[i] == ':':
                    rule_end = i
                    i += 1  # Skip the colon
                    
                    # Find the rule type
                    type_start = i
                    while i < len(expression) and not expression[i].isspace() and expression[i] not in "()":
                        i += 1
                    
                    rule_name = expression[rule_start:rule_end]
                    rule_type = expression[type_start:i]
                    
                    tokens.append(Token(TokenType.RULE, (rule_name, rule_type)))
                else:
                    raise ValueError(f"Invalid rule format at position {rule_start}")
        
        # Handle rule (attribute:type)
        else:
            rule_start = i
            while i < len(expression) and not expression[i].isspace() and expression[i] not in "()":
                if i + 1 < len(expression) and expression[i] == ':':
                    break
                i += 1
            
            # Process rule if we found a colon
            if i < len(expression) and expression[i] == ':':
                rule_end = i
                i += 1  # Skip the colon
                
                # Find the rule type
                type_start = i
                while i < len(expression) and not expression[i].isspace() and expression[i] not in "()":
                    i += 1
                
                rule_name = expression[rule_start:rule_end]
                rule_type = expression[type_start:i]
                
                tokens.append(Token(TokenType.RULE, (rule_name, rule_type)))
            else:
                raise ValueError(f"Invalid rule format at position {rule_start}")
    
    tokens.append(Token(TokenType.EOF))
    return tokens

class Parser:
    """
    Parser for boolean rule expressions.
    Implements a recursive descent parser for expressions with the following grammar:
    
    expression    → or_expr
    or_expr       → and_expr ("or" and_expr)*
    and_expr      → unary_expr ("and" unary_expr)*
    unary_expr    → "not" unary_expr | primary
    primary       → rule | "(" expression ")"
    rule          → ATTRIBUTE ":" RULE_TYPE
    """
    
    def __init__(self, tokens):
        self.tokens = tokens
        self.current = 0
    
    def parse(self):
        """Parse the tokens into an expression tree."""
        return self.expression()
    
    def expression(self):
        """Parse an expression (top level)."""
        return self.or_expr()
    
    def or_expr(self):
        """Parse an OR expression."""
        expr = self.and_expr()
        
        while self.match(TokenType.OR):
            right = self.and_expr()
            expr = BinaryNode(expr, right, "OR")
        
        return expr
    
    def and_expr(self):
        """Parse an AND expression."""
        expr = self.unary_expr()
        
        while self.match(TokenType.AND):
            right = self.unary_expr()
            expr = BinaryNode(expr, right, "AND")
        
        return expr
    
    def unary_expr(self):
        """Parse a unary expression (NOT)."""
        if self.match(TokenType.NOT):
            operand = self.unary_expr()
            return NotNode(operand)
        
        return self.primary()
    
    def primary(self):
        """Parse a primary expression (rule or parenthesized expression)."""
        if self.match(TokenType.RULE):
            rule_tuple = self.previous().value
            attribute_name, rule_type = rule_tuple
            
            # Validate rule type
            if rule_type not in ["any", "all"]:
                raise ValueError(f"Invalid rule type: {rule_type}. Supported types: 'any', 'all'")
                
            return RuleNode(attribute_name, rule_type)
        
        if self.match(TokenType.LPAREN):
            expr = self.expression()
            self.consume(TokenType.RPAREN, "Expected ')' after expression.")
            return expr
        
        raise ValueError(f"Expected rule or '(' at position {self.current}")
    
    def match(self, *types):
        """Check if the current token matches any of the given types."""
        for type in types:
            if self.check(type):
                self.advance()
                return True
        
        return False
    
    def check(self, type):
        """Check if the current token is of the given type without advancing."""
        if self.is_at_end():
            return False
        return self.peek().type == type
    
    def advance(self):
        """Advance to the next token and return the previous one."""
        if not self.is_at_end():
            self.current += 1
        return self.previous()
    
    def consume(self, type, message):
        """Consume the current token if it matches the expected type."""
        if self.check(type):
            return self.advance()
        
        raise ValueError(f"{message} at position {self.current}")
    
    def is_at_end(self):
        """Check if we've reached the end of the tokens."""
        return self.peek().type == TokenType.EOF
    
    def peek(self):
        """Return the current token without advancing."""
        return self.tokens[self.current]
    
    def previous(self):
        """Return the previous token."""
        return self.tokens[self.current - 1]

def parse_rule(rule_string):
    """
    Parse a rule string into an expression tree or a simple attribute-rule_type tuple.
    
    Args:
        rule_string: A string containing a rule or boolean expression of rules
        
    Returns:
        Either a tuple (attribute_name, rule_type) for simple rules,
        or an ExpressionNode for complex boolean expressions
    """
    # Check if this is a simple rule
    if "and" not in rule_string.lower() and "or" not in rule_string.lower() and "not" not in rule_string.lower() and "(" not in rule_string and ")" not in rule_string:
        # Simple rule format: attribute_name:rule_type
        parts = rule_string.split(":", 1)
        if len(parts) != 2:
            raise ValueError(f"Invalid rule format: {rule_string}. Expected format: 'attribute_name:rule_type'")
        
        attribute_name, rule_type = parts
        if rule_type not in ["any", "all"]:
            raise ValueError(f"Invalid rule type: {rule_type}. Supported types: 'any', 'all'")
        
        return attribute_name, rule_type
    else:
        # Complex rule expression
        try:
            tokens = tokenize_expression(rule_string)
            parser = Parser(tokens)
            return parser.parse()
        except Exception as e:
            raise ValueError(f"Error parsing expression '{rule_string}': {e}")

def is_complex_expression(rule):
    """Check if the rule is a complex boolean expression."""
    return isinstance(rule, ExpressionNode)

def get_matching_files(ref_files, hyp_files):
    """
    Find files that exist in both reference and hypothesis folders,
    matching by their relative paths.
    
    Returns dict mapping ref_path -> hyp_path for matched files
    """
    # First, convert to relative paths for matching
    def get_relative_path(path, base_folder):
        if path.startswith("s3://"):
            _, full_key = parse_s3_path(path)
            _, base_key = parse_s3_path(base_folder)
            return full_key[len(base_key):].lstrip("/") if full_key.startswith(base_key) else full_key
        else:
            return os.path.relpath(path, base_folder)
    
    ref_base = args.ref_folder
    hyp_base = args.hyp_folder
    
    ref_relative = {get_relative_path(path, ref_base): path for path in ref_files}
    hyp_relative = {get_relative_path(path, hyp_base): path for path in hyp_files}
    
    # Find matching files
    matched_files = {}
    for rel_path in ref_relative:
        if rel_path in hyp_relative:
            matched_files[ref_relative[rel_path]] = hyp_relative[rel_path]
    
    return matched_files

def calculate_iou(ref_ids, hyp_ids):
    """Calculate Intersection over Union of two sets of document IDs."""
    ref_set = set(ref_ids)
    hyp_set = set(hyp_ids)
    
    intersection = ref_set.intersection(hyp_set)
    union = ref_set.union(hyp_set)
    
    if not union:
        return 0.0
    
    return len(intersection) / len(union)

def collect_rule_stats(expression, doc):
    """
    Collect statistics for all rules within a complex expression.
    
    Args:
        expression: A rule expression (either a tuple or ExpressionNode)
        doc: The document to analyze
        
    Returns:
        A dictionary with rule statistics
    """
    rule_stats = defaultdict(int)
    
    # Handle simple rule
    if not is_complex_expression(expression):
        attribute_name, rule_type = expression
        # Only process if document has this attribute
        if ("attributes" in doc and doc["attributes"] and 
            attribute_name in doc["attributes"] and doc["attributes"][attribute_name]):
            # The rule name will be the key for the statistics
            rule_name = f"{attribute_name}:{rule_type}"
            
            # Count entries in the attribute
            entries = doc["attributes"][attribute_name]
            rule_stats[f"{rule_name}_total_entries"] += len(entries)
            
            # Count positive values
            for span in entries:
                if len(span) >= 3 and span[2] is True:
                    rule_stats[f"{rule_name}_positive_entries"] += 1
            
            # Check if document matches the rule
            if apply_simple_rule(doc, attribute_name, rule_type):
                rule_stats[f"{rule_name}_matched_docs"] += 1
        
        return rule_stats
    
    # For complex expressions, traverse the expression tree
    if isinstance(expression, RuleNode):
        # Base case: leaf node is a simple rule
        attribute_name, rule_type = expression.attribute_name, expression.rule_type
        if ("attributes" in doc and doc["attributes"] and 
            attribute_name in doc["attributes"] and doc["attributes"][attribute_name]):
            # The rule name will be the key for the statistics
            rule_name = f"{attribute_name}:{rule_type}"
            
            # Count entries in the attribute
            entries = doc["attributes"][attribute_name]
            rule_stats[f"{rule_name}_total_entries"] += len(entries)
            
            # Count positive values
            for span in entries:
                if len(span) >= 3 and span[2] is True:
                    rule_stats[f"{rule_name}_positive_entries"] += 1
            
            # Check if document matches the rule
            if apply_simple_rule(doc, attribute_name, rule_type):
                rule_stats[f"{rule_name}_matched_docs"] += 1
    
    elif isinstance(expression, NotNode):
        # Get stats from the operand
        operand_stats = collect_rule_stats(expression.operand, doc)
        # Merge with current stats
        for key, value in operand_stats.items():
            rule_stats[key] += value
    
    elif isinstance(expression, BinaryNode):
        # Get stats from both sides
        left_stats = collect_rule_stats(expression.left, doc)
        right_stats = collect_rule_stats(expression.right, doc)
        
        # Merge with current stats
        for key, value in left_stats.items():
            rule_stats[key] += value
        for key, value in right_stats.items():
            rule_stats[key] += value
    
    return rule_stats

def get_expression_summary(expression):
    """
    Generate a string representation of a rule expression.
    
    Args:
        expression: A rule expression (either a tuple or ExpressionNode)
        
    Returns:
        A string representation of the expression
    """
    if not is_complex_expression(expression):
        return f"{expression[0]}:{expression[1]}"
    
    if isinstance(expression, RuleNode):
        return f"{expression.attribute_name}:{expression.rule_type}"
    
    elif isinstance(expression, NotNode):
        return f"not {get_expression_summary(expression.operand)}"
    
    elif isinstance(expression, BinaryNode):
        left_summary = get_expression_summary(expression.left)
        right_summary = get_expression_summary(expression.right)
        return f"({left_summary} {expression.operator.lower()} {right_summary})"
    
    return str(expression)

def compare_files(ref_path, hyp_path, ref_rule, hyp_rule, s3_client=None):
    """
    Compare two JSONL files using the specified rules and calculate IoU.
    
    Args:
        ref_path: Path to reference JSONL file
        hyp_path: Path to hypothesis JSONL file
        ref_rule: Rule expression for reference (tuple or ExpressionNode)
        hyp_rule: Rule expression for hypothesis (tuple or ExpressionNode)
        s3_client: S3 client for S3 paths
        
    Returns:
        Dictionary with comparison results
    """
    # Load the files
    ref_docs = load_jsonl_file(ref_path, s3_client)
    hyp_docs = load_jsonl_file(hyp_path, s3_client)
    
    # Extract document IDs and create ID-to-document maps
    ref_id_to_doc = {doc["id"]: doc for doc in ref_docs if "id" in doc}
    hyp_id_to_doc = {doc["id"]: doc for doc in hyp_docs if "id" in doc}
    
    # Get common document IDs
    common_ids = set(ref_id_to_doc.keys()).intersection(set(hyp_id_to_doc.keys()))
    
    # Apply rules to each document
    ref_matches = set()
    hyp_matches = set()
    
    # Track rule statistics
    ref_rule_stats = defaultdict(int)
    hyp_rule_stats = defaultdict(int)
    
    for doc_id in common_ids:
        ref_doc = ref_id_to_doc[doc_id]
        hyp_doc = hyp_id_to_doc[doc_id]
        
        # Collect statistics for all rules in the expressions
        doc_ref_rule_stats = collect_rule_stats(ref_rule, ref_doc)
        doc_hyp_rule_stats = collect_rule_stats(hyp_rule, hyp_doc)
        
        # Merge with overall stats
        for key, value in doc_ref_rule_stats.items():
            ref_rule_stats[key] += value
        for key, value in doc_hyp_rule_stats.items():
            hyp_rule_stats[key] += value
        
        # Check if document matches the rule expressions
        if apply_rule(ref_doc, ref_rule):
            ref_matches.add(doc_id)
            ref_rule_stats["expression_matched_docs"] += 1
        
        if apply_rule(hyp_doc, hyp_rule):
            hyp_matches.add(doc_id)
            hyp_rule_stats["expression_matched_docs"] += 1
    
    # Calculate IoU
    iou = calculate_iou(ref_matches, hyp_matches)
    
    # Collect detailed statistics
    tp = len(ref_matches.intersection(hyp_matches))
    fp = len(hyp_matches - ref_matches)
    fn = len(ref_matches - hyp_matches)
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    # Generate string representations of the expressions
    ref_rule_str = get_expression_summary(ref_rule)
    hyp_rule_str = get_expression_summary(hyp_rule)
    
    return {
        "ref_file": ref_path,
        "hyp_file": hyp_path,
        "total_docs": len(common_ids),
        "ref_rule": ref_rule_str,
        "hyp_rule": hyp_rule_str,
        "ref_matches": len(ref_matches),
        "hyp_matches": len(hyp_matches),
        "intersection": tp,
        "union": tp + fp + fn,
        "true_positives": tp,
        "false_positives": fp,
        "false_negatives": fn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "iou": iou,
        "ref_rule_stats": dict(ref_rule_stats),
        "hyp_rule_stats": dict(hyp_rule_stats)
    }

def format_rule_stats(rule_stats):
    """Format rule statistics for display."""
    # Group the statistics by rule name
    grouped_stats = defaultdict(dict)
    
    # Process regular rule stats (format: "{rule_name}_{stat_type}")
    for key, value in rule_stats.items():
        if key == "expression_matched_docs":
            # Special case for the overall expression match count
            continue
            
        # Extract rule name and stat type
        if "_total_entries" in key:
            rule_name = key.replace("_total_entries", "")
            grouped_stats[rule_name]["total_entries"] = value
        elif "_positive_entries" in key:
            rule_name = key.replace("_positive_entries", "")
            grouped_stats[rule_name]["positive_entries"] = value
        elif "_matched_docs" in key:
            rule_name = key.replace("_matched_docs", "")
            grouped_stats[rule_name]["matched_docs"] = value
    
    # Format the grouped statistics as a list of strings
    formatted_stats = []
    for rule_name, stats in grouped_stats.items():
        formatted_stats.append(
            f"  {rule_name}:\n"
            f"    - Total Entries: {stats.get('total_entries', 0)}\n"
            f"    - Positive Entries: {stats.get('positive_entries', 0)}\n"
            f"    - Matched Documents: {stats.get('matched_docs', 0)}"
        )
    
    # Add the expression matched count if available
    if "expression_matched_docs" in rule_stats:
        formatted_stats.append(f"  Overall Expression Matched Documents: {rule_stats['expression_matched_docs']}")
    
    return "\n".join(formatted_stats)

def main():
    global args
    args = parse_args()
    
    # Set up S3 client if needed
    s3_client = None
    if args.ref_folder.startswith("s3://") or args.hyp_folder.startswith("s3://"):
        session = boto3.Session(profile_name=args.aws_profile) if args.aws_profile else boto3.Session()
        s3_client = session.client("s3")
    
    # Parse the rules
    logger.info(f"Parsing reference rule expression: {args.ref_rule}")
    ref_rule = parse_rule(args.ref_rule)
    
    logger.info(f"Parsing hypothesis rule expression: {args.hyp_rule}")
    hyp_rule = parse_rule(args.hyp_rule)
    
    # Generate string representations of the expressions
    ref_rule_str = get_expression_summary(ref_rule)
    hyp_rule_str = get_expression_summary(hyp_rule)
    
    logger.info(f"Reference rule parsed as: {ref_rule_str}")
    logger.info(f"Hypothesis rule parsed as: {hyp_rule_str}")
    
    # List JSONL files in both folders
    logger.info(f"Finding JSONL files in reference folder: {args.ref_folder}")
    ref_files = list_jsonl_files(args.ref_folder, s3_client, args.recursive)
    
    logger.info(f"Finding JSONL files in hypothesis folder: {args.hyp_folder}")
    hyp_files = list_jsonl_files(args.hyp_folder, s3_client, args.recursive)
    
    logger.info(f"Found {len(ref_files)} files in reference folder and {len(hyp_files)} files in hypothesis folder")
    
    # Find matching files
    matched_files = get_matching_files(ref_files, hyp_files)
    logger.info(f"Found {len(matched_files)} matching files between folders")
    
    if not matched_files:
        logger.error("No matching files found between reference and hypothesis folders")
        sys.exit(1)
    
    # Process each pair of files
    results = []
    overall_stats = {
        "total_files": len(matched_files),
        "total_docs": 0,
        "ref_matches": 0,
        "hyp_matches": 0,
        "true_positives": 0,
        "false_positives": 0,
        "false_negatives": 0,
        # Initialize rule stats counters
        "ref_rule_stats": defaultdict(int),
        "hyp_rule_stats": defaultdict(int)
    }
    
    for i, (ref_path, hyp_path) in enumerate(matched_files.items()):
        logger.info(f"Processing file pair {i+1}/{len(matched_files)}: {os.path.basename(ref_path)}")
        
        file_result = compare_files(ref_path, hyp_path, ref_rule, hyp_rule, s3_client)
        results.append(file_result)
        
        # Accumulate overall statistics
        overall_stats["total_docs"] += file_result["total_docs"]
        overall_stats["ref_matches"] += file_result["ref_matches"]
        overall_stats["hyp_matches"] += file_result["hyp_matches"]
        overall_stats["true_positives"] += file_result["true_positives"]
        overall_stats["false_positives"] += file_result["false_positives"]
        overall_stats["false_negatives"] += file_result["false_negatives"]
        
        # Accumulate rule statistics
        for key, value in file_result["ref_rule_stats"].items():
            overall_stats["ref_rule_stats"][key] += value
        for key, value in file_result["hyp_rule_stats"].items():
            overall_stats["hyp_rule_stats"][key] += value
    
    # Calculate overall metrics
    tp = overall_stats["true_positives"]
    fp = overall_stats["false_positives"]
    fn = overall_stats["false_negatives"]
    
    overall_stats["precision"] = tp / (tp + fp) if (tp + fp) > 0 else 0
    overall_stats["recall"] = tp / (tp + fn) if (tp + fn) > 0 else 0
    overall_stats["f1"] = (
        2 * overall_stats["precision"] * overall_stats["recall"] / 
        (overall_stats["precision"] + overall_stats["recall"])
        if (overall_stats["precision"] + overall_stats["recall"]) > 0 else 0
    )
    overall_stats["iou"] = tp / (tp + fp + fn) if (tp + fp + fn) > 0 else 0
    
    # Convert defaultdicts to regular dicts for JSON serialization
    overall_stats["ref_rule_stats"] = dict(overall_stats["ref_rule_stats"])
    overall_stats["hyp_rule_stats"] = dict(overall_stats["hyp_rule_stats"])
    
    # Prepare final output
    output = {
        "config": {
            "ref_folder": args.ref_folder,
            "hyp_folder": args.hyp_folder,
            "ref_rule": args.ref_rule,
            "ref_rule_parsed": ref_rule_str,
            "hyp_rule": args.hyp_rule,
            "hyp_rule_parsed": hyp_rule_str
        },
        "overall": overall_stats,
        "file_results": results
    }
    
    # Save results
    with open(args.output_file, "w") as f:
        json.dump(output, f, indent=2)
    
    # Print summary
    logger.info("\n--- COMPARISON SUMMARY ---")
    logger.info(f"Reference Rule Expression: {args.ref_rule}")
    logger.info(f"  Parsed as: {ref_rule_str}")
    logger.info(f"Hypothesis Rule Expression: {args.hyp_rule}")
    logger.info(f"  Parsed as: {hyp_rule_str}")
    logger.info(f"Total Documents: {overall_stats['total_docs']}")
    
    # Print rule statistics
    logger.info("\n--- RULE MATCH STATISTICS ---")
    
    logger.info("\nReference Rules:")
    logger.info(format_rule_stats(overall_stats["ref_rule_stats"]))
    
    logger.info("\nHypothesis Rules:")
    logger.info(format_rule_stats(overall_stats["hyp_rule_stats"]))
    
    # Print comparison metrics
    logger.info("\n--- COMPARISON METRICS ---")
    logger.info(f"IoU: {overall_stats['iou']:.4f}")
    logger.info(f"Precision: {overall_stats['precision']:.4f}")
    logger.info(f"Recall: {overall_stats['recall']:.4f}")
    logger.info(f"F1 Score: {overall_stats['f1']:.4f}")
    logger.info(f"Detailed results saved to: {args.output_file}")

if __name__ == "__main__":
    main()