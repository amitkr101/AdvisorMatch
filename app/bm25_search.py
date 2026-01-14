"""
BM25 Search implementation for AdvisorMatch.
"""

import sqlite3
import math
from collections import defaultdict
from typing import List, Dict
import re


class BM25Searcher:
    """
    BM25 (Best Matching 25) searcher for lexical search.
    """
    
    def __init__(self, db_path, k1=1.5, b=0.75):
        """
        Initialize BM25 searcher.
        
        Args:
            db_path: Path to SQLite database
            k1: BM25 parameter controlling term frequency saturation (default: 1.5)
            b: BM25 parameter controlling length normalization (default: 0.75)
        """
        self.db_path = db_path
        self.k1 = k1
        self.b = b
        
    def tokenize(self, text):
        """Simple tokenization: lowercase and split on non-alphanumeric."""
        if not text:
            return []
        text = text.lower()
        tokens = re.findall(r'\b\w+\b', text)
        return tokens
    
    def search(self, query, top_k=10):
        """
        Search for papers using BM25.
        
        Args:
            query: Search query string
            top_k: Number of top results to return
            
        Returns:
            List of paper results with scores
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Tokenize query
        query_tokens = self.tokenize(query)
        if not query_tokens:
            return []
        
        # Get all papers with their text
        cursor.execute("""
            SELECT paper_id, title, abstract, year, citation_count, venue, url
            FROM papers
        """)
        papers = cursor.fetchall()
        
        if not papers:
            return []
        
        # Calculate document lengths and average length
        doc_lengths = {}
        total_length = 0
        for paper in papers:
            paper_id, title, abstract, *_ = paper
            text = f"{title or ''} {abstract or ''}"
            tokens = self.tokenize(text)
            doc_lengths[paper_id] = len(tokens)
            total_length += len(tokens)
        
        avg_doc_length = total_length / len(papers) if papers else 1
        
        # Calculate IDF for query terms
        N = len(papers)
        idf = {}
        for term in set(query_tokens):
            # Count documents containing this term
            df = 0
            for paper in papers:
                paper_id, title, abstract, *_ = paper
                text = f"{title or ''} {abstract or ''}"
                if term in self.tokenize(text):
                    df += 1
            
            # Calculate IDF
            if df > 0:
                idf[term] = math.log((N - df + 0.5) / (df + 0.5) + 1)
            else:
                idf[term] = 0
        
        # Calculate BM25 scores
        scores = []
        for paper in papers:
            paper_id, title, abstract, year, citations, venue, url = paper
            text = f"{title or ''} {abstract or ''}"
            tokens = self.tokenize(text)
            
            # Calculate term frequencies
            tf = defaultdict(int)
            for token in tokens:
                tf[token] += 1
            
            # Calculate BM25 score
            score = 0
            doc_length = doc_lengths[paper_id]
            
            for term in query_tokens:
                if term in tf:
                    term_freq = tf[term]
                    numerator = term_freq * (self.k1 + 1)
                    denominator = term_freq + self.k1 * (1 - self.b + self.b * (doc_length / avg_doc_length))
                    score += idf.get(term, 0) * (numerator / denominator)
            
            if score > 0:
                scores.append({
                    'paper_id': paper_id,
                    'title': title,
                    'year': year,
                    'score': score,
                    'citations': citations or 0,
                    'venue': venue,
                    'url': url
                })
        
        conn.close()
        
        # Sort by score and return top_k
        scores.sort(key=lambda x: x['score'], reverse=True)
        return scores[:top_k]
