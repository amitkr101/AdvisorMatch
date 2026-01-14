import sqlite3
import string
from rank_bm25 import BM25Okapi
from typing import List, Dict, Any

class BM25Searcher:
    def __init__(self, db_path: str):
        self.db_path = db_path
        self.bm25 = None
        self.papers = []
        self._build_index()

    def _tokenize(self, text: str) -> List[str]:
        # Simple tokenizer: lowercase and remove punctuation
        if not text:
            return []
        text = text.lower()
        # Replace punctuation with spaces
        for char in string.punctuation:
            text = text.replace(char, ' ')
        return text.split()

    def _build_index(self):
        print("Building BM25 index...")
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Fetch all papers
        cursor.execute("SELECT paper_id, title, abstract, year, citation_count, venue, url FROM publications")
        rows = cursor.fetchall()
        
        tokenized_corpus = []
        self.papers = []
        
        for row in rows:
            paper_id, title, abstract, year, citations, venue, url = row
            
            # Combine title and abstract for indexing
            content = f"{title or ''} {abstract or ''}"
            tokens = self._tokenize(content)
            
            tokenized_corpus.append(tokens)
            self.papers.append({
                'paper_id': paper_id,
                'title': title,
                'abstract': abstract,
                'year': year,
                'citations': citations,
                'venue': venue,
                'url': url
            })
            
        conn.close()
        
        if tokenized_corpus:
            self.bm25 = BM25Okapi(tokenized_corpus)
            print(f"BM25 index built with {len(self.papers)} documents.")
        else:
            print("Warning: No documents found for BM25 index.")

    def search(self, query: str, top_k: int = 10) -> List[Dict[str, Any]]:
        if not self.bm25:
            return []
            
        tokenized_query = self._tokenize(query)
        # Get scores
        doc_scores = self.bm25.get_scores(tokenized_query)
        
        # Zip scores with papers and sort
        scored_papers = []
        for i, score in enumerate(doc_scores):
            if score > 0:
                paper = self.papers[i].copy()
                paper['score'] = float(score)  # Ensure float for JSON serialization
                scored_papers.append(paper)
        
        # Sort by score descending
        scored_papers.sort(key=lambda x: x['score'], reverse=True)
        
        return scored_papers[:top_k]
