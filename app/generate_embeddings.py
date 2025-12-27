#!/usr/bin/env python3
"""
Generate vector embeddings for all publications in the database.

This script uses Sentence-BERT (all-mpnet-base-v2) to generate semantic embeddings
for each publication's title and abstract, then stores them in the database.

Usage:
    python3 generate_embeddings.py [--batch-size N] [--test]
"""

import sqlite3
import numpy as np
import pickle
import sys
from sentence_transformers import SentenceTransformer
from typing import List, Tuple
import argparse

from config import MODEL_NAME, EMBEDDING_DIM

DB_NAME = "advisormatch_openalex.db"

def load_model():
    """Load the Sentence-BERT model."""
    print(f"Loading model: {MODEL_NAME}...")
    model = SentenceTransformer(MODEL_NAME)
    print(f"Model loaded successfully. Embedding dimension: {EMBEDDING_DIM}")
    return model

def fetch_publications(conn) -> List[Tuple[str, str, str]]:
    """
    Fetch all publications from database.
    Returns: List of (paper_id, title, abstract) tuples
    """
    cursor = conn.cursor()
    # Force re-generation for all papers since we changed the model
    cursor.execute("""
        SELECT paper_id, title, abstract 
        FROM publications
    """)
    return cursor.fetchall()

def generate_text_for_embedding(title: str, abstract: str) -> str:
    """
    Combine title and abstract for embedding generation.
    SPECTRE expects: title + [SEP] + abstract
    """
    # Handle None values
    title = title or ""
    abstract = abstract or ""
    
    # Combine with separator
    if abstract:
        return f"{title} [SEP] {abstract}"
    return title

def generate_embeddings_batch(model, texts: List[str], batch_size: int = 32) -> np.ndarray:
    """
    Generate embeddings for a batch of texts.
    Returns normalized embeddings for cosine similarity.
    """
    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=True,
        normalize_embeddings=True  # L2 normalization for cosine similarity
    )
    return embeddings

def store_embedding(conn, paper_id: str, embedding: np.ndarray):
    """
    Store embedding in database as pickled BLOB.
    """
    cursor = conn.cursor()
    embedding_blob = pickle.dumps(embedding)
    cursor.execute("""
        UPDATE publications 
        SET embedding = ?
        WHERE paper_id = ?
    """, (embedding_blob, paper_id))

def main(batch_size: int = 32, test_mode: bool = False):
    """
    Main function to generate and store embeddings.
    """
    print("="*80)
    print("AdvisorMatch: Embedding Generation")
    print("="*80)
    
    # Load model
    model = load_model()
    
    # Connect to database
    print(f"\nConnecting to database: {DB_NAME}")
    conn = sqlite3.connect(DB_NAME)
    
    # Fetch publications
    print("\nFetching publications without embeddings...")
    publications = fetch_publications(conn)
    
    if not publications:
        print("✓ All publications already have embeddings!")
        conn.close()
        return
    
    print(f"Found {len(publications)} publications to process")
    
    if test_mode:
        print("\n[TEST MODE] Processing only first 5 publications...")
        publications = publications[:5]
    
    # Prepare texts for embedding
    print("\nPreparing texts for embedding generation...")
    paper_ids = []
    texts = []
    
    for paper_id, title, abstract in publications:
        paper_ids.append(paper_id)
        text = generate_text_for_embedding(title, abstract)
        texts.append(text)
    
    # Generate embeddings
    print(f"\nGenerating embeddings (batch_size={batch_size})...")
    embeddings = generate_embeddings_batch(model, texts, batch_size)
    
    # Verify embedding dimensions
    print(f"\nEmbedding shape: {embeddings.shape}")
    assert embeddings.shape[1] == EMBEDDING_DIM, f"Expected {EMBEDDING_DIM} dimensions, got {embeddings.shape[1]}"
    
    # Store embeddings
    print("\nStoring embeddings in database...")
    for i, (paper_id, embedding) in enumerate(zip(paper_ids, embeddings)):
        store_embedding(conn, paper_id, embedding)
        if (i + 1) % 50 == 0:
            print(f"  Processed {i + 1}/{len(paper_ids)} publications")
            conn.commit()
    
    # Final commit
    conn.commit()
    print(f"\n✓ Successfully generated and stored {len(embeddings)} embeddings")
    
    # Verification
    print("\nVerifying embeddings in database...")
    cursor = conn.cursor()
    cursor.execute("SELECT COUNT(*) FROM publications WHERE embedding IS NOT NULL AND embedding != ''")
    count = cursor.fetchone()[0]
    print(f"✓ Total publications with embeddings: {count}")
    
    conn.close()
    print("\n" + "="*80)
    print("Embedding generation complete!")
    print("="*80)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate embeddings for publications")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size for embedding generation")
    parser.add_argument("--test", action="store_true", help="Test mode: process only 5 publications")
    
    args = parser.parse_args()
    
    try:
        main(batch_size=args.batch_size, test_mode=args.test)
    except Exception as e:
        print(f"\n❌ Error: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)
