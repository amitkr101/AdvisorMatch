#!/usr/bin/env python3
"""
Build FAISS index from stored embeddings for efficient similarity search.

This script loads embeddings from the database, builds a FAISS index,
and saves it to disk along with the paper_id mapping.

Usage:
    python3 build_faiss_index.py [--verify]
"""

import sqlite3
import numpy as np
import pickle
import json
import sys
import argparse
import faiss

DB_NAME = "advisormatch_openalex.db"
INDEX_FILE = "faiss_index.bin"
MAPPING_FILE = "paper_id_mapping.json"
from config import EMBEDDING_DIM

def load_embeddings_from_db(conn):
    """
    Load all embeddings from database.
    Returns: (paper_ids, embeddings_matrix)
    """
    cursor = conn.cursor()
    cursor.execute("""
        SELECT paper_id, embedding 
        FROM publications
        WHERE embedding IS NOT NULL AND embedding != ''
        ORDER BY paper_id
    """)
    
    rows = cursor.fetchall()
    
    if not rows:
        raise ValueError("No embeddings found in database. Run generate_embeddings.py first!")
    
    paper_ids = []
    embeddings_list = []
    
    print(f"Loading {len(rows)} embeddings from database...")
    for paper_id, embedding_blob in rows:
        paper_ids.append(paper_id)
        
        # Try to deserialize - handle both pickle and raw binary formats
        try:
            # First try pickle format
            embedding = pickle.loads(embedding_blob)
        except:
            # If pickle fails, assume it's raw binary float32 data
            # Each float32 is 4 bytes, so divide by 4 to get number of dimensions
            embedding = np.frombuffer(embedding_blob, dtype=np.float32)
        
        embeddings_list.append(embedding)
    
    # Convert to numpy matrix
    embeddings_matrix = np.vstack(embeddings_list).astype('float32')
    
    return paper_ids, embeddings_matrix

def build_faiss_index(embeddings: np.ndarray) -> faiss.Index:
    """
    Build FAISS index for cosine similarity search.
    Using IndexFlatIP (Inner Product) with normalized vectors = cosine similarity
    """
    print(f"\nBuilding FAISS index...")
    print(f"  Embedding dimension: {embeddings.shape[1]}")
    print(f"  Number of vectors: {embeddings.shape[0]}")
    
    # Verify embeddings are normalized (for cosine similarity)
    norms = np.linalg.norm(embeddings, axis=1)
    print(f"  Vector norms - min: {norms.min():.4f}, max: {norms.max():.4f}, mean: {norms.mean():.4f}")
    
    # Create index (Inner Product for normalized vectors = cosine similarity)
    index = faiss.IndexFlatIP(embeddings.shape[1])
    
    # Add vectors to index
    index.add(embeddings)
    
    print(f"✓ FAISS index built successfully")
    print(f"  Index size: {index.ntotal} vectors")
    
    return index

def save_index_and_mapping(index: faiss.Index, paper_ids: list):
    """
    Save FAISS index and paper_id mapping to disk.
    """
    print(f"\nSaving FAISS index to: {INDEX_FILE}")
    faiss.write_index(index, INDEX_FILE)
    
    print(f"Saving paper ID mapping to: {MAPPING_FILE}")
    # Create mapping: index -> paper_id
    mapping = {i: paper_id for i, paper_id in enumerate(paper_ids)}
    with open(MAPPING_FILE, 'w') as f:
        json.dump(mapping, f, indent=2)
    
    print("✓ Files saved successfully")

def verify_index(index: faiss.Index, embeddings: np.ndarray):
    """
    Verify the index works correctly by performing a test search.
    """
    print("\n" + "="*80)
    print("Verifying FAISS index...")
    print("="*80)
    
    # Test with first embedding (should return itself as top result)
    test_vector = embeddings[0:1]
    k = 5
    
    distances, indices = index.search(test_vector, k)
    
    print(f"\nTest search (k={k}):")
    print(f"  Query: Vector 0")
    print(f"  Results:")
    for i, (idx, dist) in enumerate(zip(indices[0], distances[0])):
        print(f"    {i+1}. Index {idx}, Similarity: {dist:.4f}")
    
    # Verify top result is the query itself
    if indices[0][0] == 0:
        print("\n✓ Verification passed: Top result matches query vector")
    else:
        print("\n⚠ Warning: Top result doesn't match query vector")
    
    print("="*80)

def main(verify: bool = False):
    """
    Main function to build FAISS index.
    """
    print("="*80)
    print("AdvisorMatch: FAISS Index Builder")
    print("="*80)
    
    # Connect to database
    print(f"\nConnecting to database: {DB_NAME}")
    conn = sqlite3.connect(DB_NAME)
    
    # Load embeddings
    paper_ids, embeddings = load_embeddings_from_db(conn)
    conn.close()
    
    print(f"\n✓ Loaded {len(paper_ids)} embeddings")
    print(f"  Embedding shape: {embeddings.shape}")
    
    # Build FAISS index
    index = build_faiss_index(embeddings)
    
    # Save to disk
    save_index_and_mapping(index, paper_ids)
    
    # Verify if requested
    if verify:
        verify_index(index, embeddings)
    
    print("\n" + "="*80)
    print("FAISS index creation complete!")
    print("="*80)
    print(f"\nGenerated files:")
    print(f"  - {INDEX_FILE} (FAISS index)")
    print(f"  - {MAPPING_FILE} (paper ID mapping)")
    print(f"\nNext step: Use test_search.py to test semantic search")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build FAISS index from embeddings")
    parser.add_argument("--verify", action="store_true", help="Verify index after building")
    
    args = parser.parse_args()
    
    try:
        main(verify=args.verify)
    except Exception as e:
        print(f"\n❌ Error: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)
