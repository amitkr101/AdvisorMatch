"""
AdvisorMatch FastAPI Application

REST API for semantic search of thesis advisors based on research interests.
"""

import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import sqlite3
import json
import time
from typing import List
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

from config import (
    API_TITLE, API_VERSION, API_DESCRIPTION, CORS_ORIGINS,
    DB_PATH, INDEX_PATH, MAPPING_PATH, MODEL_NAME, TOP_K_PAPERS
)
from models import (
    SearchRequest, SearchResponse, ProfessorResult, PublicationSummary,
    ProfessorDetail, PublicationDetail, HealthResponse
)
from ranking import (
    rank_professors, get_professor_details, get_publication_details
)
from spellcheck import DomainSpellChecker
from bm25_search import BM25Searcher


app = FastAPI(
    title=API_TITLE,
    version=API_VERSION,
    description=API_DESCRIPTION
)

# Add CORS middleware - Allow all origins for development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins (including file://)
    allow_credentials=False,  # Set to False when using allow_origins=["*"]
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables for model, index, and mapping
model = None
index = None
paper_mapping = None
spell_checker = None
bm25_searcher = None


@app.on_event("startup")
async def startup_event():
    """Load model, FAISS index, and paper mapping on startup"""
    global model, index, paper_mapping, spell_checker, bm25_searcher
    
    print("Loading Sentence-BERT model...")
    model = SentenceTransformer(MODEL_NAME)
    
    print("Loading FAISS index...")
    index = faiss.read_index(str(INDEX_PATH))
    
    print("Loading paper ID mapping...")
    with open(MAPPING_PATH, 'r') as f:
        paper_mapping = json.load(f)
    # Convert string keys to integers
    paper_mapping = {int(k): v for k, v in paper_mapping.items()}
    
    print("Loading spell checker...")
    spell_checker = DomainSpellChecker(str(DB_PATH))
    
    print("Loading BM25 searcher...")
    bm25_searcher = BM25Searcher(str(DB_PATH))
    
    print(f"✓ Startup complete. Index size: {index.ntotal} vectors")


@app.get("/", tags=["Root"])
async def root():
    """Root endpoint"""
    return {
        "message": "AdvisorMatch API",
        "version": API_VERSION,
        "docs": "/docs"
    }


@app.get("/health", response_model=HealthResponse, tags=["Health"])
async def health_check():
    """Health check endpoint"""
    db_connected = False
    try:
        conn = sqlite3.connect(str(DB_PATH))
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM professors")
        cursor.fetchone()
        conn.close()
        db_connected = True
    except:
        pass
    
    return HealthResponse(
        status="healthy" if all([model, index, paper_mapping, db_connected]) else "degraded",
        version=API_VERSION,
        database_connected=db_connected,
        index_loaded=index is not None,
        model_loaded=model is not None
    )


@app.post("/api/search", response_model=SearchResponse, tags=["Search"])
async def search_advisors(request: SearchRequest):
    """
    Search for advisors based on research query.
    
    Uses semantic search with enhanced ranking algorithm that considers:
    - Semantic similarity to query
    - Publication recency (exponential decay)
    - Author activity (recent publications bonus)
    - Citation impact (log-normalized citation counts)
    """
    start_time = time.time()
    
    # Validate model and index are loaded
    if not all([model, index, paper_mapping]):
        raise HTTPException(status_code=503, detail="Service not ready. Model or index not loaded.")
    
    try:
        # Spell check query
        corrected_query = spell_checker.correct_text(request.query)
        if corrected_query != request.query.lower():
            print(f"Corrected query: '{request.query}' -> '{corrected_query}'")
        
        # Generate query embedding (use corrected query)
        query_embedding = model.encode([corrected_query], normalize_embeddings=True)
        query_embedding = query_embedding.astype('float32')
        
        # Search FAISS index
        distances, indices = index.search(query_embedding, TOP_K_PAPERS)
        
        # Get paper IDs and similarities
        paper_ids = [paper_mapping[idx] for idx in indices[0]]
        similarities = distances[0].tolist()
        
        # Connect to database
        conn = sqlite3.connect(str(DB_PATH))
        
        # Rank professors
        rankings = rank_professors(paper_ids, similarities, conn, top_k=request.top_k)
        
        # Build response
        results = []
        for ranking in rankings:
            prof_id = ranking['professor_id']
            
            # Get professor details
            prof_details = get_professor_details(prof_id, conn)
            if not prof_details:
                continue
            
            # Get top publications if requested
            top_pubs = None
            if request.include_publications:
                top_pubs = []
                for paper_id in ranking['top_paper_ids'][:3]:  # Top 3 publications
                    pub_details = get_publication_details(paper_id, conn)
                    if pub_details:
                        # Find similarity for this paper
                        try:
                            paper_idx = paper_ids.index(paper_id)
                            similarity = similarities[paper_idx]
                        except ValueError:
                            similarity = 0.0
                        
                        top_pubs.append(PublicationSummary(
                            paper_id=pub_details['paper_id'],
                            title=pub_details['title'],
                            year=pub_details['year'],
                            similarity=similarity,
                            citations=pub_details['citation_count'],
                            venue=pub_details['venue'],
                            url=pub_details['url']
                        ))
            
            # Create professor result
            results.append(ProfessorResult(
                professor_id=prof_id,
                name=prof_details['name'],
                department=prof_details['department'],
                college=prof_details['college'],
                interests=prof_details['interests'],
                url=prof_details['url'],
                image_url=prof_details.get('image_url'),
                final_score=ranking['final_score'],
                avg_similarity=ranking['avg_similarity'],
                recency_weight=ranking['recency_weight'],
                activity_bonus=ranking['activity_bonus'],
                citation_impact=ranking['citation_impact'],
                num_matching_papers=ranking['num_matching_papers'],
                top_publications=top_pubs
            ))
        
        conn.close()
        
        # Calculate search time
        search_time_ms = (time.time() - start_time) * 1000
        
        return SearchResponse(
            query=request.query,
            corrected_query=corrected_query if corrected_query != request.query.lower() else None,
            results=results,
            total_results=len(results),
            search_time_ms=search_time_ms
        )
    
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Search failed: {str(e)}")


@app.post("/api/bm25/search", response_model=SearchResponse, tags=["Search"])
async def bm25_search(request: SearchRequest):
    """
    Search for papers using BM25 (Lexical Search).
    """
    global bm25_searcher
    start_time = time.time()
    
    if not bm25_searcher:
        raise HTTPException(status_code=503, detail="Service not ready. BM25 not loaded.")
    
    try:
        # Get raw paper results from BM25
        # Request more papers initially to ensure we have enough coverage for grouping
        raw_results = bm25_searcher.search(request.query, top_k=request.top_k * 5)
        
        conn = sqlite3.connect(str(DB_PATH))
        
        # Group papers by professor
        prof_papers = {}
        for paper in raw_results:
            # We need to get the professor ID for this paper
            # The BM25Searcher only returns professor name, so we query the DB
            cursor = conn.cursor()
            cursor.execute("""
                SELECT p.id 
                FROM professors p
                JOIN author_bridge ab ON p.id = ab.professor_id
                WHERE ab.paper_id = ?
                LIMIT 1
            """, (paper['paper_id'],))
            row = cursor.fetchone()
            
            if row:
                prof_id = row[0]
                if prof_id not in prof_papers:
                    prof_papers[prof_id] = []
                prof_papers[prof_id].append(paper)
        
        # Calculate scores and create results
        results = []
        for prof_id, papers in prof_papers.items():
            # Get professor details
            prof_details = get_professor_details(prof_id, conn)
            if not prof_details:
                continue
                
            # Calculate average BM25 score for top papers
            # Sort papers by score descending
            papers.sort(key=lambda x: x['score'], reverse=True)
            top_papers = papers[:3] # Keep top 3 for display
            
            avg_score = sum(p['score'] for p in top_papers) / len(top_papers)
            
            # Create publication summaries
            pub_summaries = []
            for p in top_papers:
                pub_summaries.append(PublicationSummary(
                    paper_id=p['paper_id'],
                    title=p['title'],
                    year=p['year'],
                    similarity=p['score'], # Use BM25 score as "similarity"
                    citations=p['citations'],
                    venue=p['venue'],
                    url=p['url']
                ))
            
            # Create professor result
            # We map BM25 score to "final_score" and "avg_similarity" for compatibility
            results.append(ProfessorResult(
                professor_id=prof_id,
                name=prof_details['name'],
                department=prof_details['department'],
                college=prof_details['college'],
                interests=prof_details['interests'],
                url=prof_details['url'],
                image_url=prof_details.get('image_url'),
                final_score=avg_score,
                avg_similarity=avg_score, # Using BM25 score here
                recency_weight=0.0, # Not applicable for raw BM25
                activity_bonus=0.0,
                citation_impact=0.0,
                num_matching_papers=len(papers),
                top_publications=pub_summaries
            ))
            
        conn.close()
        
        # Sort by score and take top_k
        results.sort(key=lambda x: x.final_score, reverse=True)
        results = results[:request.top_k]
        
        search_time_ms = (time.time() - start_time) * 1000
        
        return SearchResponse(
            query=request.query,
            results=results,
            total_results=len(results),
            search_time_ms=search_time_ms
        )
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"BM25 Search failed: {str(e)}")


@app.get("/api/professor/{professor_id}", response_model=ProfessorDetail, tags=["Professors"])
async def get_professor(professor_id: int):
    """Get detailed information about a specific professor"""
    try:
        conn = sqlite3.connect(str(DB_PATH))
        prof_details = get_professor_details(professor_id, conn)
        conn.close()
        
        if not prof_details:
            raise HTTPException(status_code=404, detail="Professor not found")
        
        return ProfessorDetail(**prof_details)
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to retrieve professor: {str(e)}")


@app.get("/api/publication/{paper_id}", response_model=PublicationDetail, tags=["Publications"])
async def get_publication(paper_id: str):
    """Get detailed information about a specific publication"""
    try:
        conn = sqlite3.connect(str(DB_PATH))
        pub_details = get_publication_details(paper_id, conn)
        conn.close()
        
        if not pub_details:
            raise HTTPException(status_code=404, detail="Publication not found")
        
        return PublicationDetail(**pub_details)
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to retrieve publication: {str(e)}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
