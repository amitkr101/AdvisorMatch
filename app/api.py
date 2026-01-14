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
from fastapi import FastAPI, HTTPException, UploadFile, File
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
    ProfessorDetail, PublicationDetail, HealthResponse,
    UnderstandResponse, AngleRequest, AngleResponse, NextStepsRequest, NextStepsResponse,
    ChatRequest, ChatResponse
)
from ranking import (
    rank_professors, get_professor_details, get_publication_details
)
from spellcheck import DomainSpellChecker
from bm25_search import BM25Searcher
from llm_service import llm_service


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



# --- AI Assistant Endpoints ---

@app.get("/api/assistant/understand/{professor_id}", response_model=UnderstandResponse, tags=["Assistant"])
async def understand_advisor(professor_id: int):
    """
    Mode 1: Understand - Summarize a professor's research trajectory.
    """
    try:
        conn = sqlite3.connect(str(DB_PATH))
        
        # 0. Check Cache (Mode 1 only)
        # Check if we have a cached response younger than CACHE_TTL_HOURS
        from datetime import datetime, timedelta
        from config import CACHE_TTL_HOURS
        import json
        
        cursor = conn.cursor()
        cursor.execute("""
            SELECT response_json, last_updated 
            FROM llm_cache 
            WHERE professor_id = ?
        """, (professor_id,))
        row = cursor.fetchone()
        
        if row:
            response_json, last_updated_str = row
            # Parse timestamp (SQLite default is YYYY-MM-DD HH:MM:SS)
            try:
                last_updated = datetime.strptime(last_updated_str, "%Y-%m-%d %H:%M:%S")
                if datetime.now() - last_updated < timedelta(hours=CACHE_TTL_HOURS):
                    print(f"Returning cached summary for prof {professor_id}")
                    conn.close()
                    return json.loads(response_json)
            except ValueError:
                # If timestamp parsing fails, just ignore cache
                pass

        # 1. Get Professor Name
        prof = get_professor_details(professor_id, conn)
        if not prof:
            conn.close()
            raise HTTPException(status_code=404, detail="Professor not found")
            
        # 2. Get Recent Publications
        cursor.execute("""
            SELECT title, abstract 
            FROM publications p
            JOIN author_bridge ab ON p.paper_id = ab.paper_id
            WHERE ab.professor_id = ?
            ORDER BY p.year DESC
            LIMIT 15
        """, (professor_id,))
        
        rows = cursor.fetchall()
        
        if not rows:
            conn.close()
            raise HTTPException(status_code=404, detail="No publications found for this professor")
            
        abstracts = [f"Title: {row[0]}\nAbstract: {row[1]}" for row in rows if row[1]]
        
        # 3. Call LLM
        llm_response = llm_service.understand_advisor(prof['name'], abstracts)
        
        # 4. Save to Cache
        try:
            # We use REPLACE INTO to upsert
            cursor.execute("""
                INSERT OR REPLACE INTO llm_cache (professor_id, response_json, last_updated)
                VALUES (?, ?, datetime('now', 'localtime'))
            """, (professor_id, json.dumps(llm_response)))
            conn.commit()
        except Exception as e:
            print(f"Failed to cache response: {e}")
            
        conn.close()
        return llm_response

    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/assistant/find-angle", response_model=AngleResponse, tags=["Assistant"])
async def find_research_angle(request: AngleRequest):
    """
    Mode 2: Find My Angle - Suggest alignment between student interests and professor.
    """
    try:
        # 1. Embed Student Interest + Resume (if provided)
        query_text = request.student_interest
        if request.resume_text:
            query_text += f"\n\nResume Context:\n{request.resume_text}"
            
        query_embedding = model.encode([query_text], normalize_embeddings=True)
        query_embedding = query_embedding.astype('float32')[0]
        
        conn = sqlite3.connect(str(DB_PATH))
        
        # 2. Fetch Professor's Papers & Embeddings
        # Since N is small (<200), we fetch all and compute cosine sim in Python
        cursor = conn.cursor()
        cursor.execute("""
            SELECT p.paper_id, p.title, p.abstract, p.embedding
            FROM publications p
            JOIN author_bridge ab ON p.paper_id = ab.paper_id
            WHERE ab.professor_id = ?
            AND p.embedding IS NOT NULL
        """, (request.professor_id,))
        
        rows = cursor.fetchall()
        
        if not rows:
             conn.close()
             raise HTTPException(status_code=404, detail="Professor has no embedded papers")
        
        prof_details = get_professor_details(request.professor_id, conn)
        conn.close()

        # 3. Compute Similarity
        import pickle
        scored_papers = []
        for pid, title, abstract, blob in rows:
            paper_emb = pickle.loads(blob)
            score = np.dot(query_embedding, paper_emb)
            scored_papers.append({
                "paper_id": pid,
                "title": title,
                "abstract": abstract,
                "score": score
            })
            
        # 4. Get Top 5 Context Papers
        scored_papers.sort(key=lambda x: x['score'], reverse=True)
        top_papers = scored_papers[:5]
        
        # 5. Call LLM
        return llm_service.find_research_angle(prof_details['name'], request.student_interest, top_papers)

    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/assistant/next-steps", response_model=NextStepsResponse, tags=["Assistant"])
async def generate_next_steps(request: NextStepsRequest):
    """
    Mode 3: Next Steps - Generate logic checklist.
    """
    try:
        conn = sqlite3.connect(str(DB_PATH))
        prof = get_professor_details(request.professor_id, conn)
        conn.close()
        
        if not prof:
             raise HTTPException(status_code=404, detail="Professor not found")
             
        return llm_service.generate_next_steps(
            prof['name'], 
            request.selected_angle, 
            request.student_level
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/assistant/chat", response_model=ChatResponse, tags=["Assistant"])
async def chat_with_assistant(request: ChatRequest):
    """
    Constrained chat about a professor's research topics.
    """
    try:
        conn = sqlite3.connect(str(DB_PATH))
        
        # 1. Get Professor Details
        prof = get_professor_details(request.professor_id, conn)
        if not prof:
            conn.close()
            raise HTTPException(status_code=404, detail="Professor not found")
            
        # 2. Get Recent Publications for Context
        cursor = conn.cursor()
        cursor.execute("""
            SELECT title, abstract 
            FROM publications p
            JOIN author_bridge ab ON p.paper_id = ab.paper_id
            WHERE ab.professor_id = ?
            ORDER BY p.year DESC
            LIMIT 15
        """, (request.professor_id,))
        
        rows = cursor.fetchall()
        conn.close()
        
        abstracts = [f"Title: {row[0]}\nAbstract: {row[1]}" for row in rows if row[1]]
        
        # 3. Convert history to dicts for LLM Service
        history_dicts = [{"role": msg.role, "content": msg.content} for msg in request.history]
        
        # 4. Call LLM Service
        answer = llm_service.chat_with_professor(prof['name'], abstracts, history_dicts)
        
        return ChatResponse(answer=answer)

    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))



@app.post("/api/parse-resume", tags=["Utils"])
async def parse_resume(file: UploadFile = File(...)):
    """
    Parse uploaded PDF resume and return text.
    """
    if not file.filename.endswith('.pdf'):
        raise HTTPException(status_code=400, detail="Only PDF files are supported")
    
    try:
        import PyPDF2
        import io
        
        # Read file into memory
        content = await file.read()
        pdf_file = io.BytesIO(content)
        
        # Parse PDF
        reader = PyPDF2.PdfReader(pdf_file)
        text = ""
        for page in reader.pages:
            text += page.extract_text() + "\n"
            
        return {"filename": file.filename, "text": text.strip()}
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Failed to parse resume: {str(e)}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)