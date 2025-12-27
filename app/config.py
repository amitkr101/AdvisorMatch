"""
Configuration settings for AdvisorMatch API
"""

import os
from pathlib import Path

# Base paths
BASE_DIR = Path(__file__).parent
DB_PATH = BASE_DIR / "advisormatch_openalex.db"
INDEX_PATH = BASE_DIR / "faiss_index.bin"
MAPPING_PATH = BASE_DIR / "paper_id_mapping.json"

# Model settings
MODEL_NAME = "all-MiniLM-L6-v2"  # 384-dimensional embeddings
EMBEDDING_DIM = 384

# Ranking parameters
TOP_K_PAPERS = 50  # Number of papers to retrieve from FAISS
TOP_N_PER_PROFESSOR = 10  # Number of top papers to consider per professor (increased from 5)
DECAY_RATE = 0.05  # Exponential decay rate for recency weighting (lowered from 0.1 to preserve older high-sim papers)
ACTIVITY_THRESHOLD_YEARS = 2  # Years to consider for activity bonus
ACTIVITY_BONUS_PER_PAPER = 0.05  # Bonus per recent paper
MAX_ACTIVITY_BONUS = 0.2  # Maximum activity bonus cap

# Citation impact parameters
CITATION_WEIGHT = 0.15  # Weight for citation impact in final score
CITATION_LOG_BASE = 10  # Log base for citation normalization

# Authorship parameters
FIRST_AUTHOR_BONUS = 0.2  # Multiplier bonus for first-author papers (20% boost)

# API settings
API_TITLE = "AdvisorMatch API"
API_VERSION = "1.0.0"
API_DESCRIPTION = """
AdvisorMatch API provides semantic search for finding thesis advisors based on research interests.

## Features
- Semantic search using Sentence-BERT embeddings
- Enhanced ranking with recency weighting and activity bonuses
- Professor and publication details retrieval
"""

# CORS settings
CORS_ORIGINS = [
    "http://localhost:3000",  # React default
    "http://localhost:8080",
    "http://127.0.0.1:3000",
    "http://127.0.0.1:8080",
]
