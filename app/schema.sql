-- Enable Foreign Keys
PRAGMA foreign_keys = ON;

-- 1. Professors Table
-- Stores the faculty members you want to track.
CREATE TABLE IF NOT EXISTS professors (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL UNIQUE,
    college TEXT,
    dept TEXT,
    interests TEXT,
    openalex_author_id TEXT, -- To avoid re-searching the API
    image_url TEXT, -- URL to professor's photo
    embedding BLOB, -- Placeholder for vector embedding (e.g., from bio/interests)
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- 2. Publications Table
-- Stores the actual papers. We use the Semantic Scholar Paper ID as the PK.
CREATE TABLE IF NOT EXISTS publications (
    paper_id TEXT PRIMARY KEY, -- S2 Paper ID
    title TEXT NOT NULL,
    abstract TEXT,
    venue TEXT,
    year INTEGER,
    citation_count INTEGER,
    url TEXT,
    embedding BLOB, -- Placeholder for SPECTER/BERT embedding
    retrieved_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- 3. Author Bridge Table
-- Links Professors to Publications (Many-to-Many).
CREATE TABLE IF NOT EXISTS author_bridge (
    professor_id INTEGER,
    paper_id TEXT,
    is_primary_author BOOLEAN DEFAULT 0, -- Logic can be derived from author position
    author_position INTEGER, -- 1st author, 2nd author, etc.
    PRIMARY KEY (professor_id, paper_id),
    FOREIGN KEY (professor_id) REFERENCES professors(id) ON DELETE CASCADE,
    FOREIGN KEY (paper_id) REFERENCES publications(paper_id) ON DELETE CASCADE
);