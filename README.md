# AdvisorMatch

A smart tool to help students find the perfect thesis advisor based on their research interests. This system uses AI to understand what you're looking for and matches you with professors who work on similar topics.

## What is AdvisorMatch?

AdvisorMatch is a web application that helps students find thesis advisors at Texas A&M University. Instead of manually searching through hundreds of professors, you can simply describe your research interests and get a ranked list of professors who match your needs.

## How It Works

1. **You describe your research interests** - For example: "machine learning for robotics" or "natural language processing"

2. **The system searches** - It uses AI to understand your query and finds professors whose publications are similar to your interests

3. **You get ranked results** - Professors are ranked based on:
   - How similar their research is to your interests
   - How recent their publications are
   - How active they are in research
   - How much their work is cited by others

## Features

- **Semantic Search**: Uses AI to understand the meaning of your query, not just keywords
- **BM25 Search**: Traditional keyword-based search as an alternative option
- **Smart Ranking**: Considers multiple factors to give you the best matches
- **Publication Details**: See the top matching publications for each professor
- **Easy to Use**: Simple web interface that works in any browser

## Project Structure

```
AdvisorMatch/
├── app/                    # Backend API and data processing
│   ├── api.py             # Main API server
│   ├── ingest.py          # Script to fetch professor data from OpenAlex
│   ├── ranking.py         # Ranking algorithm
│   ├── models.py          # Data models
│   └── config.py          # Configuration settings
├── frontend/              # Web interface
│   ├── index.html         # Main search page (Semantic Search)
│   ├── bm25.html          # BM25 search page
│   ├── css/               # Styling
│   └── js/                # JavaScript code
└── README.md              # This file
```

## Getting Started

### Prerequisites

Before you start, make sure you have:
- Python 3.7 or higher
- A web browser (Chrome, Firefox, Safari, or Edge)

### Installation

1. **Install Python packages**

   Open a terminal in the project folder and run:
   ```bash
   pip install fastapi uvicorn sentence-transformers faiss-cpu numpy sqlite3 requests
   ```

2. **Set up the database**

   The system needs professor and publication data. If you don't have a database yet:
   - Make sure you have a `professors.json` file in the `app/` folder with professor information
   - Run the ingestion script to fetch data from OpenAlex:
     ```bash
     cd app
     python ingest.py
     ```

3. **Generate embeddings and build search index**

   After ingesting data, you need to create the search index:
   ```bash
   cd app
   python generate_embeddings.py
   python build_faiss_index.py
   ```

### Running the Application

1. **Start the backend API**

   Open a terminal and run:
   ```bash
   cd app
   python api.py
   ```
   
   The API will start at: http://localhost:8000

2. **Open the frontend**

   You can open the frontend in two ways:
   
   **Option 1: Direct file opening**
   - Simply open `frontend/index.html` in your web browser
   
   **Option 2: Using a local server** (recommended)
   ```bash
   cd frontend
   python3 -m http.server 3000
   ```
   Then visit: http://localhost:3000

## How to Use

1. Open the web interface in your browser
2. Type your research interests in the search box (e.g., "deep learning for computer vision")
3. Choose how many results you want (1-20)
4. Click "Search" or press Enter
5. Browse the ranked results showing:
   - Professor name and department
   - Match score and breakdown
   - Top matching publications
   - Links to professor profiles

## API Endpoints

The backend provides several API endpoints:

- `GET /` - API information
- `GET /health` - Check if the API is running
- `POST /api/search` - Semantic search for advisors
- `POST /api/bm25/search` - BM25 keyword search
- `GET /api/professor/{id}` - Get detailed professor information
- `GET /api/publication/{id}` - Get detailed publication information

You can view the full API documentation at: http://localhost:8000/docs

## Technologies Used

- **FastAPI**: Backend web framework
- **Sentence-BERT**: AI model for understanding text meaning
- **FAISS**: Fast vector search library
- **OpenAlex**: Academic database for professor publications
- **SQLite**: Database for storing professor and publication data
- **HTML/CSS/JavaScript**: Frontend interface

## Troubleshooting

**Problem: "Search failed" error**
- Make sure the backend API is running (check http://localhost:8000)
- Verify the database and search index files exist in the `app/` folder

**Problem: No results found**
- Try different search terms
- Use more general keywords
- Check that the database has been populated with data

**Problem: API won't start**
- Make sure all Python packages are installed
- Check that the database file exists
- Verify that the FAISS index file exists

