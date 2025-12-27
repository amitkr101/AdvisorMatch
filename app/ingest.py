import sqlite3
import json
import time
import requests
import sys

# Configuration
DB_NAME = "advisormatch_openalex.db"
INPUT_FILE = "professors.json"
SCHEMA_FILE = "schema.sql"

# OpenAlex settings
# It is polite to include an email so they can contact you if you break something.
# This gives you access to the "Polite Pool" (faster/higher limits).
HEADERS = {
    "User-Agent": "mailto:test@advisormatch.edu"
}
MAX_PAPERS_PER_PROF = 20  # Increased from 5 to get more comprehensive data

def init_db():
    conn = sqlite3.connect(DB_NAME)
    with open(SCHEMA_FILE, 'r') as f:
        conn.executescript(f.read())
    conn.commit()
    return conn

def search_openalex_author(name):
    """
    Search OpenAlex for an author.
    Filters results to find one affiliated with Texas A&M.
    """
    base_url = "https://api.openalex.org/authors"
    params = {
        "search": name,
        "per_page": 10
    }

    try:
        r = requests.get(base_url, params=params, headers=HEADERS)
        if r.status_code != 200:
            print(f"  [!] API Error: {r.status_code}")
            return None

        results = r.json().get('results', [])

        # Disambiguation logic: Look for 'Texas A&M' in affiliations
        best_match = None

        for author in results:
            affiliations = author.get('affiliations', [])
            # Check last known institution
            for aff in affiliations:
                inst_name = aff.get('institution', {}).get('display_name', '').lower()
                if 'texas a&m' in inst_name or 'tamu' in inst_name:
                    best_match = author
                    break
            if best_match:
                break

        # Fallback: If no affiliation match, take the top result if the name is very similar
        # (Risky, but useful for prototype if data is messy)
        if not best_match and results:
            print(f"  [!] No explicit TAMU match for {name}. Using top result: {results[0]['display_name']} ({results[0]['works_count']} works)")
            best_match = results[0]

        return best_match['id'] if best_match else None

    except Exception as e:
        print(f"  [!] Exception searching author: {e}")
        return None

def get_openalex_works(author_id, limit=MAX_PAPERS_PER_PROF):
    """
    Get works for an author ID with pagination.
    """
    base_url = "https://api.openalex.org/works"
    all_results = []
    per_page = 200 if limit > 200 else limit
    page = 1
    
    while len(all_results) < limit:
        params = {
            "filter": f"author.id:{author_id}",
            "sort": "publication_year:desc",
            "per_page": per_page,
            "page": page
        }

        try:
            r = requests.get(base_url, params=params, headers=HEADERS)
            if r.status_code != 200:
                print(f"  [!] Error {r.status_code} fetching page {page}")
                break

            results = r.json().get('results', [])
            if not results:
                break
                
            all_results.extend(results)
            page += 1
            
            # If we got fewer than requested, we are done
            if len(results) < per_page:
                break
                
        except Exception as e:
            print(f"  [!] Exception fetching works: {e}")
            break
            
    return all_results[:limit]



def extract_author_stats(openalex_author_id, authorships):
    """
    Find position and primary status in the authorship list.
    """
    position = -1
    is_primary = False

    # OpenAlex IDs are full URLs like 'https://openalex.org/A123...'
    # We need to match robustly.
    short_id = openalex_author_id.split('/')[-1]

    for i, auth in enumerate(authorships):
        # auth['author']['id'] is the full URL
        curr_id = auth.get('author', {}).get('id') or ''
        if short_id in curr_id:
            position = i + 1
            if auth.get('author_position') == 'first' or position == 1:
                is_primary = True
            break

    return is_primary, position

def reconstruct_abstract(inverted_index):
    """
    Reconstructs abstract from OpenAlex inverted index.
    """
    if not inverted_index:
        return ""
    
    word_positions = []
    for word, positions in inverted_index.items():
        for pos in positions:
            word_positions.append((pos, word))
    
    word_positions.sort()
    return " ".join(word for _, word in word_positions)

def ingest():
    # Initialize DB schema and get connection
    conn = init_db()
    cursor = conn.cursor()
    
    # Disable foreign key constraints during ingestion
    cursor.execute("PRAGMA foreign_keys = OFF")
    conn.commit()  # Commit the pragma change
    
    with open(INPUT_FILE, 'r') as f:
        professors = json.load(f)

    print(f"Starting ingestion for {len(professors)} professors using OpenAlex...")

    for prof in professors:
        print(f"\nProcessing: {prof['name']}...")

        # 1. Determine Author ID (Manual input OR Search)
        oa_id = prof.get('openalex_author_id')

        if oa_id:
            print(f"  [i] Using provided OpenAlex ID: {oa_id}")
        else:
            print(f"  [?] ID not provided. Searching OpenAlex for {prof['name']}...")
            oa_id = search_openalex_author(prof['name'])

        if not oa_id:
            print(f"  [-] Not found on OpenAlex.")
            continue

        if not prof.get('openalex_author_id'):
            print(f"  [i] Found ID: {oa_id}")

        # 2. Insert Professor
        # Generate placeholder image URL
        import urllib.parse
        encoded_name = urllib.parse.quote(prof['name'])
        image_url = f"https://ui-avatars.com/api/?name={encoded_name}&background=random&color=fff&size=128&format=svg"

        cursor.execute('''
            INSERT OR IGNORE INTO professors (name, college, dept, interests, openalex_author_id, image_url)
            VALUES (?, ?, ?, ?, ?, ?)
        ''', (prof['name'], prof['college'], prof['dept'], prof['interests'], oa_id, image_url))
        
        # Get the ID (whether new or existing)
        cursor.execute('SELECT id FROM professors WHERE name = ?', (prof['name'],))
        prof_db_id = cursor.fetchone()[0]

        # 3. Get Works
        # If affiliation filter is present, fetch MORE papers to ensure we find the right ones
        # amidst the noise of a merged profile.
        limit = MAX_PAPERS_PER_PROF
        if prof.get('affiliation_filter'):
            limit = 1000
            print(f"  [i] Affiliation filter detected. Deep fetching {limit} works...")
        
        works = get_openalex_works(oa_id, limit=limit)
        
        print(f"  [+] Found {len(works)} works.")

        for work in works:
            title = work.get('title')
            paper_id = work.get('id')
            year = work.get('publication_year')
            cited = work.get('cited_by_count')

            # Safe extraction for nested fields (venue and url)
            venue = None
            url = work.get('doi') # Default to DOI

            primary_loc = work.get('primary_location')
            if primary_loc and isinstance(primary_loc, dict):
                # Safe extract venue
                source = primary_loc.get('source')
                if source and isinstance(source, dict):
                    venue = source.get('display_name')

                # Fallback URL if DOI is missing
                if not url:
                    url = primary_loc.get('landing_page_url')

            # Reconstruct abstract
            abstract = reconstruct_abstract(work.get('abstract_inverted_index'))

            # Check affiliation filter if exists
            aff_filter = prof.get('affiliation_filter')
            if aff_filter:
                # Get affiliations for this author in this work
                author_affiliations = []
                for authorship in work.get('authorships', []):
                    if oa_id in authorship.get('author', {}).get('id', ''):
                        for inst in authorship.get('institutions', []):
                            author_affiliations.append(inst.get('display_name', ''))
                
                # Check match
                match = False
                for aff in author_affiliations:
                    for keyword in aff_filter:
                        if keyword.lower() in aff.lower():
                            match = True
                            break
                    if match: break
                
                if not match:
                    print(f"    [x] Skipped '{title[:30]}...' (Affiliation mismatch: {author_affiliations})")
                    # Debug: Print what we were looking for
                    # print(f"        Expected one of: {aff_filter}")
                    continue
                else:
                    print(f"    [v] Kept '{title[:30]}...' (Matched: {author_affiliations})")

            cursor.execute('''
                INSERT OR IGNORE INTO publications 
                (paper_id, title, abstract, venue, year, citation_count, url)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            ''', (paper_id, title, abstract, venue, year, cited, url))

            # 4. Link Bridge
            is_primary, position = extract_author_stats(oa_id, work.get('authorships', []))

            cursor.execute('''
                INSERT OR IGNORE INTO author_bridge 
                (professor_id, paper_id, is_primary_author, author_position)
                VALUES (?, ?, ?, ?)
            ''', (prof_db_id, paper_id, is_primary, position))

        conn.commit()
        time.sleep(0.5) # Be polite

    # Re-enable foreign key constraints
    cursor.execute("PRAGMA foreign_keys = ON")
    conn.close()
    print("\nDone!")

if __name__ == "__main__":
    ingest()