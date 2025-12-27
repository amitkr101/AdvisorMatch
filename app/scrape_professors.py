#!/usr/bin/env python3
"""
Scrape professor names and information from TAMU CSE faculty directory.

This script scrapes the TAMU CSE faculty directory page and extracts:
- Professor names
- Profile URLs
- Research interests (from individual profile pages)

Outputs to professors.json in the format expected by ingest.py

Usage:
    python3 scrape_professors.py [--fetch-interests] [--output OUTPUT_FILE]
"""

import requests
from bs4 import BeautifulSoup
import json
import time
import re
import sys
import argparse
from urllib.parse import urljoin, urlparse
from typing import List, Dict, Optional


BASE_URL = "https://engineering.tamu.edu"
FACULTY_DIRECTORY_URL = "https://engineering.tamu.edu/cse/profiles/index.html#Faculty"
PROFILES_BASE_URL = "https://engineering.tamu.edu/cse/profiles/"


def get_page_content(url: str) -> Optional[BeautifulSoup]:
    """Fetch and parse a web page."""
    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        return BeautifulSoup(response.content, 'html.parser')
    except Exception as e:
        print(f"  [!] Error fetching {url}: {e}")
        return None


def extract_faculty_links(soup: BeautifulSoup) -> List[Dict[str, str]]:
    """
    Extract faculty member links from the directory page.
    
    Returns list of dicts with 'name' and 'url' keys.
    """
    faculty_list = []
    seen_urls = set()
    
    # Method 1: Look for links that point to profile pages
    links = soup.find_all('a', href=True)
    
    for link in links:
        href = link.get('href', '')
        text = link.get_text(strip=True)
        
        # Check if this is a profile link
        if '/cse/profiles/' in href and href.endswith('.html'):
            # Get full URL
            if href.startswith('http'):
                full_url = href
            else:
                full_url = urljoin(BASE_URL, href)
            
            # Skip if we've seen this URL
            if full_url in seen_urls:
                continue
            seen_urls.add(full_url)
            
            # Get name from link text
            name = text.strip()
            
            # If name is empty or too short, try to extract from URL
            if not name or len(name) < 3:
                # Extract from URL: /cse/profiles/name.html -> name
                url_path = urlparse(href).path
                name_match = re.search(r'/([^/]+)\.html$', url_path)
                if name_match:
                    # Convert URL slug to readable name (basic conversion)
                    slug = name_match.group(1)
                    # Try to make it more readable
                    name = slug.replace('-', ' ').title()
                    # Handle special cases like "aklappenecker" -> "A Klappenecker"
                    # This is a simple heuristic - may need refinement
            
            # Clean up name (remove extra whitespace, special chars)
            name = re.sub(r'\s+', ' ', name).strip()
            
            # Only add if we have a reasonable name
            if name and len(name) > 2:
                faculty_list.append({
                    'name': name,
                    'url': full_url
                })
    
    # Method 2: If we didn't find many links, try looking for list items or divs
    if len(faculty_list) < 5:
        print("  Trying alternative extraction methods...")
        
        # Look for list items containing profile links
        for li in soup.find_all(['li', 'div', 'tr']):
            link = li.find('a', href=re.compile(r'/cse/profiles/.*\.html'))
            if link:
                href = link.get('href', '')
                if href.startswith('http'):
                    full_url = href
                else:
                    full_url = urljoin(BASE_URL, href)
                
                if full_url not in seen_urls:
                    seen_urls.add(full_url)
                    name = link.get_text(strip=True) or li.get_text(strip=True)
                    if name and len(name) > 2:
                        faculty_list.append({
                            'name': name.strip(),
                            'url': full_url
                        })
    
    # Remove duplicates and sort by name
    unique_faculty = []
    seen_names = set()
    for faculty in faculty_list:
        # Use URL as primary key, but also check name
        if faculty['url'] not in seen_urls and faculty['name'].lower() not in seen_names:
            seen_urls.add(faculty['url'])
            seen_names.add(faculty['name'].lower())
            unique_faculty.append(faculty)
    
    # Sort by name
    unique_faculty.sort(key=lambda x: x['name'])
    
    return unique_faculty


def extract_interests_from_profile(profile_url: str) -> str:
    """
    Extract research interests from a professor's profile page.
    
    Returns a string of interests, or empty string if not found.
    """
    soup = get_page_content(profile_url)
    if not soup:
        return ""
    
    interests = []
    
    # Look for common patterns for research interests
    # Try different selectors that might contain interests
    
    # Pattern 1: Look for headings containing "Research", "Interests", "Areas"
    headings = soup.find_all(['h1', 'h2', 'h3', 'h4', 'strong', 'b'])
    for heading in headings:
        text = heading.get_text(strip=True).lower()
        if any(keyword in text for keyword in ['research', 'interest', 'area', 'expertise', 'focus']):
            # Get the next sibling or parent content
            next_elem = heading.find_next_sibling()
            if next_elem:
                interests_text = next_elem.get_text(strip=True)
                if interests_text:
                    interests.append(interests_text)
    
    # Pattern 2: Look for divs/sections with class names containing "research", "interest"
    for elem in soup.find_all(['div', 'section', 'p'], class_=re.compile(r'research|interest|expertise', re.I)):
        text = elem.get_text(strip=True)
        if text and len(text) > 10:
            interests.append(text)
    
    # Pattern 3: Look for lists (ul/ol) near research-related headings
    for heading in soup.find_all(['h2', 'h3'], string=re.compile(r'research|interest|area', re.I)):
        next_list = heading.find_next(['ul', 'ol'])
        if next_list:
            items = [li.get_text(strip=True) for li in next_list.find_all('li')]
            interests.extend(items)
    
    # Pattern 4: Generic search for paragraphs containing common research keywords
    paragraphs = soup.find_all('p')
    for p in paragraphs:
        text = p.get_text(strip=True).lower()
        if any(keyword in text for keyword in ['machine learning', 'artificial intelligence', 'computer', 'algorithm', 'system', 'software', 'data']):
            if len(p.get_text(strip=True)) > 20:
                interests.append(p.get_text(strip=True))
    
    # Combine all interests
    combined = '\n'.join(interests[:5])  # Limit to first 5 to avoid too much text
    
    return combined.strip()


def scrape_faculty_directory(fetch_interests: bool = False) -> List[Dict]:
    """
    Scrape the faculty directory and return list of professors.
    
    Args:
        fetch_interests: If True, visit each profile page to get interests
    
    Returns:
        List of professor dictionaries
    """
    print("="*80)
    print("TAMU CSE Faculty Directory Scraper")
    print("="*80)
    print(f"\nFetching faculty directory from: {FACULTY_DIRECTORY_URL}")
    
    soup = get_page_content(FACULTY_DIRECTORY_URL)
    if not soup:
        print("❌ Failed to fetch faculty directory page")
        return []
    
    print("✓ Successfully fetched directory page")
    
    # Extract faculty links
    print("\nExtracting faculty links...")
    faculty_links = extract_faculty_links(soup)
    
    if not faculty_links:
        print("⚠ No faculty links found. The page structure may have changed.")
        print("Trying alternative extraction method...")
        
        # Alternative: Look for any links in the page
        all_links = soup.find_all('a', href=True)
        for link in all_links:
            href = link.get('href', '')
            if 'profiles' in href and '.html' in href:
                full_url = urljoin(BASE_URL, href)
                name = link.get_text(strip=True)
                if name:
                    faculty_links.append({'name': name, 'url': full_url})
    
    print(f"✓ Found {len(faculty_links)} faculty members")
    
    # Build professor list
    professors = []
    
    for i, faculty in enumerate(faculty_links, 1):
        name = faculty['name']
        url = faculty['url']
        
        print(f"\n[{i}/{len(faculty_links)}] Processing: {name}")
        print(f"  URL: {url}")
        
        prof_data = {
            'name': name,
            'college': 'TAMU',
            'dept': 'CSE',
            'interests': '',
            'url': url
        }
        
        # Optionally fetch interests from profile page
        if fetch_interests:
            print("  Fetching research interests...")
            interests = extract_interests_from_profile(url)
            if interests:
                prof_data['interests'] = interests
                print(f"  ✓ Found interests: {interests[:100]}...")
            else:
                print("  - No interests found")
            
            # Be polite - add delay between requests
            time.sleep(0.5)
        
        professors.append(prof_data)
    
    return professors


def main():
    parser = argparse.ArgumentParser(description="Scrape TAMU CSE faculty directory")
    parser.add_argument(
        '--fetch-interests',
        action='store_true',
        help='Visit each profile page to fetch research interests (slower)'
    )
    parser.add_argument(
        '--output',
        default='professors.json',
        help='Output JSON file (default: professors.json)'
    )
    
    args = parser.parse_args()
    
    # Scrape faculty directory
    professors = scrape_faculty_directory(fetch_interests=args.fetch_interests)
    
    if not professors:
        print("\n❌ No professors found. Exiting.")
        sys.exit(1)
    
    # Save to JSON
    output_file = args.output
    print(f"\n{'='*80}")
    print(f"Saving {len(professors)} professors to {output_file}...")
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(professors, f, indent=4, ensure_ascii=False)
    
    print(f"✓ Successfully saved to {output_file}")
    print(f"\nNext steps:")
    print(f"  1. Review {output_file} and add openalex_author_id if known")
    print(f"  2. Run: python3 ingest.py")
    print(f"\n{'='*80}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠ Interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Error: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)

