"""
Task 1: Search Engine - Standalone Script
Converted from 'Proper Assignment.ipynb'

Usage:
    python run_task1_search.py

This script allows you to:
1. Crawl the target website
2. Build the Inverted Index
3. Search for documents
"""

import webbrowser
import requests
import time
import json
import re
import math
import sys
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse
from collections import deque, defaultdict
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

# Try importing Selenium for better visual mode
try:
    from selenium import webdriver
    from selenium.webdriver.chrome.options import Options
    SELENIUM_AVAILABLE = True
except ImportError:
    SELENIUM_AVAILABLE = False

# ==========================================
# CONFIGURATION
# ==========================================
SEED_URL = "https://pureportal.coventry.ac.uk/en/organisations/ics-research-centre-for-computational-science-and-mathematical-mo"
BASE_DOMAIN = "pureportal.coventry.ac.uk"
CRAWL_DELAY = 5 
MAX_PAGES = 100  

def setup_nltk():
    """Ensure NLTK data is available"""
    try:
        nltk.data.find('corpora/stopwords')
    except LookupError:
        print("Downloading NLTK stopwords...")
        nltk.download('stopwords', quiet=True)

# ==========================================
# CRAWLER COMPONENTS
# ==========================================
def is_allowed_url(url: str) -> bool:
    """Manual robots.txt compliance"""
    if "format=rss" in url: return False
    if "export=xls" in url: return False
    return True

def crawl_website(max_pages=MAX_PAGES, visual_mode=False):
    print(f"\n[1/3] Starting BFS Crawl (Limit: {max_pages} pages)...")
    
    driver = None
    if visual_mode:
        if SELENIUM_AVAILABLE:
            print("       [VISUAL MODE] Starting Browser (Selenium)...")
            try:
                # Setup minimal options
                opts = Options()
                opts.add_argument("--window-size=1200,800")
                opts.add_experimental_option('excludeSwitches', ['enable-logging'])
                driver = webdriver.Chrome(options=opts)
            except Exception as e:
                print(f"       [WARNING] Failed to start Selenium: {e}")
                print("       Falling back to default browser tabs.")
                driver = None
        else:
             print("       [NOTE] 'selenium' not installed. Opening new tabs for visual mode.")
             print("              Run 'pip install selenium' for single-window experience.")

    queue = deque([SEED_URL])
    visited = set()
    publications = []
    doc_id = 0

    try:
        while queue and len(visited) < max_pages:
            current_url = queue.popleft()

            if current_url in visited:
                continue
            
            # Helper to check robots/domain rules
            if not is_allowed_url(current_url):
                 continue
                 
            parsed = urlparse(current_url)
            if BASE_DOMAIN not in parsed.netloc:
                continue

            print(f"  Crawling: {current_url}")
            visited.add(current_url)
            
            # VISUAL OPERATION
            if visual_mode:
                if driver:
                    try:
                        driver.get(current_url)
                        # We use requests for the actual scraping to keep logic consistent
                        # but we show the user the page in the driver
                    except Exception:
                        pass
                else:
                    webbrowser.open_new_tab(current_url)
                    time.sleep(1)

            try:
                response = requests.get(current_url, timeout=10)
                # Polite Delay
                time.sleep(CRAWL_DELAY)

                if response.status_code != 200:
                    continue
                
                # ... Rest of logic stays same ...
                soup = BeautifulSoup(response.text, "html.parser")

                # Extract Title (Simple Publication Extraction)
                title_tag = soup.find("h1")
                title = title_tag.get_text(strip=True) if title_tag else None

                if title:
                    publications.append({
                        "doc_id": doc_id,
                        "title": title,
                        "url": current_url
                    })
                    doc_id += 1

                # Extract Links
                for link in soup.find_all("a", href=True):
                    absolute_url = urljoin(current_url, link["href"])
                    absolute_url = absolute_url.split("#")[0]

                    if absolute_url not in visited:
                        if BASE_DOMAIN in absolute_url:
                            queue.append(absolute_url)

            except Exception as e:
                print(f"  Error fetching {current_url}: {e}")

    finally:
        if driver:
            print("       Closing browser...")
            driver.quit()

    # Save Results
    with open("publications.json", "w", encoding="utf-8") as f:
        json.dump(publications, f, indent=4, ensure_ascii=False)
    
    print(f"[OK] Crawled {len(visited)} pages, found {len(publications)} documents.")
    print("     Saved to 'publications.json'")
    return publications

# ==========================================
# INDEXER COMPONENTS
# ==========================================
def preprocess(text, stemmer, stop_words):
    text = text.lower()
    tokens = re.findall(r"\b[a-z]+\b", text)
    tokens = [stemmer.stem(t) for t in tokens if t not in stop_words]
    return tokens

def build_index():
    print("\n[2/3] Building Inverted Index...")
    
    try:
        with open("publications.json", "r", encoding="utf-8") as f:
            documents = json.load(f)
    except FileNotFoundError:
        print("[ERROR] publications.json not found. Run crawler first.")
        return None, None

    stop_words = set(stopwords.words("english"))
    stemmer = PorterStemmer()
    
    inverted_index = defaultdict(dict)
    doc_lengths = {}

    for doc in documents:
        doc_id = doc["doc_id"]
        title = doc["title"]
        
        tokens = preprocess(title, stemmer, stop_words)
        doc_lengths[doc_id] = len(tokens)

        for token in tokens:
            # TF Calculation
            if doc_id in inverted_index[token]:
                inverted_index[token][doc_id] += 1
            else:
                inverted_index[token][doc_id] = 1

    # Save Index
    index_data = {
        "inverted_index": dict(inverted_index),
        "doc_lengths": doc_lengths
    }
    
    with open("inverted_index.json", "w", encoding="utf-8") as f:
        json.dump(index_data, f, indent=4)
        
    print(f"[OK] Indexed {len(documents)} documents with {len(inverted_index)} unique terms.")
    print("     Saved to 'inverted_index.json'")
    
    return inverted_index, doc_lengths

# ==========================================
# SEARCH COMPONENTS
# ==========================================
def search_engine(query, top_k=5):
    # Load Data
    try:
        with open("publications.json", "r", encoding="utf-8") as f:
            documents = json.load(f)
        with open("inverted_index.json", "r", encoding="utf-8") as f:
            index_data = json.load(f)
            inverted_index = index_data["inverted_index"]
            doc_lengths = index_data["doc_lengths"]
    except FileNotFoundError:
        print("[ERROR] Data files not found. Please run crawler and indexer first.")
        return

    stop_words = set(stopwords.words("english"))
    stemmer = PorterStemmer()
    
    N = len(documents)
    query_tokens = preprocess(query, stemmer, stop_words)
    scores = {}

    # Calculate Scores
    for term in query_tokens:
        if term not in inverted_index:
            continue

        posting_list = inverted_index[term]
        df = len(posting_list)
        idf = math.log(N / df) if df > 0 else 0

        for doc_id, tf in posting_list.items():
            doc_id = str(doc_id) # JSON keys are strings
            scores[doc_id] = scores.get(doc_id, 0) + (tf * idf)

    # Normalize
    for doc_id in scores:
        length = doc_lengths.get(str(doc_id), 1)
        if length > 0:
            scores[doc_id] /= length

    ranked_docs = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:top_k]
    
    return ranked_docs, documents

# ==========================================
# MAIN MENU
# ==========================================
def main():
    setup_nltk()
    
    print("===========================================")
    print("   TASK 1: SEARCH ENGINE (CLI VERSION)")
    print("===========================================")
    
    while True:
        print("\nOptions:")
        print("1. Crawl Website")
        print("2. Build Index")
        print("3. Search")
        print("4. Exit")
        
        choice = input("\nEnter choice (1-4): ").strip()
        
        if choice == '1':
            vis = input("Enable Visual Mode (open browser tabs)? (y/n): ").lower() == 'y'
            crawl_website(visual_mode=vis)
        elif choice == '2':
            build_index()
        elif choice == '3':
            query = input("Enter search query: ")
            results, docs = search_engine(query)
            if results:
                print(f"\nTop results for '{query}':")
                for rank, (doc_id, score) in enumerate(results, 1):
                    # Docs list is indexed by integer, doc_id from json key is string
                    doc_data = next((d for d in docs if str(d['doc_id']) == str(doc_id)), None)
                    if doc_data:
                        print(f"{rank}. {doc_data['title']}")
                        print(f"   Relevance: {score:.4f}")
                        print(f"   URL: {doc_data['url']}")
            else:
                print("No results found.")
        elif choice == '4':
            print("Goodbye!")
            break
        else:
            print("Invalid choice, try again.")

if __name__ == "__main__":
    main()
