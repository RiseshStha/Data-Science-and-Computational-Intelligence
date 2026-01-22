import re
import math
import requests
import time
import json
from collections import deque, defaultdict
from urllib.parse import urljoin, urlparse
from bs4 import BeautifulSoup
# Unused imports removed

from django.conf import settings
from .models import Document
from datetime import datetime

from utils import TextProcessor, DataHelper


class Crawler:
    SEED_URL = "https://pureportal.coventry.ac.uk/en/organisations/ics-research-centre-for-computational-science-and-mathematical-mo"
    BASE_DOMAIN = "pureportal.coventry.ac.uk"
    CRAWL_DELAY = 5  # Strictly following robots.txt rule
    MAX_PAGES = 100

    def __init__(self):
        self.visited = set()

    def get_soup(self, url):
        try:
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
            }
            response = requests.get(url, headers=headers, timeout=10)
            if response.status_code == 200:
                return BeautifulSoup(response.text, 'html.parser')
        except Exception:
            pass
        return None

    def crawl(self, seed_url=None):
        base_url = seed_url if seed_url else self.SEED_URL
        yield f"Starting crawl from Department: {base_url}...\n"
        
        soup = self.get_soup(base_url)
        if not soup:
            yield "Failed to fetch base URL.\n"
            return

        # 1. Extract Author Links
        author_links = set()
        patterns = [r'/en/persons/[\w-]+', r'/persons/[\w-]+']
        
        for link in soup.find_all('a', href=True):
            href = link['href']
            for pattern in patterns:
                if re.search(pattern, href):
                    full_url = urljoin(base_url, href)
                    if self.BASE_DOMAIN in full_url:
                        author_links.add(full_url)
        
        author_links = list(author_links)[:50] # Limit authors
        yield f"Found {len(author_links)} authors. Starting detailed extraction...\n"

        # 2. Crawl Each Author
        for idx, author_url in enumerate(author_links):
            if idx >= self.MAX_PAGES: break
            
            yield f"Accessing Author [{idx+1}/{len(author_links)}]: {author_url}\n"
            time.sleep(self.CRAWL_DELAY)
            
            author_soup = self.get_soup(author_url)
            if not author_soup: continue

            # Extract Name
            name_elem = (author_soup.find('h1') or author_soup.find('h2') or author_soup.find('span', class_=re.compile('name', re.I)))
            author_name = name_elem.get_text(strip=True) if name_elem else "Unknown Author"

            # Extract Publications
            # PurePortal uses tabbed content or listed articles. We look for 'article' or specific classes.
            pub_containers = (
                author_soup.find_all('article') or 
                author_soup.find_all('div', class_=re.compile('result-container|list-result-item', re.I)) or
                author_soup.find_all('li', class_=re.compile('portal_list_item', re.I))
            )

            count_new = 0
            for container in pub_containers:
                try:
                    # Parse using methodology from notebook
                    title_elem = container.find(['h3', 'h4', 'h2', 'a', 'span'], class_=re.compile('title|link', re.I))
                    
                    # Sometimes the title is just the first link
                    if not title_elem:
                         links = container.find_all('a')
                         if links: title_elem = links[0]

                    if not title_elem: continue
                    title = title_elem.get_text(strip=True)
                    if len(title) < 10: continue

                    link_elem = container.find('a', href=True)
                    pub_url = urljoin(author_url, link_elem['href']) if link_elem else author_url

                    # STRICT FILTERING: Only defined types
                    # Valid: /publications/, /studentTheses/
                    # Invalid: /activities/, /projects/, /clippings/
                    if not any(x in pub_url for x in ['/publications/', '/studentTheses/']):
                        continue
                    
                    if any(x in pub_url for x in ['/activities/', '/projects/', '/persons/', '/organisations/']):
                        continue

                    # Check existence
                    if Document.objects.filter(url=pub_url).exists():
                        continue

                    # Extract Content (Abstract/Year/Keywords)
                    text_content = container.get_text(" ")
                    
                    year_match = re.search(r'(19|20)\d{2}', text_content)
                    year = year_match.group() if year_match else "N/A"

                    # Authors logic (simplified extraction)
                    # Often "By: Name1, Name2..."
                    authors = [author_name] # Default to profile owner

                    Document.objects.create(
                        title=title,
                        url=pub_url,
                        authors=json.dumps(authors),
                        year=year,
                        profile_link=author_url,
                        abstract=text_content[:500], # Store snippet of text as abstract
                        keywords="" 
                    )
                    count_new += 1
                except Exception as e:
                    pass
            
            if count_new > 0:
                yield f"  -> Indexed {count_new} papers for {author_name}\n"
        
        yield "Crawl Finished.\n"

# --- ADVANCED SEARCH ENGINE (Ported logic) ---
class SearchEngine:
    _instance = None

    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def __init__(self):
        # Weighted Index Structure
        self.index = defaultdict(lambda: defaultdict(float)) # token -> {doc_id: score}
        self.documents = {} # doc_id -> Document
        self.N = 0
        self.build_index()

    def build_index(self):
        docs = Document.objects.all()
        self.N = docs.count()
        self.documents = {doc.id: doc for doc in docs}
        # self.index is reset below

        # Weights from notebook
        FIELD_WEIGHTS = {
            'title': 3.0,
            'authors': 2.5,
            'year': 1.5,
            'abstract': 1.0,
            'keywords': 2.0
        }

        # Index Construction
        # Structure: token -> { doc_id: score }
        self.index = defaultdict(lambda: defaultdict(float))

        
        for doc in docs:
            fields = {
                'title': doc.title,
                'authors': doc.authors or "",
                'year': doc.year or "",
                'abstract': doc.abstract or "",
                'keywords': doc.keywords or ""
            }
            
            for field, text in fields.items():
                if not text: continue
                # NEW: preprocess returns list of processed, stemmed tokens
                tokens = TextProcessor.preprocess(text)
                
                weight = FIELD_WEIGHTS.get(field, 1.0)
                
                for token in tokens:
                    self.index[token][doc.id] += (1 * weight)

    def search(self, query):
        if self.N == 0: self.build_index()
        if self.N == 0: return []

        tokens = TextProcessor.preprocess(query)

        if not tokens: return []

        doc_scores = defaultdict(float)
        
        # Calculate Scores
        for token in tokens:
            if token in self.index:
                # IDF
                df = len(self.index[token])
                idf = math.log(self.N / (df + 1)) + 1
                
                for doc_id, weighted_tf in self.index[token].items():
                    doc_scores[doc_id] += weighted_tf * idf

        # Sort
        ranked = sorted(doc_scores.items(), key=lambda x: x[1], reverse=True)[:50]

        results = []
        for doc_id, score in ranked:
            doc = self.documents[doc_id]
            
            # Parse authors json if needed
            authors_str = DataHelper.format_authors(doc.authors)

            results.append({
                "title": doc.title,
                "url": doc.url,
                "score": round(score, 2),
                "authors": authors_str,
                "year": doc.year,
                "abstract": doc.abstract[:200] + "..." if doc.abstract else "",
                "profile_link": doc.profile_link
            })
            
        return results
