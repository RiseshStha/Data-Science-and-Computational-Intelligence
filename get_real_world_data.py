import requests
from bs4 import BeautifulSoup
import feedparser
import pandas as pd
import time
from datetime import datetime
import json


# Multi-category RSS feed scraper for Health, Business, Entertainment
# Collects ~150 documents (50 per category) in your exact format
def scrape_rss_category(category_feeds, category_name, max_articles=100):
    """Scrape multiple RSS feeds for one category"""
    all_articles = []

    for feed_url, source in category_feeds:
        try:
            print(f"Scraping {source} {category_name}...")
            feed = feedparser.parse(feed_url)

            for entry in feed.entries[:max_articles//len(category_feeds)]:
                # Combine title and summary/description
                raw_text = entry.get('title', '') + '. ' + (entry.get('summary') or entry.get('description', ''))

                # Remove HTML tags and extra whitespace
                text = BeautifulSoup(raw_text, 'html.parser').get_text()
                text = ' '.join(text.split())  # normalize whitespace

                # Limit to first 500 words
                text = ' '.join(text.split()[:1000])

                # Limit to first 3 sentences
                sentences = text.split('.')
                if len(sentences) > 9:
                    text = '. '.join(sentences[:5]) + '.'

                article = {
                    "text": text.strip(),
                    "category": category_name,
                    "source": source
                }

                all_articles.append(article)
                if len(all_articles) >= max_articles:
                    break

            time.sleep(1)  # Rate limiting

        except Exception as e:
            print(f"Error scraping {source}: {e}")
            continue

    return all_articles[:max_articles]


# RSS FEEDS CONFIGURATION (All public domain/usage allowed for academic)
RSS_FEEDS = {
    "Health": [
        ("http://feeds.bbci.co.uk/news/health/rss.xml", "BBC News"),
        ("https://www.reuters.com/arc/outboundfeeds/news-rss/?category=health", "Reuters"),
        ("https://www.theguardian.com/society/healthcare/rss", "The Guardian"),
        ("https://www.who.int/rss-feeds/", "World Health Organization"),
    ],
    "Business": [
        ("http://feeds.bbci.co.uk/news/business/rss.xml", "BBC News"),
        ("https://www.reuters.com/arc/outboundfeeds/news-rss/?category=business", "Reuters"),
        ("https://www.theguardian.com/uk/business/rss", "The Guardian"),
        ("http://rss.cnn.com/rss/money_latest.rss", "CNN Business"),
    ],
    "Entertainment": [
        ("http://feeds.bbci.co.uk/news/entertainment-arts/rss.xml", "BBC News"),
        ("https://www.reuters.com/arc/outboundfeeds/news-rss/?category=entertainment", "Reuters"),
        ("https://www.theguardian.com/film/rss", "The Guardian"),
        ("https://www.theguardian.com/music/rss", "The Guardian"),
    ]
}


def main():
    print("Scraping REAL news articles for 3 categories...")
    all_documents = []

    for category, feeds in RSS_FEEDS.items():
        print(f"\nProcessing {category} category...")
        category_articles = scrape_rss_category(feeds, category, max_articles=200)
        all_documents.extend(category_articles)
        print(f"   Got {len(category_articles)} {category} articles")

    # FIXED: Take EXACTLY 50 from EACH category (total 150)
    documents_list = []
    for category in ["Health", "Business", "Entertainment"]:
        cat_docs = [doc for doc in all_documents if doc['category'] == category][:50]
        documents_list.extend(cat_docs)
        print(f"Kept 50 {category} articles")

    # Python list format (your requirement)
    output = f'documents = {json.dumps(documents_list, indent=4)}'

    # Save files
    with open('data.py', 'w', encoding='utf-8') as f:
        f.write(output)

    print(f"SUCCESS!")
    print(f"Total articles: {len(documents_list)} (50 per category)")
    print(f"Files saved: data.py")

    return documents_list


if __name__ == "__main__":
    documents = main()
