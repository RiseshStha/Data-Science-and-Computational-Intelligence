
import re
import json
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

class TextProcessor:
    """
    Shared utility for text processing tasks across the application.
    Implements the DRY principle by centralizing NLTK setup and preprocessing logic.
    """
    _stopwords = None
    _stemmer = None

    @classmethod
    def _ensure_resources(cls):
        """
        Ensures NLTK resources are available and loaded.
        """
        if cls._stopwords is None:
            try:
                cls._stopwords = set(stopwords.words('english'))
            except LookupError:
                nltk.download('stopwords', quiet=True)
                cls._stopwords = set(stopwords.words('english'))
        
        if cls._stemmer is None:
            cls._stemmer = PorterStemmer()

    @classmethod
    def preprocess(cls, text, preserve_terms=None):
        """
        Full Pipeline:
        1. Tokenization: Extracts words (a-z)
        2. Stopword Removal
        3. Stemming: Porter Stemmer (skips 'preserve_terms')
        
        Args:
            text (str): Input text
            preserve_terms (set, optional): Set of words to not stem.
            
        Returns:
            list: List of processed, stemmed tokens
        """
        cls._ensure_resources()
        if not text: 
            return []
        
        # Lowercase
        text = str(text).lower()
        
        # Tokenization (only letters)
        tokens = re.findall(r"[a-z]+", text)
        
        processed_tokens = []
        preserve_terms = set(preserve_terms) if preserve_terms else set()
        
        for t in tokens:
            if t not in cls._stopwords and len(t) > 2:
                if t in preserve_terms:
                    processed_tokens.append(t)
                else:
                    processed_tokens.append(cls._stemmer.stem(t))
                    
        return processed_tokens

    @classmethod
    def get_stopwords(cls):
        cls._ensure_resources()
        return cls._stopwords

    @classmethod
    def get_stemmer(cls):
        cls._ensure_resources()
        return cls._stemmer

class DataHelper:
    """
    Helper class for data parsing and formatting.
    """
    @staticmethod
    def parse_authors(authors_data):
        """
        Parses author data which could be a JSON string, a list, or a comma-separated string.
        Returns a list of author names.
        """
        if not authors_data:
            return []
            
        if isinstance(authors_data, list):
            return [str(a).strip() for a in authors_data]
            
        if isinstance(authors_data, str):
            authors_data = authors_data.strip()
            if not authors_data:
                return []
                
            # Try JSON
            try:
                if authors_data.startswith('[') or authors_data.startswith('{'):
                    parsed = json.loads(authors_data)
                    if isinstance(parsed, list):
                        return [str(a).strip() for a in parsed]
                    else:
                        return [str(parsed).strip()]
            except json.JSONDecodeError:
                pass
            
            # Fallback to comma separation
            return [p.strip() for p in authors_data.split(',') if p.strip()]
            
        return [str(authors_data).strip()]

    @staticmethod
    def format_authors(authors_data):
        """
        Returns a comma-separated string of authors.
        """
        authors_list = DataHelper.parse_authors(authors_data)
        return ", ".join(authors_list)
