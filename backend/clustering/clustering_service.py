"""
PURE CLUSTERING SERVICE (UNSUPERVISED)
Task 2: Document Clustering with Intrinsic Evaluation

This module performs pure unsupervised clustering using K-Means.
It does NOT use any ground-truth labels for training or evaluation.
Evaluation is based on intrinsic metrics:
1. Silhouette Score
2. Davies-Bouldin Index
3. Calinski-Harabasz Index
"""

import pickle
import json
import os
import pandas as pd
import numpy as np
from pathlib import Path
from utils import TextProcessor

# Machine Learning Imports
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import (
    silhouette_score,
    davies_bouldin_score,
    calinski_harabasz_score
)

# Use Agg backend for matplotlib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Import Dataset
try:
    from .data import documents
except ImportError:
    from clustering.data import documents

class ClusteringService:
    """
    Service class for unsupervised document clustering.
    Uses K-Means and intrinsic evaluation metrics.
    """
    
    def __init__(self):
        self.base_dir = Path(__file__).resolve().parent.parent
        self.models_dir = self.base_dir / 'models'
        self.data_dir = self.base_dir / 'data'
        self.static_dir = self.base_dir / 'static'
        
        # Ensure directories exist
        self.models_dir.mkdir(exist_ok=True)
        self.data_dir.mkdir(exist_ok=True)
        self.static_dir.mkdir(exist_ok=True)
        
        self.kmeans_model = None
        self.vectorizer = None
        self.top_terms = {}
        
        self.load_models()
    
    def load_models(self):
        """Load saved models and metadata"""
        try:
            kmeans_path = self.models_dir / 'kmeans_model.pkl'
            if kmeans_path.exists():
                with open(kmeans_path, 'rb') as f:
                    self.kmeans_model = pickle.load(f)
            
            vectorizer_path = self.models_dir / 'tfidf_vectorizer.pkl'
            if vectorizer_path.exists():
                with open(vectorizer_path, 'rb') as f:
                    self.vectorizer = pickle.load(f)
            
            terms_path = self.models_dir / 'cluster_terms.json'
            if terms_path.exists():
                with open(terms_path, 'r') as f:
                    # Convert string keys back to integers
                    data = json.load(f)
                    self.top_terms = {int(k): v for k, v in data.items()}
            
            return True
        except Exception as e:
            print(f"Error loading models: {e}")
            return False

    def preprocess_text(self, text):
        """Standard preprocessing"""
        tokens = TextProcessor.preprocess(text)
        return ' '.join(tokens)
    
    def get_top_keywords(self, data, clusters, vectorizer, n_terms=10):
        """Extract top terms per cluster to interpret them"""
        df = pd.DataFrame(data.toarray(), columns=vectorizer.get_feature_names_out())
        df['cluster'] = clusters
        
        cluster_terms = {}
        for cluster_id in df['cluster'].unique():
            # Get mean tf-idf scores for this cluster
            mean_scores = df[df['cluster'] == cluster_id].drop('cluster', axis=1).mean()
            # Get top N terms
            top_terms = mean_scores.sort_values(ascending=False).head(n_terms).index.tolist()
            cluster_terms[int(cluster_id)] = top_terms
            
        return cluster_terms

    def train_model(self):
        """
        Train K-Means model without using any labels.
        """
        print("[ClusteringService] Starting Pure Unsupervised Training...")
        print("=" * 60)
        
        # 1. Prepare Data
        df = pd.DataFrame(documents)
        df['doc_id'] = range(len(df))
        df['processed_text'] = df['text'].apply(self.preprocess_text)
        
        print(f"[OK] Preprocessed {len(df)} documents")
        
        # 2. Vectorization
        self.vectorizer = TfidfVectorizer(
            max_features=1000,
            stop_words='english',
            ngram_range=(1, 2)
        )
        X = self.vectorizer.fit_transform(df['processed_text'])
        print(f"[OK] TF-IDF Matrix shape: {X.shape}")
        
        # 3. K-Means Clustering
        # Number of clusters set based on prior domain knowledge (Health, Business, Entertainment distinct topics)
        n_clusters = 3
        self.kmeans_model = KMeans(
            n_clusters=n_clusters,
            init='k-means++',
            max_iter=300,
            n_init=10,
            random_state=42
        )
        clusters = self.kmeans_model.fit_predict(X)
        df['cluster'] = clusters
        
        # 4. Interpret Clusters (Get Top Terms)
        print("\n[Cluster Interpretation]")
        self.top_terms = self.get_top_keywords(X, clusters, self.vectorizer)
        for cid, terms in self.top_terms.items():
            print(f"  Cluster {cid}: {', '.join(terms[:5])}...")
            
        # 5. Intrinsic Evaluation (No Labels)
        self._evaluate_model(X, clusters)
        
        # 6. Save Models & Results
        with open(self.models_dir / 'kmeans_model.pkl', 'wb') as f:
            pickle.dump(self.kmeans_model, f)
            
        with open(self.models_dir / 'tfidf_vectorizer.pkl', 'wb') as f:
            pickle.dump(self.vectorizer, f)
            
        with open(self.models_dir / 'cluster_terms.json', 'w') as f:
            json.dump(self.top_terms, f, indent=4)
            
        # Save clustered data - We do NOT save 'category' if trying to be purely unsupervised,
        # but to keep compatibility with parts of the system that might use it for display (if any),
        # we keeps the original dataframe columns but add cluster.
        # We will add a 'predicted_category' column which is just "Cluster X"
        df['predicted_category'] = df['cluster'].apply(lambda x: f"Cluster {x}")
        df.to_csv(self.data_dir / 'clustered_documents.csv', index=False)
        
        # 7. Visualization
        self.generate_visualization(X, df, n_clusters)
        
        print("\n[ClusteringService] [OK] Training Completed.")
        return df

    def _evaluate_model(self, X, labels):
        """
        Calculate intrinsic clustering metrics (No Ground Truth).
        """
        print("\n" + "=" * 60)
        print("INTRINSIC EVALUATION (NO LABELS)")
        print("=" * 60)
        
        # Silhouette Score (ranges from -1 to 1, higher is better)
        sil = silhouette_score(X, labels)
        
        # Davies-Bouldin Index (lower is better)
        db = davies_bouldin_score(X.toarray(), labels)
        
        # Calinski-Harabasz Index (higher is better)
        ch = calinski_harabasz_score(X.toarray(), labels)
        
        # Inertia (WCSS)
        inertia = self.kmeans_model.inertia_
        
        print(f"{'Metric':<30} {'Score':<15} {'Goal'}")
        print("-" * 60)
        print(f"{'Silhouette Score':<30} {sil:<15.4f} Maximize (close to 1)")
        print(f"{'Calinski-Harabasz Index':<30} {ch:<15.4f} Maximize")
        print(f"{'Davies-Bouldin Index':<30} {db:<15.4f} Minimize (close to 0)")
        print(f"{'Inertia (WCSS)':<30} {inertia:<15.4f} Minimize")
        print("=" * 60)
        
        metrics = {
            'silhouette_score': float(sil),
            'davies_bouldin_score': float(db),
            'calinski_harabasz_score': float(ch),
            'inertia': float(inertia),
            'n_clusters': int(self.kmeans_model.n_clusters)
        }
        
        with open(self.data_dir / 'clustering_metrics.json', 'w') as f:
            json.dump(metrics, f, indent=4)

    def generate_visualization(self, X, df, n_clusters):
        """Generate PCA scatter plot of clusters"""
        pca = PCA(n_components=2, random_state=42)
        X_pca = pca.fit_transform(X.toarray())
        
        df['pca1'] = X_pca[:, 0]
        df['pca2'] = X_pca[:, 1]
        
        plt.figure(figsize=(10, 7))
        colors = plt.cm.viridis(np.linspace(0, 1, n_clusters))
        
        for cluster_id in range(n_clusters):
            cluster_data = df[df['cluster'] == cluster_id]
            # Use top terms for label
            top_term = self.top_terms.get(cluster_id, ['Unknown'])[0]
            label = f"Cluster {cluster_id} ({top_term})"
            
            plt.scatter(
                cluster_data['pca1'], 
                cluster_data['pca2'],
                color=colors[cluster_id],
                label=label,
                alpha=0.6, s=100, edgecolors='black'
            )
            
        # Plot centroids
        centers = self.kmeans_model.cluster_centers_
        centers_pca = pca.transform(centers)
        plt.scatter(centers_pca[:, 0], centers_pca[:, 1], c='red', marker='X', 
                   s=200, label='Centroids', edgecolors='white', linewidth=2)
        
        plt.title('Document Clusters (PCA Projection)', fontsize=14)
        plt.xlabel('Principal Component 1')
        plt.ylabel('Principal Component 2')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.savefig(self.static_dir / 'clustering_visualization.png', dpi=300, bbox_inches='tight')
        plt.close()
        print(f"[OK] Visualization saved to {self.static_dir / 'clustering_visualization.png'}")

    def predict_cluster(self, text, return_probabilities=False):
        """
        Predict cluster for new text.
        Returns Cluster ID and distance metrics.
        """
        if not self.kmeans_model or not self.vectorizer:
            if not self.load_models():
                return {'error': 'Models not loaded'}
        
        processed = self.preprocess_text(text)
        vector = self.vectorizer.transform([processed])
        
        cluster_id = int(self.kmeans_model.predict(vector)[0])
        
        # Calculate true distance to centroid (Clustering metric)
        distances = self.kmeans_model.transform(vector)[0]
        distance_to_centroid = float(distances[cluster_id])
        
        # Calculate a similarity score (inverse distance) for UI display purposes
        # This is NOT a probability, just a normalized score where closer = higher
        similarity_score = 1.0 / (1.0 + distance_to_centroid)
        
        # Get interpretation
        top_terms = self.top_terms.get(cluster_id, [])
        cluster_label = f"Cluster {cluster_id}"
        
        result = {
            'cluster': cluster_id,
            'category': cluster_label, # Kept for frontend compatibility
            'cluster_label': cluster_label,
            'top_terms': top_terms[:5],
            'distance_to_centroid': distance_to_centroid,
            'similarity_score': similarity_score # Renamed from confidence
        }
        
        # REMOVED: return_probabilities logic as it is not semantically valid for K-Means
        
        return result

    def get_cluster_statistics(self):
        try:
            metrics_path = self.data_dir / 'clustering_metrics.json'
            if metrics_path.exists():
                with open(metrics_path, 'r') as f:
                    return json.load(f)
            return {}
        except Exception:
            return {}
            
    def get_sample_documents(self, category=None, limit=10):
        try:
            docs_path = self.data_dir / 'clustered_documents.csv'
            if not docs_path.exists(): return []
            df = pd.read_csv(docs_path)
            
            # Filter by 'predicted_category' if 'category' is passed (assuming it's like "Cluster 0")
            if category:
                df = df[df['predicted_category'] == category]
                
            return df[['doc_id', 'text', 'predicted_category', 'cluster']].head(limit).to_dict('records')
        except Exception:
            return []

# Singleton
_clustering_service = None

def get_clustering_service():
    global _clustering_service
    if _clustering_service is None:
        _clustering_service = ClusteringService()
    return _clustering_service
