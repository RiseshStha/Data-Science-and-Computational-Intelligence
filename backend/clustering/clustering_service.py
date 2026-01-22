"""
PURE CLUSTERING SERVICE (UNSUPERVISED WITH EMBEDDINGS)
Task 2: Document Clustering with Intrinsic Evaluation

This version uses SentenceTransformer embeddings to create dense semantic vectors,
then applies K-Means clustering. Predictions and robustness tests will be more accurate.
"""

import pickle
import json
import os
import pandas as pd
import numpy as np
from pathlib import Path

# Machine Learning Imports
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import (
    silhouette_score,
    davies_bouldin_score,
    calinski_harabasz_score
)
from sentence_transformers import SentenceTransformer

# Use Agg backend for matplotlib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Import Dataset
try:
    from .data import documents
except ImportError:
    from clustering.data import documents

class ClusteringServiceEmbeddings:
    """Service class for unsupervised document clustering using semantic embeddings."""

    def __init__(self):
        self.base_dir = Path(__file__).resolve().parent.parent
        self.models_dir = self.base_dir / 'models'
        self.data_dir = self.base_dir / 'data'
        self.static_dir = self.base_dir / 'static'

        self.models_dir.mkdir(exist_ok=True)
        self.data_dir.mkdir(exist_ok=True)
        self.static_dir.mkdir(exist_ok=True)

        self.kmeans_model = None
        self.embedding_model = None
        self.embeddings = None
        self.top_terms = {}

        self.load_models()

    def load_models(self):
        """Load saved K-Means model and embeddings model if available"""
        try:
            kmeans_path = self.models_dir / 'kmeans_model.pkl'
            if kmeans_path.exists():
                with open(kmeans_path, 'rb') as f:
                    self.kmeans_model = pickle.load(f)

            self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
            
            # Load top terms if available
            terms_path = self.models_dir / 'cluster_terms.json'
            if terms_path.exists():
                with open(terms_path, 'r') as f:
                    data = json.load(f)
                    self.top_terms = {int(k): v for k, v in data.items()}
            
            return True
        except Exception as e:
            print(f"Error loading models: {e}")
            return False

    def preprocess_text(self, text: str) -> str:
        """Basic preprocessing: lowercase and strip"""
        return text.lower().strip()

    def get_top_keywords(self, df: pd.DataFrame, clusters: np.ndarray, n_terms=10):
        """
        Extract representative keywords for each cluster
        Using TF-IDF over the documents assigned to each cluster
        """
        from sklearn.feature_extraction.text import TfidfVectorizer
        cluster_terms = {}
        for cid in np.unique(clusters):
            # Get docs for this cluster
            cluster_docs = df[df['cluster'] == cid]['text'].tolist()
            if not cluster_docs: 
                continue
                
            # Use TF-IDF to find distinctive words for this cluster
            vectorizer = TfidfVectorizer(stop_words='english', max_features=500)
            try:
                X = vectorizer.fit_transform(cluster_docs)
                mean_scores = X.mean(axis=0).A1  # flatten
                terms = np.array(vectorizer.get_feature_names_out())
                # Get indices of top scores
                top_indices = mean_scores.argsort()[::-1][:n_terms]
                top_terms = terms[top_indices].tolist()
                cluster_terms[int(cid)] = top_terms
            except ValueError:
                # Handle cases with empty vocabulary
                cluster_terms[int(cid)] = []
                
        return cluster_terms

    def train_model(self, n_clusters=3):
        """Train K-Means on semantic embeddings"""
        print("[ClusteringService] Starting semantic embedding clustering...")
        print("=" * 60)

        df = pd.DataFrame(documents)
        df['doc_id'] = range(len(df))
        df['processed_text'] = df['text'].apply(self.preprocess_text)
        print(f"[OK] Preprocessed {len(df)} documents")

        # 1️ Create embeddings
        print("[INFO] Generating sentence embeddings (Dataset size: {0})...".format(len(df)))
        # Load model explicitly here to ensure it's ready for training
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        self.embeddings = self.embedding_model.encode(df['processed_text'].tolist(), show_progress_bar=True)

        # 2️ K-Means clustering
        self.kmeans_model = KMeans(
            n_clusters=n_clusters,
            init='k-means++',
            max_iter=300,
            n_init=10,
            random_state=42
        )
        clusters = self.kmeans_model.fit_predict(self.embeddings)
        df['cluster'] = clusters

        # 3️ Extract top keywords for interpretation
        print("\n[Cluster Interpretation]")
        self.top_terms = self.get_top_keywords(df, clusters)
        for cid, terms in self.top_terms.items():
            print(f"  Cluster {cid}: {', '.join(terms[:5])}...")

        # 4️ Intrinsic evaluation
        self._evaluate_model(clusters)

        # 5️ Save models
        with open(self.models_dir / 'kmeans_model.pkl', 'wb') as f:
            pickle.dump(self.kmeans_model, f)
            
        with open(self.models_dir / 'cluster_terms.json', 'w') as f:
            json.dump(self.top_terms, f, indent=4)

        df['predicted_category'] = df['cluster'].apply(lambda x: f"Cluster {x}")
        df.to_csv(self.data_dir / 'clustered_documents.csv', index=False)

        # 6️ Visualization
        self.generate_visualization(df, n_clusters)
        print("\n[ClusteringService] [OK] Training Completed.")
        return df

    def _evaluate_model(self, clusters: np.ndarray):
        print("\n" + "=" * 60)
        print("INTRINSIC EVALUATION (NO LABELS)")
        print("=" * 60)

        X = self.embeddings
        sil = silhouette_score(X, clusters)
        db = davies_bouldin_score(X, clusters)
        ch = calinski_harabasz_score(X, clusters)
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

    def generate_visualization(self, df: pd.DataFrame, n_clusters: int):
        """PCA scatter plot of embeddings"""
        pca = PCA(n_components=2, random_state=42)
        X_pca = pca.fit_transform(self.embeddings)

        df['pca1'] = X_pca[:, 0]
        df['pca2'] = X_pca[:, 1]

        plt.figure(figsize=(10, 7))
        colors = plt.cm.viridis(np.linspace(0, 1, n_clusters))
        for cluster_id in range(n_clusters):
            cluster_data = df[df['cluster'] == cluster_id]
            top_term = self.top_terms.get(cluster_id, ['Unknown'])[0]
            label = f"Cluster {cluster_id} ({top_term})"
            plt.scatter(
                cluster_data['pca1'],
                cluster_data['pca2'],
                color=colors[cluster_id],
                label=label,
                alpha=0.6, s=100, edgecolors='black'
            )

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

    def predict_cluster(self, text: str):
        """Assign a new document to an existing cluster"""
        if not self.kmeans_model or not self.embedding_model:
            if not self.load_models():
                return {'error': 'Models not loaded'}

        processed = self.preprocess_text(text)
        vector = self.embedding_model.encode([processed])
        cluster_id = int(self.kmeans_model.predict(vector)[0])

        distances = self.kmeans_model.transform(vector)[0]
        distance_to_centroid = float(distances[cluster_id])
        similarity_score = 1.0 / (1.0 + distance_to_centroid)
        top_terms = self.top_terms.get(cluster_id, [])

        result = {
            'cluster': cluster_id,
            'category': f"Cluster {cluster_id}",
            'cluster_label': f"Cluster {cluster_id}",
            'top_terms': top_terms[:5],
            'distance_to_centroid': distance_to_centroid,
            'similarity_score': similarity_score
        }
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
        _clustering_service = ClusteringServiceEmbeddings()
    return _clustering_service
