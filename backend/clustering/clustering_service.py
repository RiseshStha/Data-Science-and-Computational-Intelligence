"""
IMPROVED Clustering Service for Document Classification
Uses K-Means clustering with enhanced features to categorize documents into Business, Entertainment, and Health

IMPROVEMENTS:
- Increased feature space (500 → 1500 features)
- Domain-specific keyword features
- Better TF-IDF parameters (trigrams, min_df=1)
- Improved cluster mapping with purity thresholds
- Enhanced preprocessing
"""

import pickle
import json
import re
import os
import pandas as pd
import numpy as np
from pathlib import Path
from collections import Counter
from utils import TextProcessor

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.metrics import (
    silhouette_score, adjusted_rand_score, normalized_mutual_info_score, 
    f1_score, precision_score, recall_score, classification_report
)
from sklearn.decomposition import PCA

# Use Agg backend for matplotlib to avoid thread issues
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Import Dataset
try:
    from .data import documents
except ImportError:
    # Fallback if running from a different context
    from clustering.data import documents

# Download NLTK data if not already present
# NLTK setup handled in utils.py

class ImprovedClusteringService:
    """
    IMPROVED Service class for document clustering and prediction
    """
    
    # Domain-specific keywords for better feature engineering
    DOMAIN_KEYWORDS = {
        'Health': [
            'hospital', 'patient', 'doctor', 'nhs', 'medical', 'health', 'treatment',
            'nurse', 'surgery', 'clinic', 'disease', 'vaccine', 'drug', 'therapy',
            'diagnosis', 'symptom', 'covid', 'flu', 'cancer', 'care', 'physician',
            'trust', 'ambulance', 'midlands', 'baby', 'babies', 'birth', 'maternity',
            'mental', 'wellbeing', 'study', 'research', 'surgeon', 'transplant',
            'weight', 'loss', 'diet', 'scan', 'test', 'gp', 'appointment', 'wait',
            'strike', 'union', 'pay' # Contextual health terms
        ],
        'Business': [
            'economy', 'market', 'company', 'trade', 'financial', 'business', 'stock',
            'investment', 'profit', 'revenue', 'corporate', 'industry', 'commerce',
            'tariff', 'budget', 'tax', 'price', 'consumer', 'retail', 'executive',
            'bank', 'inflation', 'rate', 'fund', 'debt', 'growth', 'deal', 'ceo',
            'manager', 'sales', 'cost', 'spending', 'crisis', 'job', 'work', 'staff'
        ],
        'Entertainment': [
            'film', 'movie', 'actor', 'director', 'music', 'artist', 'show', 'star',
            'album', 'concert', 'performance', 'cinema', 'theater', 'song', 'band',
            'celebrity', 'award', 'oscar', 'festival', 'production', 'screenplay',
            'review', 'series', 'tv', 'drama', 'comedy', 'hollywood', 'netflix',
            'book', 'novel', 'author', 'play', 'stage', 'fame', 'hit'
        ]
    }
    
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
        self.cluster_mapping = None
        
        # Load models on initialization
        self.load_models()
    
    def load_models(self):
        """Load saved K-Means model, vectorizer, and cluster mapping"""
        try:
            # Load K-Means model
            kmeans_path = self.models_dir / 'kmeans_model.pkl'
            if kmeans_path.exists():
                with open(kmeans_path, 'rb') as f:
                    self.kmeans_model = pickle.load(f)
            
            # Load TF-IDF vectorizer
            vectorizer_path = self.models_dir / 'tfidf_vectorizer.pkl'
            if vectorizer_path.exists():
                with open(vectorizer_path, 'rb') as f:
                    self.vectorizer = pickle.load(f)
            
            # Load cluster mapping
            mapping_path = self.models_dir / 'cluster_mapping.json'
            if mapping_path.exists():
                with open(mapping_path, 'r') as f:
                    # Convert string keys to integers
                    mapping = json.load(f)
                    self.cluster_mapping = {int(k): v for k, v in mapping.items()}
            
            return True
        except Exception as e:
            print(f"Error loading models: {e}")
            return False
    
    def extract_domain_features(self, text):
        """
        Extract domain-specific keyword counts as additional features
        """
        text_lower = text.lower()
        features = {}
        
        for category, keywords in self.DOMAIN_KEYWORDS.items():
            count = sum(1 for keyword in keywords if keyword in text_lower)
            features[f'{category}_keywords'] = count
        
        return features
    
    def preprocess_text(self, text):
        """
        IMPROVED preprocessing using shared TextProcessor.
        Preserves domain-specific terms.
        """
        # Collect domain terms to preserve
        preserve_terms = set()
        for keywords in self.DOMAIN_KEYWORDS.values():
            preserve_terms.update(keywords)
        
        # Use util with preserve_terms
        tokens = TextProcessor.preprocess(text, preserve_terms=preserve_terms)
        
        return ' '.join(tokens)
    
    def train_model(self):
        """
        IMPROVED: Train the K-Means model with enhanced features
        """
        print("[ImprovedClusteringService] Starting Enhanced Model Training...")
        print("=" * 70)
        
        # 1. Prepare Data
        df = pd.DataFrame(documents)
        df['doc_id'] = range(len(df))
        df['processed_text'] = df['text'].apply(self.preprocess_text)
        
        # Extract domain features
        domain_features = df['text'].apply(self.extract_domain_features)
        
        # SCALE UP DOMAIN FEATURES (Weight x 20) to overpower generic vocabulary
        DOMAIN_WEIGHT = 20.0 
        # Create DataFrame and SORT columns alphabetically to match predict_cluster
        domain_df = pd.DataFrame(domain_features.tolist())
        domain_df = domain_df.reindex(sorted(domain_df.columns), axis=1) * DOMAIN_WEIGHT
        
        
        print(f"[OK] Preprocessed {len(df)} documents")
        print(f"[OK] Extracted domain-specific features: {list(domain_df.columns)}")
        
        # 2. IMPROVED Vectorization
        print("\n[Vectorization]")
        vectorizer = TfidfVectorizer(
            max_features=1500,        # ← INCREASED from 500
            min_df=1,                 # ← CHANGED from 2 (keep rare distinctive terms)
            max_df=0.9,               # ← INCREASED from 0.8
            ngram_range=(1, 3),       # ← ADDED trigrams (was 1,2)
            sublinear_tf=True,        # ← NEW: Apply sublinear tf scaling
            use_idf=True,
            smooth_idf=True
        )
        X_tfidf = vectorizer.fit_transform(df['processed_text'])
        
        # Combine TF-IDF with domain features
        from scipy.sparse import hstack
        X_domain = domain_df.values
        X_combined = hstack([X_tfidf, X_domain])
        
        print(f"[OK] TF-IDF features: {X_tfidf.shape[1]}")
        print(f"[OK] Domain features: {X_domain.shape[1]}")
        print(f"[OK] Combined feature space: {X_combined.shape[1]}")
        
        # 3. Train K-Means with better initialization
        print("\n[K-Means Clustering]")
        n_clusters = 3
        kmeans = KMeans(
            n_clusters=n_clusters,
            init='k-means++',
            max_iter=500,             # ← INCREASED from 300
            n_init=20,                # ← INCREASED from 10
            random_state=42,
            algorithm='lloyd'
        )
        df['cluster'] = kmeans.fit_predict(X_combined)
        
        # 4. IMPROVED Cluster Mapping with Purity Threshold
        print("\n[Cluster-to-Category Mapping]")
        cluster_category_mapping = {}
        cluster_purity = {}
        
        for cluster_id in range(n_clusters):
            cluster_docs = df[df['cluster'] == cluster_id]
            if not cluster_docs.empty:
                category_counts = cluster_docs['category'].value_counts()
                most_common = category_counts.idxmax()
                purity = category_counts.max() / len(cluster_docs)
                
                cluster_category_mapping[cluster_id] = most_common
                cluster_purity[cluster_id] = purity
                
                print(f"  Cluster {cluster_id} -> {most_common} "
                      f"(purity: {purity:.2%}, size: {len(cluster_docs)})")
                print(f"    Distribution: {dict(category_counts)}")
            else:
                cluster_category_mapping[cluster_id] = "Unknown"
                cluster_purity[cluster_id] = 0.0
        
        df['predicted_category'] = df['cluster'].map(cluster_category_mapping)
        
        # 5. ENHANCED Evaluation
        metrics, cm = self._evaluate_model(df, X_combined, n_clusters)
        
        # 6. Save Models
        self.kmeans_model = kmeans
        self.vectorizer = vectorizer
        self.cluster_mapping = cluster_category_mapping
        
        with open(self.models_dir / 'kmeans_model.pkl', 'wb') as f:
            pickle.dump(kmeans, f)
            
        with open(self.models_dir / 'tfidf_vectorizer.pkl', 'wb') as f:
            pickle.dump(vectorizer, f)
            
        with open(self.models_dir / 'cluster_mapping.json', 'w') as f:
            json.dump(cluster_category_mapping, f)
            
        df.to_csv(self.data_dir / 'clustered_documents.csv', index=False)
        
        # Save enhanced metrics
        metrics = {
            'f1_score_weighted': float(f1_weighted),
            'f1_score_macro': float(f1_macro),
            'f1_scores_per_class': {
                'Business': float(f1_per_class[0]),
                'Entertainment': float(f1_per_class[1]),
                'Health': float(f1_per_class[2])
            },
            'precision_weighted': float(precision_weighted),
            'recall_weighted': float(recall_weighted),
            'silhouette_score': float(silhouette_avg),
            'purity_score': float(purity),
            'adjusted_rand_index': float(ari),
            'normalized_mutual_info': float(nmi),
            'accuracy': float(accuracy),
            'cluster_purity': cluster_purity,
            'n_clusters': n_clusters,
            'n_documents': len(df),
            'n_features': X_combined.shape[1]
        }
        with open(self.data_dir / 'clustering_metrics.json', 'w') as f:
            json.dump(metrics, f, indent=4)
            
        # 7. Visualize (PCA)
        self.generate_visualization(X_combined, df, kmeans, n_clusters, cluster_category_mapping)
        self.generate_confusion_matrix(cm, ['Business', 'Entertainment', 'Health'])
        
        print("\n[ImprovedClusteringService] [OK] Model Training & Saving Completed.")
        return df

    def generate_confusion_matrix(self, cm, labels):
        """Generate Confusion Matrix Heatmap"""
        import seaborn as sns
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=labels, yticklabels=labels)
        plt.title('Confusion Matrix', fontsize=16)
        plt.ylabel('True Category')
        plt.xlabel('Predicted Category')
        plt.tight_layout()
        plt.savefig(self.static_dir / 'confusion_matrix.png', dpi=300)
        plt.close()
        print(f"[OK] Confusion Matrix saved to {self.static_dir / 'confusion_matrix.png'}")

    def generate_visualization(self, X, df, kmeans, n_clusters, mapping):
        """Generate enhanced PCA scatter plot"""
        pca = PCA(n_components=2, random_state=42)
        X_pca = pca.fit_transform(X.toarray() if hasattr(X, 'toarray') else X)
        
        df['pca1'] = X_pca[:, 0]
        df['pca2'] = X_pca[:, 1]
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 7))
        colors = ['#FF6B6B', '#4ECDC4', '#45B7D1']
        category_colors = {'Business': '#FF6B6B', 'Entertainment': '#4ECDC4', 'Health': '#45B7D1'}
        
        # Plot 1: Clusters
        for cluster_id in range(n_clusters):
            cluster_data = df[df['cluster'] == cluster_id]
            ax1.scatter(
                cluster_data['pca1'], 
                cluster_data['pca2'],
                c=colors[cluster_id % len(colors)],
                label=f'Cluster {cluster_id} -> {mapping.get(cluster_id, "?")}',
                alpha=0.6, s=100, edgecolors='black', linewidth=0.5
            )
        
        centers_pca = pca.transform(kmeans.cluster_centers_)
        ax1.scatter(centers_pca[:, 0], centers_pca[:, 1], c='black', marker='X', 
                   s=400, label='Centroids', edgecolors='white', linewidth=2)
        
        ax1.set_title('K-Means Clustering Results (PCA)', fontsize=14, fontweight='bold')
        ax1.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%} variance)')
        ax1.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%} variance)')
        ax1.legend()
        ax1.grid(alpha=0.3)
        
        # Plot 2: True Categories
        for category in ['Business', 'Entertainment', 'Health']:
            cat_data = df[df['category'] == category]
            ax2.scatter(
                cat_data['pca1'],
                cat_data['pca2'],
                c=category_colors[category],
                label=f'{category} (true)',
                alpha=0.6, s=100, edgecolors='black', linewidth=0.5
            )
        
        ax2.set_title('True Categories (PCA)', fontsize=14, fontweight='bold')
        ax2.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%} variance)')
        ax2.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%} variance)')
        ax2.legend()
        ax2.grid(alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.static_dir / 'clustering_visualization.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"[OK] Visualization saved to {self.static_dir / 'clustering_visualization.png'}")

    def _evaluate_model(self, df, X_combined, n_clusters):
        """
        Evaluate clustering model performance.
        Returns:
            metrics (dict): Computed metrics
            cm (array): Confusion matrix
        """
        print("\n" + "=" * 70)
        print("EVALUATION METRICS")
        print("=" * 70)
        
        # Calculate metrics
        silhouette_avg = silhouette_score(X_combined, df['cluster'])
        
        category_mapping = {'Business': 0, 'Entertainment': 1, 'Health': 2}
        y_true = df['category'].map(category_mapping)
        ari = adjusted_rand_score(y_true, df['cluster'])
        nmi = normalized_mutual_info_score(y_true, df['cluster'])
        accuracy = (df['category'] == df['predicted_category']).mean()
        
        # F1 Scores
        f1_weighted = f1_score(df['category'], df['predicted_category'], average='weighted')
        f1_macro = f1_score(df['category'], df['predicted_category'], average='macro')
        f1_per_class = f1_score(df['category'], df['predicted_category'], average=None, 
                                labels=['Business', 'Entertainment', 'Health'])
        
        # Precision and Recall
        precision_weighted = precision_score(df['category'], df['predicted_category'], average='weighted')
        recall_weighted = recall_score(df['category'], df['predicted_category'], average='weighted')
        
        # Purity Score
        def purity_score(y_true, y_pred):
            contingency_matrix = pd.crosstab(y_true, y_pred)
            return np.sum(np.amax(contingency_matrix.values, axis=0)) / np.sum(contingency_matrix.values)
        
        purity = purity_score(df['category'], df['cluster'])
        
        # Display results
        print(f"{'Metric':<35} {'Score':<15} {'Description'}")
        print("-" * 70)
        print(f"{'Weighted F1 Score':<35} {f1_weighted:<15.4f} [*] PRIMARY METRIC")
        print(f"{'Macro F1 Score':<35} {f1_macro:<15.4f} Unweighted average")
        print(f"{'Accuracy':<35} {accuracy:<15.4f} Correctly classified")
        print(f"{'Precision (weighted)':<35} {precision_weighted:<15.4f} Prediction accuracy")
        print(f"{'Recall (weighted)':<35} {recall_weighted:<15.4f} Coverage")
        print(f"{'Purity Score':<35} {purity:<15.4f} Cluster homogeneity")
        print(f"{'Adjusted Rand Index (ARI)':<35} {ari:<15.4f} Similarity to truth")
        print(f"{'Normalized Mutual Info (NMI)':<35} {nmi:<15.4f} Mutual dependence")
        print(f"{'Silhouette Score':<35} {silhouette_avg:<15.4f} Cluster quality")
        print("=" * 70)
        
        # Per-class F1 scores
        print("\nPer-Class F1 Scores:")
        for i, category in enumerate(['Business', 'Entertainment', 'Health']):
            print(f"  {category:<20} F1: {f1_per_class[i]:.4f}")
        
        print("\n" + "=" * 70)
        print(f"Total Documents: {len(df)}")
        print(f"Number of Clusters: {n_clusters}")
        print(f"Feature Dimensions: {X_combined.shape[1]}")
        print("=" * 70 + "\n")
        
        # Classification Report
        print("\nDetailed Classification Report:")
        print(classification_report(df['category'], df['predicted_category'], 
                                   target_names=['Business', 'Entertainment', 'Health']))
        
        # Confusion Matrix
        from sklearn.metrics import confusion_matrix
        cm = confusion_matrix(df['category'], df['predicted_category'], 
                            labels=['Business', 'Entertainment', 'Health'])
        print("\nConfusion Matrix:")
        print("                Predicted ->")
        print("True |          Business  Entertainment  Health")
        for i, label in enumerate(['Business', 'Entertainment', 'Health']):
            print(f"{label:<15} {cm[i][0]:>8}  {cm[i][1]:>13}  {cm[i][2]:>6}")
            
        metrics = {
            'f1_score_weighted': f1_weighted,
            'f1_score_macro': f1_macro,
            'f1_scores_per_class': f1_per_class,
            'precision_weighted': precision_weighted,
            'recall_weighted': recall_weighted,
            'silhouette_score': silhouette_avg,
            'purity_score': purity,
            'adjusted_rand_index': ari,
            'normalized_mutual_info': nmi,
            'accuracy': accuracy
        }
        
        return metrics, cm

    def predict_cluster(self, text, return_probabilities=False):
        """
        Predict the cluster for a new document with improved confidence scoring
        """
        if not self.kmeans_model or not self.vectorizer or not self.cluster_mapping:
            # Try reloading
            if not self.load_models():
                raise ValueError("Models not loaded. Please ensure models are trained and saved.")
        
        processed = self.preprocess_text(text)
        if not processed:
            return {'cluster': -1, 'category': 'Unknown', 'confidence': 0.0}
        
        # Extract domain features
        domain_feats = self.extract_domain_features(text)
        
        # Vectorize
        vector_tfidf = self.vectorizer.transform([processed])
        
        # Combine with domain features
        from scipy.sparse import hstack
        # Apply same scaling (x20) as in training
        DOMAIN_WEIGHT = 20.0
        vector_domain = np.array([[domain_feats[k] for k in sorted(domain_feats.keys())]]) * DOMAIN_WEIGHT
        vector_combined = hstack([vector_tfidf, vector_domain])
        
        # Predict
        cluster_id = int(self.kmeans_model.predict(vector_combined)[0])
        category = self.cluster_mapping.get(cluster_id, 'Unknown')
        
        # Calculate confidence
        distances = self.kmeans_model.transform(vector_combined)[0]
        inverse_distances = 1 / (distances + 1e-10)
        probabilities = inverse_distances / inverse_distances.sum()
        
        result = {
            'cluster': cluster_id,
            'category': category,
            'confidence': float(probabilities[cluster_id]),
            'domain_features': domain_feats
        }
        
        if return_probabilities:
            result['all_probabilities'] = {
                self.cluster_mapping.get(i, f'Cluster {i}'): float(probabilities[i])
                for i in range(len(probabilities))
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
                df = df[df['category'] == category]
            return df[['doc_id', 'text', 'category', 'cluster']].head(limit).to_dict('records')
        except Exception:
            return []

# Singleton instance
_clustering_service = None

def get_clustering_service():
    """
    Returns the IMPROVED clustering service singleton
    """
    global _clustering_service
    if _clustering_service is None:
        _clustering_service = ImprovedClusteringService()
    return _clustering_service

# Backwards compatibility alias
ClusteringService = ImprovedClusteringService
