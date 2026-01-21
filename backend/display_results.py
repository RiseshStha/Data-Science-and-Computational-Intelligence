import json
from pathlib import Path

def main():
    base_dir = Path(__file__).resolve().parent
    data_dir = base_dir / 'data'
    
    # Load Metrics
    try:
        with open(data_dir / 'clustering_metrics.json', 'r') as f:
            metrics = json.load(f)
            
        print("\n\n")
        print("=" * 70)
        print("EVALUATION METRICS - DOCUMENT CLUSTERING")
        print("=" * 70)
        
        # Header
        print(f"{'Metric':<35} {'Score':<15} {'Description'}")
        print("-" * 70)
        
        # Data Rows
        # (Metric Name, Key in JSON, Description)
        rows = [
            ("Weighted F1 Score", "f1_score_weighted", "Harmonic mean of precision/recall"),
            ("Accuracy", "accuracy", "Correctly classified documents"),
            ("Purity Score", "purity_score", "Cluster homogeneity"),
            ("Adjusted Rand Index (ARI)", "adjusted_rand_index", "Similarity to ground truth"),
            ("Normalized Mutual Info (NMI)", "normalized_mutual_info", "Mutual dependence measure"),
            ("Silhouette Score", "silhouette_score", "Cluster cohesion/separation"),
        ]
        
        for name, key, desc in rows:
            score = metrics.get(key, 0.0)
            print(f"{name:<35} {score:<15.4f} {desc}")
            
        print("=" * 70)
        print(f"Total Documents: {metrics.get('n_documents', 'N/A')}")
        print(f"Number of Clusters: {metrics.get('n_clusters', 3)}")
        print("=" * 70)
        print("\n")

    except FileNotFoundError:
        print("Metrics file not found. Please run training first.")
    except Exception as e:
        print(f"Error displaying results: {e}")

if __name__ == "__main__":
    main()
