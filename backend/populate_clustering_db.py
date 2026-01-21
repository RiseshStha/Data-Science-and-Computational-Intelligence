"""
Script to populate clustering database with documents
Run this after running the notebook to train the clustering model
"""

import os
import sys
import django
import pandas as pd

# Setup Django
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'config.settings')
django.setup()

from clustering.models import ClusteredDocument


def populate_clustering_db():
    """Load clustered documents from CSV into database"""
    
    # csv_path is relative to the backend directory where this script lives
    base_dir = os.path.dirname(os.path.abspath(__file__))
    csv_path = os.path.join(base_dir, 'data', 'clustered_documents.csv')
    
    if not os.path.exists(csv_path):
        print(f"Error: {csv_path} not found!")
        print("Please run the Task2_Document_Clustering.ipynb notebook first to generate the data.")
        return
    
    print("Loading clustered documents from CSV...")
    df = pd.read_csv(csv_path)
    
    print(f"Found {len(df)} documents")
    print("\nClearing existing documents...")
    ClusteredDocument.objects.all().delete()
    
    print("Inserting documents into database...")
    documents_to_create = []
    
    for _, row in df.iterrows():
        doc = ClusteredDocument(
            doc_id=int(row['doc_id']),
            text=row['text'],
            category=row['category'],
            cluster=int(row['cluster']),
            predicted_category=row['predicted_category'],
            source=row.get('source', 'Sample')
        )
        documents_to_create.append(doc)
    
    # Bulk create for efficiency
    ClusteredDocument.objects.bulk_create(documents_to_create, batch_size=100)
    
    print(f"Successfully inserted {len(documents_to_create)} documents!")
    
    # Show statistics
    print("\n" + "="*60)
    print("DATABASE STATISTICS")
    print("="*60)
    
    for category in ['Business', 'Entertainment', 'Health']:
        count = ClusteredDocument.objects.filter(category=category).count()
        print(f"{category:15s}: {count:3d} documents")
    
    print("="*60)
    print(f"Total: {ClusteredDocument.objects.count()} documents")
    print("="*60)


if __name__ == '__main__':
    populate_clustering_db()
