"""
Task 2: Document Clustering - Main Entry Point
Cleaned Wrapper that uses the Backend Service.
"""
import os
import sys
import django

# 1. Setup Django Environment (Required for populate_clustering_db)
current_dir = os.path.dirname(os.path.abspath(__file__))
backend_dir = os.path.join(current_dir, 'backend')
sys.path.append(backend_dir)

os.environ.setdefault("DJANGO_SETTINGS_MODULE", "config.settings")
django.setup()

# 2. Import Service and Populator
from clustering.clustering_service import get_clustering_service
from populate_clustering_db import populate_clustering_db

def main():
    print("========================================")
    print("   TASK 2: DOCUMENT CLUSTERING")
    print("========================================")
    
    # Initialize Service
    service = get_clustering_service()
    
    # Run Training (Generates CSV and Models)
    print("\n[1/2] Training Model & Generating Data...")
    service.train_model()
    
    # Run DB Population (Loads CSV into SQLite)
    print("\n[2/2] Populating Database...")
    populate_clustering_db()
    
    print("\n[OK] Task 2 Completed Successfully.")

if __name__ == "__main__":
    main()
