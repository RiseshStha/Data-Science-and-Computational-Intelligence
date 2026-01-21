
import pandas as pd
from sklearn.metrics import confusion_matrix
import sys

try:
    df = pd.read_csv('data/clustered_documents.csv')
    y_true = df['category']
    y_pred = df['predicted_category']
    
    labels = sorted(list(set(y_true)))
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    
    print("\nConfusion Matrix:")
    print("Labels:", labels)
    print(cm)
    
    # Format for markdown
    print("\nMarkdown Table:")
    print("| True \\ Predicted | " + " | ".join(labels) + " |")
    print("| :--- | " + " | ".join([":---:" for _ in labels]) + " |")
    for i, label in enumerate(labels):
        row = " | ".join(map(str, cm[i]))
        print(f"| **{label}** | {row} |")

except Exception as e:
    print(e)
