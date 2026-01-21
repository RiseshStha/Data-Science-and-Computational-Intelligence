from backend.clustering.clustering_service import get_clustering_service

service = get_clustering_service()

# Test cases aimed at each category
test_cases = [
    {
        "text": "Inflation rates and corporate profits are rising in the financial market.",
        "expected": "Business"
    },
    {
        "text": "The new film won an Oscar for best director and actor performance.",
        "expected": "Entertainment"
    },
    {
        "text": "NHS hospitals and doctors are treating patients with new surgery methods.",
        "expected": "Health"
    }
]

print("-" * 50)
print("TESTING SAVED MODEL PREDICTIONS")
print("-" * 50)

for case in test_cases:
    result = service.predict_cluster(case["text"])
    pred = result["category"]
    conf = result["confidence"]
    status = "[PASS]" if pred == case["expected"] else "[FAIL]"
    
    print(f"Text: {case['text'][:50]}...")
    print(f"Expected: {case['expected']}")
    print(f"Predicted: {pred} (Confidence: {conf:.4f})")
    print(f"Status: {status}")
    print("-" * 50)
