from rest_framework import serializers
from .models import ClusteredDocument, ClusterPrediction


class ClusteredDocumentSerializer(serializers.ModelSerializer):
    """Serializer for clustered documents"""
    
    class Meta:
        model = ClusteredDocument
        fields = ['doc_id', 'text', 'category', 'cluster', 'predicted_category', 'source', 'created_at']


class ClusterPredictionSerializer(serializers.ModelSerializer):
    """Serializer for cluster predictions"""
    
    class Meta:
        model = ClusterPrediction
        fields = ['id', 'input_text', 'predicted_cluster', 'predicted_category', 'confidence', 'created_at']


class PredictClusterInputSerializer(serializers.Serializer):
    """Serializer for prediction input"""
    text = serializers.CharField(required=True, allow_blank=False)
    return_probabilities = serializers.BooleanField(required=False, default=False)


class PredictClusterOutputSerializer(serializers.Serializer):
    """Serializer for prediction output"""
    cluster = serializers.IntegerField()
    category = serializers.CharField()
    confidence = serializers.FloatField()
    all_probabilities = serializers.DictField(required=False)
    error = serializers.CharField(required=False)
