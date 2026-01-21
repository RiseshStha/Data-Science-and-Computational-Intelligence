from rest_framework import status
from rest_framework.decorators import api_view
from rest_framework.response import Response
from django.views.decorators.csrf import csrf_exempt
from .models import ClusteredDocument, ClusterPrediction
from .serializers import (
    ClusteredDocumentSerializer,
    ClusterPredictionSerializer,
    PredictClusterInputSerializer,
    PredictClusterOutputSerializer
)
from .clustering_service import get_clustering_service


@api_view(['POST'])
@csrf_exempt
def predict_cluster(request):
    """
    Predict the cluster/category for a new document
    
    POST /api/clustering/predict/
    Body: {
        "text": "Your document text here",
        "return_probabilities": true  // optional
    }
    """
    serializer = PredictClusterInputSerializer(data=request.data)
    
    if not serializer.is_valid():
        return Response(
            {'error': 'Invalid input', 'details': serializer.errors},
            status=status.HTTP_400_BAD_REQUEST
        )
    
    text = serializer.validated_data['text']
    return_probabilities = serializer.validated_data.get('return_probabilities', False)
    
    try:
        # Get clustering service
        clustering_service = get_clustering_service()
        
        # Predict
        result = clustering_service.predict_cluster(text, return_probabilities)
        
        # Save prediction to database
        if 'error' not in result:
            ClusterPrediction.objects.create(
                input_text=text,
                predicted_cluster=result['cluster'],
                predicted_category=result['category'],
                confidence=result['confidence']
            )
        
        return Response(result, status=status.HTTP_200_OK)
        
    except Exception as e:
        return Response(
            {'error': str(e)},
            status=status.HTTP_500_INTERNAL_SERVER_ERROR
        )


@api_view(['GET'])
def get_documents(request):
    """
    Get clustered documents
    
    GET /api/clustering/documents/?category=Business&limit=20
    """
    category = request.query_params.get('category', None)
    limit = int(request.query_params.get('limit', 50))
    
    try:
        queryset = ClusteredDocument.objects.all()
        
        if category:
            queryset = queryset.filter(category=category)
        
        queryset = queryset[:limit]
        serializer = ClusteredDocumentSerializer(queryset, many=True)
        
        return Response({
            'count': queryset.count(),
            'documents': serializer.data
        }, status=status.HTTP_200_OK)
        
    except Exception as e:
        return Response(
            {'error': str(e)},
            status=status.HTTP_500_INTERNAL_SERVER_ERROR
        )


@api_view(['GET'])
def get_statistics(request):
    """
    Get clustering statistics and metrics
    
    GET /api/clustering/statistics/
    """
    try:
        clustering_service = get_clustering_service()
        stats = clustering_service.get_cluster_statistics()
        
        # Add database statistics
        stats['total_documents_in_db'] = ClusteredDocument.objects.count()
        stats['total_predictions'] = ClusterPrediction.objects.count()
        
        # Category distribution from database
        category_counts = {}
        for category in ['Business', 'Entertainment', 'Health']:
            count = ClusteredDocument.objects.filter(category=category).count()
            category_counts[category] = count
        
        stats['category_distribution'] = category_counts
        
        return Response(stats, status=status.HTTP_200_OK)
        
    except Exception as e:
        return Response(
            {'error': str(e)},
            status=status.HTTP_500_INTERNAL_SERVER_ERROR
        )


@api_view(['GET'])
def get_predictions_history(request):
    """
    Get history of user predictions
    
    GET /api/clustering/predictions/?limit=20
    """
    limit = int(request.query_params.get('limit', 20))
    
    try:
        predictions = ClusterPrediction.objects.all()[:limit]
        serializer = ClusterPredictionSerializer(predictions, many=True)
        
        return Response({
            'count': predictions.count(),
            'predictions': serializer.data
        }, status=status.HTTP_200_OK)
        
    except Exception as e:
        return Response(
            {'error': str(e)},
            status=status.HTTP_500_INTERNAL_SERVER_ERROR
        )


@api_view(['GET'])
def get_sample_documents(request):
    """
    Get sample documents for each category
    
    GET /api/clustering/samples/?category=Business&limit=5
    """
    category = request.query_params.get('category', None)
    limit = int(request.query_params.get('limit', 5))
    
    try:
        clustering_service = get_clustering_service()
        samples = clustering_service.get_sample_documents(category, limit)
        
        return Response({
            'count': len(samples),
            'samples': samples
        }, status=status.HTTP_200_OK)
        
    except Exception as e:
        return Response(
            {'error': str(e)},
            status=status.HTTP_500_INTERNAL_SERVER_ERROR
        )
