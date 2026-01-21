from django.db import models

class ClusteredDocument(models.Model):
    """
    Model to store clustered documents
    """
    doc_id = models.IntegerField(unique=True)
    text = models.TextField()
    category = models.CharField(max_length=50)  # Business, Entertainment, Health
    cluster = models.IntegerField()
    predicted_category = models.CharField(max_length=50)
    source = models.CharField(max_length=200, default='Sample')
    created_at = models.DateTimeField(auto_now_add=True)
    
    class Meta:
        ordering = ['doc_id']
        
    def __str__(self):
        return f"Doc {self.doc_id}: {self.category} (Cluster {self.cluster})"


class ClusterPrediction(models.Model):
    """
    Model to store user predictions for analysis
    """
    input_text = models.TextField()
    predicted_cluster = models.IntegerField()
    predicted_category = models.CharField(max_length=50)
    confidence = models.FloatField()
    created_at = models.DateTimeField(auto_now_add=True)
    
    class Meta:
        ordering = ['-created_at']
        
    def __str__(self):
        return f"{self.predicted_category} ({self.confidence:.2%}) - {self.input_text[:50]}"
