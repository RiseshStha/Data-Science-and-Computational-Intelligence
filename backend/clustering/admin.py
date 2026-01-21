from django.contrib import admin
from .models import ClusteredDocument, ClusterPrediction


@admin.register(ClusteredDocument)
class ClusteredDocumentAdmin(admin.ModelAdmin):
    """Admin interface for clustered documents"""
    
    list_display = ['doc_id', 'category', 'cluster', 'predicted_category', 'text_preview', 'source', 'created_at']
    list_filter = ['category', 'cluster', 'predicted_category', 'source']
    search_fields = ['text', 'doc_id']
    ordering = ['doc_id']
    list_per_page = 50
    
    def text_preview(self, obj):
        """Show preview of text"""
        return obj.text[:100] + '...' if len(obj.text) > 100 else obj.text
    
    text_preview.short_description = 'Text Preview'
    
    fieldsets = (
        ('Document Information', {
            'fields': ('doc_id', 'text', 'source')
        }),
        ('Clustering Results', {
            'fields': ('category', 'cluster', 'predicted_category')
        }),
        ('Metadata', {
            'fields': ('created_at',),
            'classes': ('collapse',)
        }),
    )
    
    readonly_fields = ['created_at']


@admin.register(ClusterPrediction)
class ClusterPredictionAdmin(admin.ModelAdmin):
    """Admin interface for cluster predictions"""
    
    list_display = ['id', 'predicted_category', 'confidence_display', 'input_preview', 'created_at']
    list_filter = ['predicted_category', 'created_at']
    search_fields = ['input_text']
    ordering = ['-created_at']
    list_per_page = 50
    
    def input_preview(self, obj):
        """Show preview of input text"""
        return obj.input_text[:80] + '...' if len(obj.input_text) > 80 else obj.input_text
    
    input_preview.short_description = 'Input Text'
    
    def confidence_display(self, obj):
        """Display confidence as percentage"""
        return f"{obj.confidence:.2%}"
    
    confidence_display.short_description = 'Confidence'
    confidence_display.admin_order_field = 'confidence'
    
    fieldsets = (
        ('Prediction Input', {
            'fields': ('input_text',)
        }),
        ('Prediction Results', {
            'fields': ('predicted_cluster', 'predicted_category', 'confidence')
        }),
        ('Metadata', {
            'fields': ('created_at',),
            'classes': ('collapse',)
        }),
    )
    
    readonly_fields = ['created_at']
