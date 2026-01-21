from django.urls import path
from . import views

urlpatterns = [
    path('predict/', views.predict_cluster, name='predict_cluster'),
    path('documents/', views.get_documents, name='get_documents'),
    path('statistics/', views.get_statistics, name='get_statistics'),
    path('predictions/', views.get_predictions_history, name='get_predictions_history'),
    path('samples/', views.get_sample_documents, name='get_sample_documents'),
]
