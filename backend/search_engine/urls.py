from django.urls import path
from .views import SearchAPIView, CrawlAPIView, StatsAPIView
from .sample_data_view import LoadSampleDataAPIView

urlpatterns = [
    path('search/', SearchAPIView.as_view(), name='search'),
    path('crawl/', CrawlAPIView.as_view(), name='crawl'),
    path('stats/', StatsAPIView.as_view(), name='stats'),
    path('load-sample-data/', LoadSampleDataAPIView.as_view(), name='load_sample_data'),
]
