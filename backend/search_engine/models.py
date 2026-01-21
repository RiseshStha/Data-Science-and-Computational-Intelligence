from django.db import models

class Document(models.Model):
    title = models.CharField(max_length=500)
    url = models.URLField(unique=True)
    authors = models.TextField(blank=True, null=True)  # Stored as JSON or comma-separated
    year = models.CharField(max_length=10, blank=True, null=True)
    abstract = models.TextField(blank=True, null=True)
    keywords = models.TextField(blank=True, null=True)
    profile_link = models.URLField(blank=True, null=True)
    crawled_at = models.DateTimeField(auto_now_add=True)
    content = models.TextField(blank=True, null=True)

    def __str__(self):
        return self.title
