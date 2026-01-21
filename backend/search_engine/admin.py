from django.contrib import admin

from django.contrib import admin
from .models import Document

@admin.register(Document)
class DocumentAdmin(admin.ModelAdmin):
    list_display = ('title', 'url', 'crawled_at')
    search_fields = ('title', 'url')
