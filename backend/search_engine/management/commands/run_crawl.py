from django.core.management.base import BaseCommand
from search_engine.services import Crawler, SearchEngine

class Command(BaseCommand):
    help = 'Runs the scheduled crawl task'

    def handle(self, *args, **options):
        self.stdout.write("Starting Scheduled Crawl...")
        
        crawler = Crawler()
        # We iterate to consume the generator
        for msg in crawler.crawl():
            self.stdout.write(msg.strip())
            
        self.stdout.write("Crawl Complete. Updating Index...")
        SearchEngine.get_instance().build_index()
        self.stdout.write("Index Updated Successfully.")
