import time
import schedule
from datetime import datetime
from django.core.management.base import BaseCommand
from django.core.management import call_command

class Command(BaseCommand):
    help = 'Runs the crawler every two weeks on Sunday'

    def handle(self, *args, **options):
        self.stdout.write("Scheduler started. Waiting for the next scheduled run (Every 2nd Sunday)...")
        
        # Schedule the job to run every Sunday at 00:00 (Midnight)
        schedule.every().sunday.at("00:00").do(self.scheduled_job)

        while True:
            schedule.run_pending()
            time.sleep(60) # Check every minute

    def scheduled_job(self):
        # Logic to ensure it runs every 2 weeks
        current_week = datetime.now().isocalendar()[1]
        
        if current_week % 2 == 0:
            self.stdout.write(f"Executing scheduled crawl for Week {current_week}...")
            try:
                # Call the existing run_crawl command
                call_command('run_crawl')
                self.stdout.write("Crawl completed successfully.")
            except Exception as e:
                self.stderr.write(f"Error during scheduled crawl: {e}")
        else:
            self.stdout.write(f"Skipping Week {current_week} (Bi-weekly schedule).")
