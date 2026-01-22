from pypdf import PdfReader
import sys

# Use raw string for path
pdf_path = r"d:\Data Science\Information Reterival\Assignment\ST7071CEM_CW_Regular-193ff57b-1698-41e6-9a2d-3cdf3528e822.pdf"

try:
    reader = PdfReader(pdf_path)
    print(f"Number of pages: {len(reader.pages)}")
    with open("task_requirements.txt", "w", encoding="utf-8") as f:
        # Read all pages
        for i, page in enumerate(reader.pages):
            f.write(f"--- Page {i+1} ---\n")
            f.write(page.extract_text() + "\n")
    print("Successfully wrote PDF content to task_requirements.txt")
except Exception as e:
    print(f"Error reading PDF: {e}")
