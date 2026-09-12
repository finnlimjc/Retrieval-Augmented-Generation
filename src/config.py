from pathlib import Path

PROJECT_ROOT = Path(__name__).resolve().parent
DATA_DIR = PROJECT_ROOT / "data"
METADATA_PATH = DATA_DIR / "metadata.jsonl"

DERIVED_DIR = DATA_DIR / "derived"
IMAGES_DIR = DERIVED_DIR / "images"
ATTACHMENTS_DIR = DERIVED_DIR / "attachments"
VISION_CACHE_DIR = DERIVED_DIR / "vision_cache"
VECTOR_DB_DIR = DERIVED_DIR / "vector_db"

VISION_MODEL = "gemini-3.6-flash"
OCR_PROMPT = """
Transcribe this scanned page exactly as it appears.

Rules:
- Reproduce all text verbatim. Do not summarise, correct or reorder anything.
- Render any table as a Markdown table, preserving every row and column.
- Preserve headings and the reading order of the page.
- If a word is genuinely illegible, write [illegible]. Never guess at it.

Return only the transcription.
"""

CHART_PROMPT = """
Extract the data from this figure so it can be read without seeing the image.

Rules:
- Start with the exact title.
- Transcribe every text annotation, callout and label on the figure word for word. These
  often carry the specific values the surrounding document does not state, so they matter
  more than the general shape of the chart.
- Give the axis labels and their units.
- Put the plotted values in a Markdown table, one column per series, one row per category
  on the horizontal axis. Read values off the axis where they are not labelled directly.
- If the figure is a table rather than a chart, reproduce it as a Markdown table.
- Do not interpret, and do not add commentary.

Return only the extraction.
"""

PROMPTS = {"scan_page": OCR_PROMPT, "figure": CHART_PROMPT, "scan_slide": OCR_PROMPT}
RESULT_TYPES = {"scan_page": "scan_ocr", "figure": "chart_extract", "scan_slide": "scan_ocr"}

if __name__ == "__main__":
    for directory in (DERIVED_DIR, IMAGES_DIR, ATTACHMENTS_DIR, VISION_CACHE_DIR, VECTOR_DB_DIR):
        directory.mkdir(parents=True, exist_ok=True)