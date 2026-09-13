# Retrieval-Augmented-Generation

This project is a Streamlit-based retrieval-augmented generation (RAG) chatbot.
It reads a locally generated `extracted_elements.pkl` knowledge base, splits its
preprocessed elements into overlapping text chunks, stores those chunks in a
persistent Chroma collection, and answers questions about the source documents.
Answers are generated with Google's Gemini API and include numbered Markdown citations for
information drawn from the documents.

The preprocessing step that creates `extracted_elements.pkl` is separate from
this application and runs on the local device. The generated file is ignored by
Git and must not be committed or uploaded to GitHub.

<img width="1847" height="842" alt="architecture_diagram" src="https://github.com/user-attachments/assets/7b3da3d8-0dc1-435a-8e94-558f4a51054e" />

## Installation

Install Python 3.10 or newer, then open a terminal in the project directory.
Creating a virtual environment is recommended:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

Install the pinned dependencies:

```powershell
python -m pip install -r requirements.txt
```

The app requires a Gemini API key. Create `.streamlit/secrets.toml` with:

```toml
GEMINI_API_KEY = "your-api-key"
```

Do not commit this file or share the API key.

## Run the App

Start Streamlit from the project directory:

```powershell
python -m streamlit run app.py
```

Streamlit will display a local URL, usually `http://localhost:8501`.

## Using the App

1. Open the local Streamlit URL in a browser.
2. Ensure the separate local preprocessing code has generated
	`extracted_elements.pkl` in the project root. If it is missing, the app shows
	**Knowledge base not available**.
3. In the **Knowledge Base** sidebar, adjust the chunk size and chunk overlap.
4. Click **Initialize chunking** to index the preprocessed elements in Chroma.
	Click it again after changing the chunk settings.
5. In **Settings**, choose the Gemini model.
6. Enter a question in the chat box. The chatbot searches all indexed source
	documents and responds with grounded answers and a matching numbered source list.

The Chroma index is persisted locally, so it remains available across app
restarts unless the `.chroma` directory is removed.

## Data

The source corpus is provided by the course (DS602) and is **not included in this
repository**. It is not ours to redistribute.

Shape of the corpus, for reference:
- ~54 files across 6 formats: Markdown, DOCX, PPTX, XLSX, PDF, and EML
- Includes: daily market digests, single-stock research notes, earnings call
  summaries, sector/macro research, central bank minutes, internal decks,
  desk memos, financial models, and forwarded emails with attachments
- Deliberately includes edge cases: scanned/image-only PDFs, versioned
  duplicate notes, a hidden spreadsheet tab, and prompt injections embedded
  in document text — see "Known limitations" below for how this pipeline
  handles them

### Formats handled

| Format | Library | Elements extracted |
|---|---|---|
| PDF (.pdf) | pdfplumber, pypdf | Per-page narrative text, tables (as Markdown), embedded raster images (routed to vision — see below) |
| Word (.docx) | python-docx | Narrative text (all paragraphs joined), tables (as Markdown) |
| PowerPoint (.pptx) | python-pptx | Per-slide text from text-bearing shapes, tables (as Markdown), speaker notes (kept as a separate element), embedded raster images (routed to vision — see below) |
| Excel (.xlsx) | openpyxl | One element per non-empty worksheet (as Markdown), including hidden sheets, flagged separately |
| Markdown (.md) | stdlib | Whole file as a single narrative element |
| Email (.eml) | stdlib `email` | Header block + `text/plain` body; attachments are saved and recursively re-ingested through the same per-format extractors |

### Problems the corpus surfaces

Naive, one-approach-fits-all ingestion (point a loader at a directory, split on
character count, embed) fails silently against this corpus. Specific problems
found and addressed:
- **Scanned/image-only PDFs**: two documents are photographs of paper with no
  text layer, so a text extractor reports success while returning nothing.
- **A hidden Excel worksheet**: a forecasts tab is hidden (`sheet_state !=
  "visible"`) and is skipped by any loader that assumes one visible sheet;
  it is also confidential, which the ingestion pipeline does not enforce
  (see "Known limitations").
- **PPTX speaker notes**: often carry material commentary but are not part of
  any slide's visible text, so a shape-only reader misses them; they may also
  be confidential.
- **Email attachments**: attachments are files in their own right (including
  the corpus's PDFs/DOCX) and are invisible unless the loader walks the
  message and re-runs extraction on each attachment.
- **Charts and tables embedded as images**: a table or chart rendered as a
  bitmap inside a PDF or PPTX page carries information the surrounding text
  does not restate, and is lost unless it is separately transcribed.
- **Vector-based images are not handled at all.** The pipeline only extracts
  *raster* images — via `pypdf`'s `page.images` for PDFs and
  `MSO_SHAPE_TYPE.PICTURE` shapes for PPTX. A chart or figure drawn with
  native vector graphics (PDF content-stream paths/lines/curves, a
  PowerPoint chart object, or a PowerPoint autoshape) is not detected, not
  saved, and not sent to the vision model — it disappears with no error and
  no signal that anything was skipped.
- **Prompt injections embedded in document text** were identified during data
  cleaning as a hazard in this corpus, but no sanitization or filtering step
  currently exists in the pipeline — this is an open gap, not a handled case.

## Key decisions

Chunking and storage:

| Decision | Chosen | Rejected | Why |
|---|---|---|---|
| Chunking | Fixed word-count split (256 words) | Structure-aware chunking | Sufficient for POC purpose. To explore larger model once initial acceptance is obtained from stakeholders. |
| Overlap | 10% | >10% | Too much overlap can reduce retrieval quality with more duplicated content stored, taking into consideration the embedding model’s max length of 256. |
| Vector DB | Chroma, persisted locally | FAISS | Simpler out-of-box persistence for a one-week POC. |

Per-format parsing:

| Format | Chosen approach | Rejected alternative | Trade-off |
|---|---|---|---|
| PDF | Two-tier: `pdfplumber` for text/tables on every page; `pypdf` to pull embedded raster images, which are sent to a vision model only when a page has no text (scan) or a text page contains a figure | Run OCR/vision on every page | Keeps vision-model cost and latency down since most pages already have a clean text layer, but the two-tier split depends on `pypdf` surfacing an image at all — vector-drawn charts surface nothing, so they are silently skipped (see "Known limitations") |
| XLSX | Read every worksheet with `openpyxl` (`data_only=True`), including hidden ones, flagged with a `sheet_hidden`/`hidden_sheet` marker | Skip non-visible sheets | The hidden sheet held business-critical forecasts, so it must be ingested — but the pipeline has no confidentiality gate, so a sheet hidden *because* it is confidential is indexed like any other |
| DOCX | Extract paragraph text and tables only; skip images | Extract embedded images too | Chosen on the unverified assumption that this corpus's `.docx` files contain no images; if that assumption is wrong, those images are dropped with no warning |
| PPTX | Mirror the PDF pipeline per slide: text-bearing shapes as one element, `shape.has_table` shapes as Markdown tables, `MSO_SHAPE_TYPE.PICTURE` shapes saved and sent to vision (`figure` if the slide has other text, `scan_slide` — OCR prompt — if the picture is the whole slide), speaker notes as a separate element | Text-only extraction (no tables/images) | Catches slide tables and pasted pictures the same way the PDF pipeline does, but `MSO_SHAPE_TYPE.PICTURE` only matches pasted raster pictures — a native PowerPoint chart object or a shape drawn with PowerPoint's own vector tools is neither a picture nor text, so it is still silently skipped (see "Known limitations") |
| EML | Read header fields (`Subject`/`From`/`To`/`Date`) and `text/plain` body only; recursively re-run the full `extract()` dispatch on every attachment | Treat attachments as opaque blobs | Attachments get identical, consistent treatment to top-level files, but an HTML-only email (no `text/plain` part) ingests as an empty body, and an attachment type outside the six supported formats raises rather than degrading gracefully |
| MD | Read the whole file as one narrative element | Structure-aware split on headings | Simplest option for a POC; loses heading hierarchy at the element level (left to the later chunking stage) |
| Images (PDF and PPTX) | Cache vision-model output (OCR transcription or chart-data extraction) on disk, keyed by a hash of the image bytes + prompt | Call the vision model on every run | Avoids repeat API cost/latency across notebook re-runs, but a bad transcription is cached and reused just as readily as a good one — there is no validation step |

## Known limitations

- **Vector-based images are not handled.** Only raster images/pictures
  embedded in PDFs and PPTX slides are detected and sent to the vision
  model; native vector graphics — PDF vector-drawn charts, PowerPoint chart
  objects, PowerPoint autoshapes — are invisible to the pipeline.
- **DOCX images are not extracted or described** — only paragraph text and
  tables are captured for `.docx` files, so any picture or chart in a Word
  document is lost.
- **No confidentiality enforcement.** The hidden Excel sheet and PPTX
  speaker notes are ingested and indexed even though both were flagged as
  confidential during data cleaning; there is no redaction or
  access-control layer.
- **Prompt injections in source documents are not sanitized or filtered**
  anywhere in the pipeline, despite being a known hazard in this corpus.
- **EML parsing only reads `text/plain` parts.** An HTML-only email with no
  plain-text alternative would ingest as an empty body.
- **Vision-model output is trusted without validation.** OCR transcriptions
  and chart-data extractions from the vision model are used as-is; a
  hallucinated or partial transcription is indistinguishable downstream from
  a correct one.
- **Unsupported attachment types raise rather than degrade.** An email
  attachment whose extension is not one of the six supported formats causes
  `extract()` to raise `ValueError` instead of being skipped or flagged.

## Future work

- Detect and transcribe vector-based charts/images — PDF content-stream
  drawings, PowerPoint chart objects, and PowerPoint autoshapes — instead of
  silently skipping anything that isn't a raster picture.
- Extend the vision pipeline to DOCX images, matching the PDF/PPTX
  figure-extraction approach.
- Add a confidentiality/redaction pass — e.g. exclude or restrict hidden
  sheets and speaker notes at retrieval time — instead of ingesting them
  unconditionally.
- Add prompt-injection detection and sanitization on extracted text before
  it is chunked and embedded.
- Fall back to the HTML body (or a rendered/vision pass) when an email has
  no `text/plain` part.
- Validate vision-model OCR/chart outputs before trusting them in the index,
  e.g. confidence scoring, a human review queue, or cross-checking against
  any surrounding page text.
- Handle unsupported or opaque attachment types gracefully instead of
  raising.
- Move from fixed word-count chunking to structure-aware chunking, once the
  POC has stakeholder buy-in to justify the added complexity.
