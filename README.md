# Retrieval-Augmented-Generation

This project is a Streamlit-based retrieval-augmented generation (RAG) chatbot.
It reads a locally generated `extracted_elements.pkl` knowledge base, splits its
preprocessed elements into overlapping text chunks, stores those chunks in a
persistent Chroma collection, and answers questions about the source documents.
Answers are generated with Google's Gemini API and include source markers for
information drawn from the documents.

The preprocessing step that creates `extracted_elements.pkl` is separate from
this application and runs on the local device. The generated file is ignored by
Git and must not be committed or uploaded to GitHub.

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
5. In **Settings**, choose the Gemini model and the number of retrieved sources
	to use for each answer.
6. Enter a question in the chat box. The chatbot searches the indexed knowledge
	base and responds with grounded answers and source footnotes.

The Chroma index is persisted locally, so it remains available across app
restarts unless the `.chroma` directory is removed.

