import hashlib
import io
import pickle
import re
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any

import chromadb
import streamlit as st
from google import genai
from pypdf import PdfReader


st.set_page_config(
	page_title="RAG Chatbot",
	page_icon=":books:",
	layout="wide",
)


CHROMA_PATH = Path(".chroma")
COLLECTION_NAME = "financial_documents"
CHUNK_SIZE = 800
CHUNK_OVERLAP = 120
SUPERSCRIPT_DIGITS = str.maketrans("0123456789", "⁰¹²³⁴⁵⁶⁷⁸⁹")


@st.cache_resource
def get_collection() -> Any:
	client = chromadb.PersistentClient(path=str(CHROMA_PATH))
	return client.get_or_create_collection(name=COLLECTION_NAME)


@st.cache_resource
def get_gemini_client() -> genai.Client:
	api_key = st.secrets.get("GEMINI_API_KEY")
	if not api_key:
		raise RuntimeError(
			"GEMINI_API_KEY is missing. Add it to .streamlit/secrets.toml."
		)
	return genai.Client(api_key=api_key)


def as_mapping(element: Any) -> dict[str, Any]:
	if is_dataclass(element):
		return asdict(element)
	if isinstance(element, dict):
		return element
	if hasattr(element, "__dict__"):
		return vars(element)
	return {"text": str(element)}


def element_text(element: Any) -> str:
	values = as_mapping(element)
	for field_name in ("text", "content", "page_content", "raw_text"):
		value = values.get(field_name)
		if value:
			return str(value).strip()

	return " ".join(str(value) for value in values.values() if isinstance(value, str)).strip()


def primitive_metadata(element: Any, source_name: str) -> dict[str, str | int | float | bool]:
	metadata: dict[str, str | int | float | bool] = {"source": source_name}
	for key, value in as_mapping(element).items():
		if key in {"text", "content", "page_content", "raw_text"}:
			continue
		if isinstance(value, (str, int, float, bool)):
			metadata[str(key)] = value
	return metadata


def load_elements(file_bytes: bytes, file_name: str) -> list[Any]:
	if file_name.lower().endswith(".pdf"):
		reader = PdfReader(io.BytesIO(file_bytes))
		return [
			{"text": page.extract_text() or "", "page_number": page_number + 1}
			for page_number, page in enumerate(reader.pages)
		]

	loaded = pickle.loads(file_bytes)
	if isinstance(loaded, (list, tuple)):
		return list(loaded)
	if isinstance(loaded, dict) and "elements" in loaded:
		elements = loaded["elements"]
		return list(elements) if isinstance(elements, (list, tuple)) else [elements]
	return [loaded]


def chunk_text(
	text: str,
	chunk_size: int = CHUNK_SIZE,
	chunk_overlap: int = CHUNK_OVERLAP,
) -> list[str]:
	if chunk_size <= 0 or chunk_overlap < 0 or chunk_overlap >= chunk_size:
		raise ValueError("Chunk overlap must be non-negative and smaller than chunk size.")

	words = re.findall(r"\S+", text)
	chunks: list[str] = []
	start = 0
	while start < len(words):
		end = min(start + chunk_size, len(words))
		chunks.append(" ".join(words[start:end]))
		if end == len(words):
			break
		start = end - chunk_overlap
	return chunks


def index_upload(uploaded_file: Any, chunk_size: int, chunk_overlap: int) -> int:
	file_bytes = uploaded_file.getvalue()
	file_hash = hashlib.sha256(file_bytes).hexdigest()[:16]
	collection = get_collection()
	elements = load_elements(file_bytes, uploaded_file.name)
	ids: list[str] = []
	documents: list[str] = []
	metadatas: list[dict[str, str | int | float | bool]] = []
	global_chunk_index = 0

	for element_index, element in enumerate(elements):
		text = element_text(element)
		for chunk_index, chunk in enumerate(chunk_text(text, chunk_size, chunk_overlap)):
			ids.append(f"{file_hash}-{element_index}-{chunk_index}")
			documents.append(chunk)
			metadata = primitive_metadata(element, uploaded_file.name)
			metadata.update(
				{
					"document_id": file_hash,
					"element_index": element_index,
					"chunk_index": global_chunk_index,
				}
			)
			metadatas.append(metadata)
			global_chunk_index += 1

	if documents:
		collection.upsert(ids=ids, documents=documents, metadatas=metadatas)
	return len(documents)


def retrieve_document_context(query: str, document_count: int = 4) -> list[dict[str, str]]:
	"""Find relevant files, then return every chunk from each matched file."""
	collection = get_collection()
	if collection.count() == 0:
		return []
	document_count = min(document_count, collection.count())
	matches = collection.query(
		query_texts=[query],
		n_results=document_count,
		include=["metadatas"],
	)
	matched_metadata = matches.get("metadatas", [[]])[0]
	document_ids = list(dict.fromkeys(
		metadata["document_id"]
		for metadata in matched_metadata
		if metadata and "document_id" in metadata
	))

	full_documents: list[dict[str, str]] = []
	for document_id in document_ids:
		document = collection.get(
			where={"document_id": document_id},
			include=["documents", "metadatas"],
		)
		chunks = list(zip(document["documents"] or [], document["metadatas"] or []))
		chunks.sort(key=lambda chunk: int(chunk[1].get("chunk_index", 0)))
		if chunks:
			full_documents.append(
				{
					"source": str(chunks[0][1].get("source", document_id)),
					"content": "\n\n".join(chunk[0] for chunk in chunks),
				}
			)
	return full_documents


def format_footnote_markers(answer: str) -> str:
	return re.sub(
		r"<sup>(\d+)</sup>",
		lambda match: match.group(1).translate(SUPERSCRIPT_DIGITS),
		answer,
	)


def generate_answer(query: str, model: str, document_count: int) -> str:
	documents = retrieve_document_context(query, document_count)
	if not documents:
		return "No indexed documents are available. Upload and process a document first."

	citation_lines = []
	context_sections = []
	for citation_number, document in enumerate(documents, start=1):
		source = document["source"]
		citation_lines.append(f"{citation_number}. {source}")
		context_sections.append(
			f"DOCUMENT {citation_number} ({source}, footnote {citation_number}):\n"
			f"{document['content']}"
		)
	context = "\n\n".join(context_sections)
	prompt = f"""You are a junior financial analyst compiling findings from the provided documents for your boss, a senior analyst.

Provide the relevant information requested by the senior analyst clearly and accurately. Use only the document context below. If the answer is not present in the documents, say that you cannot find it in the uploaded documents. Do not invent financial figures or facts.

	Cite every factual statement drawn from a document using a Chicago-style superscript footnote marker, such as `<sup>1</sup>`, immediately after the relevant sentence or figure. Use multiple superscript markers when a statement relies on multiple documents. End your answer with a "Sources" section containing numbered footnotes in this format: `1. annual_report.pdf`.

If the documents contain confidential, private, or otherwise sensitive information relevant to the answer, clearly flag it as confidential or sensitive and remind the senior analyst to handle it carefully and avoid unauthorized disclosure.

DOCUMENT CONTEXT:
{context}

USER QUESTION:
{query}"""
	response = get_gemini_client().models.generate_content(
		model=model,
		contents=prompt,
	)
	answer = response.text or "Gemini returned an empty response."
	if "Sources" not in answer:
		answer = f"{answer}\n\nSources\n" + "\n".join(citation_lines)
	return format_footnote_markers(answer)


def main() -> None:
	st.title("RAG Chatbot")
	st.caption("Ask questions about your documents and get grounded answers.")

	with st.sidebar:
		st.header("Knowledge Base")
		st.subheader("Chunking")
		chunk_size = st.slider(
			"Chunk size (words)",
			min_value=100,
			max_value=2000,
			value=CHUNK_SIZE,
			step=50,
			help="Number of words in each embedded chunk.",
		)
		chunk_overlap = st.slider(
			"Chunk overlap (words)",
			min_value=0,
			max_value=chunk_size - 1,
			value=min(CHUNK_OVERLAP, chunk_size - 1),
			step=10,
			help="Words repeated between neighboring chunks to preserve context.",
		)
		uploaded_files = st.file_uploader(
			"Upload documents",
			type=["pkl", "pdf"],
			accept_multiple_files=True,
		)
		if uploaded_files and st.button("Process documents", type="primary"):
			with st.spinner("Chunking and embedding documents..."):
				try:
					chunk_count = sum(
						index_upload(file, chunk_size, chunk_overlap)
						for file in uploaded_files
					)
				except Exception as error:
					st.error(f"Could not process the uploaded documents: {error}")
				else:
					st.success(f"Indexed {chunk_count} chunks in Chroma.")
		st.divider()
		st.subheader("Settings")
		model = st.selectbox("Model", ["gemini-3.5-flash"])
		document_count = st.slider("Retrieved sources", min_value=1, max_value=10, value=4)

	st.subheader("Chat")

	if not st.session_state.get("messages"):
		st.info("Upload a document, then ask a question to get started.")

	for message in st.session_state.get("messages", []):
		with st.chat_message(message["role"]):
			st.markdown(message["content"])

	if prompt := st.chat_input("Ask a question about your documents..."):
		st.session_state.setdefault("messages", []).append(
			{"role": "user", "content": prompt}
		)
		with st.chat_message("user"):
			st.markdown(prompt)

		with st.chat_message("assistant"):
			with st.spinner("Reviewing your documents..."):
				try:
					answer = generate_answer(prompt, model, document_count)
				except Exception as error:
					answer = f"I could not generate an answer: {error}"
				st.markdown(answer)
		st.session_state["messages"].append(
				{"role": "assistant", "content": answer}
			)


if __name__ == "__main__":
	main()
