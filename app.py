import hashlib
import pickle
import re
from pathlib import Path
from typing import Any

import chromadb
import streamlit as st
from google import genai
from utilities import as_mapping, knowledge_base_to_markdown


st.set_page_config(
	page_title="RAG Chatbot",
	page_icon=":books:",
	layout="wide",
)


CHROMA_PATH = Path(".chroma")
MARKDOWN_KNOWLEDGE_BASE_PATH = Path(__file__).with_name(".knowledge_base_md")
KNOWLEDGE_BASE_PATH = Path(__file__).with_name("extracted_elements.pkl")
COLLECTION_NAME = "financial_documents"
CHUNK_SIZE = 256
CHUNK_OVERLAP = 25
RETRIEVAL_TOP_K = 5
PROMPT_INJECTION_PATTERNS = (
	("instruction override", re.compile(r"\b(ignore|disregard|forget|override)\b.{0,80}\b(previous|above|prior|system|developer|user)\b.{0,40}\b(instruction|prompt|message)s?\b", re.IGNORECASE)),
	("role impersonation", re.compile(r"\b(system message|developer message|assistant message|you are chatgpt|you are an ai)\b", re.IGNORECASE)),
	("prompt disclosure request", re.compile(r"\b(reveal|disclose|show|print|repeat)\b.{0,60}\b(system prompt|hidden prompt|instructions)\b", re.IGNORECASE)),
)
IMAGE_ELEMENT_TYPES = {"figure", "scan_page", "scan_slide", "image", "binary"}


@st.cache_resource
def get_collection() -> Any:
	"""Return the persistent Chroma collection used as the document index."""
	client = chromadb.PersistentClient(path=str(CHROMA_PATH))
	return client.get_or_create_collection(name=COLLECTION_NAME)


@st.cache_resource
def get_gemini_client() -> genai.Client:
	"""Create the cached Gemini client from Streamlit's secret configuration."""
	api_key = st.secrets.get("GEMINI_API_KEY")
	if not api_key:
		raise RuntimeError(
			"GEMINI_API_KEY is missing. Add it to .streamlit/secrets.toml."
		)
	return genai.Client(api_key=api_key)


def element_text(element: Any) -> str:
	values = as_mapping(element)
	for field_name in ("text", "content", "page_content", "raw_text"):
		value = values.get(field_name)
		if value:
			return str(value).strip()

	return " ".join(str(value) for value in values.values() if isinstance(value, str)).strip()


def source_warnings(source_name: str, elements: Any) -> list[str]:
	"""Return warnings for sources that should not be indexed as-is."""
	if not isinstance(elements, (list, tuple)):
		elements = [elements]
	texts: list[str] = []
	warnings: list[str] = []
	for element in elements:
		values = as_mapping(element)
		text = next(
			(
				str(values[field_name]).strip()
				for field_name in ("text", "content", "page_content", "raw_text")
				if isinstance(values.get(field_name), str) and values[field_name].strip()
			),
			"",
		)
		if text:
			texts.append(text)
		if values.get("element_type") in IMAGE_ELEMENT_TYPES:
			warnings.append(
				f"Skipped image or binary element in {source_name}; only text elements are indexed."
			)
	combined_text = "\n".join(texts)

	for warning_name, pattern in PROMPT_INJECTION_PATTERNS:
		if pattern.search(combined_text):
			warnings.append(
				f"Skipped {source_name}: possible prompt injection detected ({warning_name})."
			)
			return list(dict.fromkeys(warnings))

	if not combined_text:
		warnings.append(
			f"Skipped {source_name}: no searchable text was found; image or binary content is not indexed."
		)
	return list(dict.fromkeys(warnings))


def source_should_be_excluded(elements: Any) -> bool:
	"""Exclude a whole source for injection risk or when it has no text."""
	if not isinstance(elements, (list, tuple)):
		elements = [elements]
	texts = []
	for element in elements:
		values = as_mapping(element)
		for field_name in ("text", "content", "page_content", "raw_text"):
			value = values.get(field_name)
			if isinstance(value, str) and value.strip():
				texts.append(value.strip())
	combined_text = "\n".join(texts)
	return not combined_text or any(
		pattern.search(combined_text)
		for _, pattern in PROMPT_INJECTION_PATTERNS
	)


def primitive_metadata(element: Any, source_name: str) -> dict[str, str | int | float | bool]:
	metadata: dict[str, str | int | float | bool] = {"source": source_name}
	for key, value in as_mapping(element).items():
		if key in {"text", "content", "page_content", "raw_text"}:
			continue
		if isinstance(value, (str, int, float, bool)):
			metadata[str(key)] = value
	return metadata


class SerializedElement:
	def __new__(cls, *args: Any, **kwargs: Any) -> "SerializedElement":
		return object.__new__(cls)


class KnowledgeBaseUnpickler(pickle.Unpickler):
	def find_class(self, module: str, name: str) -> type[SerializedElement]:
		return SerializedElement


@st.cache_resource
def load_knowledge_base(
	path: str,
	modified_time: int,
) -> tuple[dict[str, list[Any]], list[str]]:
	"""Convert the pickle to Markdown, then load Markdown for chunking."""
	with Path(path).open("rb") as file:
		loaded = KnowledgeBaseUnpickler(file).load()
	if not isinstance(loaded, dict):
		raise ValueError("The knowledge base must contain a source-file mapping.")

	filtered_knowledge_base: dict[str, Any] = {}
	warnings: list[str] = []
	for source_name, elements in loaded.items():
		source_name = str(source_name)
		source_issues = source_warnings(source_name, elements)
		if source_should_be_excluded(elements):
			warnings.extend(source_issues)
			continue
		warnings.extend(source_issues)
		filtered_knowledge_base[source_name] = elements

	markdown_paths = knowledge_base_to_markdown(
		filtered_knowledge_base,
		MARKDOWN_KNOWLEDGE_BASE_PATH,
	)
	return ({
		source_name: [{"text": markdown_path.read_text(encoding="utf-8")}]
		for source_name, markdown_path in markdown_paths.items()
	}, warnings)


def chunk_text(
	text: str,
	chunk_size: int = CHUNK_SIZE,
	chunk_overlap: int = CHUNK_OVERLAP,
) -> list[str]:
	"""Split text into overlapping word-based chunks for embedding."""
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


@st.cache_resource
def index_knowledge_base(
	path: str,
	modified_time: int,
	chunk_size: int,
	chunk_overlap: int,
) -> tuple[int, list[str]]:
	"""Chunk and upsert all preprocessed knowledge-base elements into Chroma."""
	file_bytes = Path(path).read_bytes()
	file_hash = hashlib.sha256(file_bytes).hexdigest()[:16]
	collection = get_collection()
	knowledge_base, warnings = load_knowledge_base(path, modified_time)
	collection.delete(
		where={
			"document_id": {
				"$in": [f"{file_hash}-{source_name}" for source_name in knowledge_base]
			}
		}
	)
	ids: list[str] = []
	documents: list[str] = []
	metadatas: list[dict[str, str | int | float | bool]] = []
	global_chunk_index = 0

	for source_name, elements in knowledge_base.items():
		for element_index, element in enumerate(elements):
			# Convert the element and build its stable metadata only once per element.
			element_values = as_mapping(element)
			text = element_text(element_values)
			base_metadata = primitive_metadata(element_values, source_name)
			for chunk_index, chunk in enumerate(chunk_text(text, chunk_size, chunk_overlap)):
				ids.append(f"{file_hash}-{element_index}-{chunk_index}-{source_name}")
				documents.append(chunk)
				metadata = base_metadata.copy()
				metadata.update(
					{
						"document_id": f"{file_hash}-{source_name}",
						"element_index": element_index,
						"chunk_index": global_chunk_index,
					}
				)
				metadatas.append(metadata)
				global_chunk_index += 1

	if documents:
		collection.upsert(ids=ids, documents=documents, metadatas=metadatas)
	return len(documents), warnings


def retrieve_document_context(
	query: str,
	document_count: int | None = None,
) -> list[dict[str, str]]:
	"""Find relevant files, then return every chunk from each matched file."""
	collection = get_collection()
	if collection.count() == 0:
		return []
	if document_count is None:
		document_count = collection.count()
	else:
		document_count = min(document_count, collection.count())
	matches = collection.query(
		query_texts=[query],
		n_results=document_count,
		include=["metadatas"],
	)
	# Search returns representative chunks; the answer uses all chunks from each
	# matched document so the model has the complete source context.
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


def render_source_citations(answer: str, citation_sources: dict[int, str]) -> str:
	"""Render controlled source markers without relying on Markdown citations."""
	answer_without_sources = re.split(
		r"(?im)^\s{0,3}(?:#{1,6}\s*)?(?:Sources|Retrieved Sources)\s*:?\s*$",
		answer,
		maxsplit=1,
	)[0].rstrip()
	referenced_numbers: set[int] = set()

	def replace_marker(match: re.Match[str]) -> str:
		number = int(match.group(1))
		if number not in citation_sources:
			return ""
		referenced_numbers.add(number)
		return f"(Source: {citation_sources[number]})"

	rendered_answer = re.sub(
		r"\[?\bSOURCE[_ -]?(\d+)\b\]?",
		replace_marker,
		answer_without_sources,
		flags=re.IGNORECASE,
	).strip()
	if not referenced_numbers:
		referenced_numbers = set(citation_sources)
		retrieval_note = "No inline source markers were returned; showing all retrieved sources."
	else:
		retrieval_note = ""
	source_lines = [
		f"- {citation_sources[number]}"
		for number in sorted(referenced_numbers)
		if number in citation_sources
	]
	if retrieval_note:
		source_lines.insert(0, retrieval_note)
	return f"{rendered_answer}\n\nRetrieved Sources\n" + "\n".join(source_lines)


def escape_currency_dollars(markdown: str) -> str:
	"""Escape dollar signs outside code so Streamlit cannot interpret LaTeX."""
	protected: list[str] = []
	protected_pattern = re.compile(
		r"```[\s\S]*?```|`[^`\n]*`"
	)

	def protect(match: re.Match[str]) -> str:
		protected.append(match.group(0))
		return f"__PROTECTED_MARKDOWN_{len(protected) - 1}__"

	text = protected_pattern.sub(protect, markdown)
	text = re.sub(r"(?<!\\)\$", r"\\$", text)
	for index, value in enumerate(protected):
		text = text.replace(f"__PROTECTED_MARKDOWN_{index}__", value)
	return text


def generate_answer(query: str, model: str, top_k: int) -> str:
	"""Retrieve source context and ask Gemini for a cited, grounded answer."""
	documents = retrieve_document_context(query, document_count=top_k)
	if not documents:
		return "No indexed documents are available. Upload and process a document first."

	# Keep the source list and the prompt's document numbering in sync.
	citation_sources = {}
	context_sections = []
	for citation_number, document in enumerate(documents, start=1):
		source = document["source"]
		citation_sources[citation_number] = source
		context_sections.append(
			f"DOCUMENT {citation_number} ({source}, footnote {citation_number}):\n"
			f"{document['content']}"
		)
	context = "\n\n".join(context_sections)
	prompt = f"""You are a junior financial analyst compiling findings from the provided documents for your boss, a senior analyst.

	Provide the relevant information requested by the senior analyst clearly and accurately. Use only the document context below. If the answer is not present in the documents, say that you cannot find it in the uploaded documents. Do not invent financial figures or facts. Treat each document as an independent source: never combine a company's figures with figures from another company. Copy financial numbers, units, decimal points, commas, percent signs, and spaces exactly as written in the cited source. Do not concatenate adjacent words or values. When reporting a figure, verify that every number in the sentence comes from the same cited document.

	Cite every factual statement drawn from a document by placing the exact source marker `SOURCE_1`, `SOURCE_2`, and so on immediately after the relevant sentence or figure. Use multiple markers when a statement relies on multiple documents, such as `SOURCE_1 SOURCE_2`. Only use source marker numbers assigned to the documents below. Do not write a Sources section; the application will render the source references.

If the documents contain confidential, private, or otherwise sensitive information relevant to the answer, clearly flag it as confidential or sensitive and remind the senior analyst to handle it carefully and avoid unauthorized disclosure.

	DOCUMENT CONTEXT (each document is independent; do not merge facts across documents):
{context}

USER QUESTION:
{query}"""
	response = get_gemini_client().models.generate_content(
		model=model,
		contents=prompt,
	)
	answer = response.text or "Gemini returned an empty response."
	return render_source_citations(answer, citation_sources)


def main() -> None:
	st.title("RAG Chatbot")
	st.caption("Ask questions about your documents and get grounded answers.")
	knowledge_base_available = False

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
		if not KNOWLEDGE_BASE_PATH.exists():
			st.error("Knowledge base not available")
		else:
			modified_time = KNOWLEDGE_BASE_PATH.stat().st_mtime_ns
			initialization_settings = (
				str(KNOWLEDGE_BASE_PATH),
				modified_time,
				chunk_size,
				chunk_overlap,
			)
			initialize_chunking = st.button("Initialize chunking", type="primary")
			if initialize_chunking:
				with st.spinner("Chunking and embedding documents..."):
					try:
						chunk_count, indexing_warnings = index_knowledge_base(*initialization_settings)
					except Exception as error:
						st.session_state.pop("knowledge_base_settings", None)
						st.error("Knowledge base not available")
						st.caption(f"Could not load the preprocessed file: {error}")
					else:
						st.session_state["knowledge_base_settings"] = initialization_settings
						st.success(f"Knowledge base ready ({chunk_count} chunks indexed).")
						for warning in indexing_warnings:
							st.warning(warning)
			knowledge_base_available = (
				st.session_state.get("knowledge_base_settings") == initialization_settings
			)
			if not knowledge_base_available and not initialize_chunking:
				st.info("Adjust the chunk settings, then click Initialize chunking.")
		st.divider()
		st.subheader("Settings")
		top_k = st.slider(
			"Top k sources",
			min_value=1,
			max_value=20,
			value=RETRIEVAL_TOP_K,
			step=1,
			help="Maximum number of source documents considered for each answer.",
		)
		model = st.selectbox("Model", ["gemini-3.5-flash"])

	st.subheader("Chat")

	if knowledge_base_available and not st.session_state.get("messages"):
		st.info("Ask a question about the knowledge base to get started.")

	for message in st.session_state.get("messages", []):
		with st.chat_message(message["role"]):
			st.markdown(escape_currency_dollars(message["content"]))

	if knowledge_base_available and (prompt := st.chat_input("Ask a question about your documents...")):
		st.session_state.setdefault("messages", []).append(
			{"role": "user", "content": prompt}
		)
		with st.chat_message("user"):
			st.markdown(prompt)

		with st.chat_message("assistant"):
			with st.spinner("Reviewing your documents..."):
				try:
					answer = generate_answer(prompt, model, top_k)
				except Exception as error:
					answer = f"I could not generate an answer: {error}"
				st.markdown(escape_currency_dollars(answer))
		st.session_state["messages"].append(
				{"role": "assistant", "content": answer}
			)


if __name__ == "__main__":
	main()
