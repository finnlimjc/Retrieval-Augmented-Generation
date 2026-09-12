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
SUPERSCRIPT_DIGITS = str.maketrans("0123456789", "⁰¹²³⁴⁵⁶⁷⁸⁹")


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
def load_knowledge_base(path: str, modified_time: int) -> dict[str, list[Any]]:
	"""Convert the pickle to Markdown, then load Markdown for chunking."""
	with Path(path).open("rb") as file:
		loaded = KnowledgeBaseUnpickler(file).load()
	if not isinstance(loaded, dict):
		raise ValueError("The knowledge base must contain a source-file mapping.")

	markdown_paths = knowledge_base_to_markdown(loaded, MARKDOWN_KNOWLEDGE_BASE_PATH)
	return {
		source_name: [{"text": markdown_path.read_text(encoding="utf-8")}]
		for source_name, markdown_path in markdown_paths.items()
	}


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
) -> int:
	"""Chunk and upsert all preprocessed knowledge-base elements into Chroma."""
	file_bytes = Path(path).read_bytes()
	file_hash = hashlib.sha256(file_bytes).hexdigest()[:16]
	collection = get_collection()
	knowledge_base = load_knowledge_base(path, modified_time)
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


def format_footnote_markers(answer: str) -> str:
	return re.sub(
		r"<sup>(\d+)</sup>",
		lambda match: match.group(1).translate(SUPERSCRIPT_DIGITS),
		answer,
	)


def generate_answer(query: str, model: str, document_count: int) -> str:
	"""Retrieve source context and ask Gemini for a cited, grounded answer."""
	documents = retrieve_document_context(query, document_count)
	if not documents:
		return "No indexed documents are available. Upload and process a document first."

	# Keep the source list and the prompt's document numbering in sync.
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
						chunk_count = index_knowledge_base(*initialization_settings)
					except Exception as error:
						st.session_state.pop("knowledge_base_settings", None)
						st.error("Knowledge base not available")
						st.caption(f"Could not load the preprocessed file: {error}")
					else:
						st.session_state["knowledge_base_settings"] = initialization_settings
						st.success(f"Knowledge base ready ({chunk_count} chunks indexed).")
			knowledge_base_available = (
				st.session_state.get("knowledge_base_settings") == initialization_settings
			)
			if not knowledge_base_available and not initialize_chunking:
				st.info("Adjust the chunk settings, then click Initialize chunking.")
		st.divider()
		st.subheader("Settings")
		model = st.selectbox("Model", ["gemini-3.5-flash"])
		document_count = st.slider("Retrieved sources", min_value=1, max_value=10, value=4)

	st.subheader("Chat")

	if knowledge_base_available and not st.session_state.get("messages"):
		st.info("Ask a question about the knowledge base to get started.")

	for message in st.session_state.get("messages", []):
		with st.chat_message(message["role"]):
			st.markdown(message["content"])

	if knowledge_base_available and (prompt := st.chat_input("Ask a question about your documents...")):
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
