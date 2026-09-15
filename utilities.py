import re
import json
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any, Mapping


TEXT_FIELDS = ("text", "content", "page_content", "raw_text")


def as_mapping(element: Any) -> dict[str, Any]:
    if is_dataclass(element):
        return asdict(element)
    if isinstance(element, dict):
        return dict(element)
    if hasattr(element, "__dict__"):
        return dict(vars(element))
    return {"text": str(element)}


def clean_text(value: Any) -> str:
    """Normalize line endings, remove null characters, and trim text whitespace."""
    text = str(value).replace("\x00", "")
    lines = [" ".join(line.split()) for line in text.replace("\r\n", "\n").replace("\r", "\n").split("\n")]
    return "\n".join(line for line in lines if line).strip()


def clean_element(element: Any) -> dict[str, Any] | None:
    """Return a cleaned element, or None when it contains no searchable text."""
    values = as_mapping(element)
    for field_name in TEXT_FIELDS:
        if values.get(field_name):
            cleaned = clean_text(values[field_name])
            if cleaned:
                values[field_name] = cleaned
                return values
            return None

    for key, value in values.items():
        if isinstance(value, str) and clean_text(value):
            values[key] = clean_text(value)
    return values if any(isinstance(value, str) and value for value in values.values()) else None


def clean_knowledge_base(knowledge_base: Mapping[Any, Any]) -> dict[str, list[dict[str, Any]]]:
    """Clean every preprocessed source while preserving source filenames and metadata."""
    cleaned_knowledge_base: dict[str, list[dict[str, Any]]] = {}
    for source_name, elements in knowledge_base.items():
        source_elements = elements if isinstance(elements, (list, tuple)) else [elements]
        cleaned_elements = [
            cleaned
            for element in source_elements
            if (cleaned := clean_element(element)) is not None
        ]
        if cleaned_elements:
            cleaned_knowledge_base[str(source_name)] = cleaned_elements
    return cleaned_knowledge_base


def element_text(element: Mapping[str, Any]) -> str:
    for field_name in TEXT_FIELDS:
        value = element.get(field_name)
        if value:
            return str(value).strip()
    return " ".join(
        str(value).strip()
        for value in element.values()
        if isinstance(value, str) and value.strip()
    )


def knowledge_base_to_markdown(
    knowledge_base: Mapping[Any, Any],
    output_directory: Path,
) -> dict[str, Path]:
    """Write cleaned sources as Markdown files and return source-to-file paths."""
    cleaned_knowledge_base = clean_knowledge_base(knowledge_base)
    output_directory.mkdir(parents=True, exist_ok=True)
    for existing_file in output_directory.glob("*.md"):
        existing_file.unlink()

    markdown_paths: dict[str, Path] = {}
    for source_name, elements in cleaned_knowledge_base.items():
        safe_name = re.sub(r"[^A-Za-z0-9._-]+", "_", source_name).strip("_")
        markdown_path = output_directory / f"{safe_name}.md"
        sections = [f"# {source_name}"]
        for element_index, element in enumerate(elements, start=1):
            metadata = []
            for key, value in element.items():
                if key in TEXT_FIELDS or value is None:
                    continue
                if isinstance(value, (dict, list, tuple)):
                    formatted_value = json.dumps(value, ensure_ascii=True, sort_keys=True)
                else:
                    formatted_value = str(value)
                metadata.append(f"**{key}:** {formatted_value}")
            section = [f"## Element {element_index}"]
            if metadata:
                section.extend(metadata)
            section.append(element_text(element))
            sections.append("\n\n".join(section))
        markdown_path.write_text("\n\n".join(sections) + "\n", encoding="utf-8")
        markdown_paths[source_name] = markdown_path
    return markdown_paths
