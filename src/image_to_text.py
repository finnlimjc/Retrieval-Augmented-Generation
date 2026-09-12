import base64
import hashlib
import json
import mimetypes

from langchain_core.messages import HumanMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages.content import TextContentBlock, ImageContentBlock
from pathlib import Path

from src.config import VISION_MODEL, PROMPTS, VISION_CACHE_DIR, RESULT_TYPES
from src.parsers import Element

def build_model(api_key:str) -> ChatGoogleGenerativeAI:
    """Construct the multimodal model used for both OCR and figure extraction. Only accepts Gemini models."""
    model = ChatGoogleGenerativeAI(model=VISION_MODEL, google_api_key=api_key)
    return model

def describe_image(image_path:Path, element_type:str, model):
    """
    Run the prompt matching element_type over one image, using the cache when possible.
    Unique cache file name is created by hashing the image bytes and prompt together.
    
    Inputs:
        - image_path: image written to disk by the parser stage (mainly from PDF extraction).
        - element_type: either "scan_page" or "figure"; selects the prompt.
        - model: reuse an existing client.
    
    Outputs:
        - text: the transcription or description.
    """
    # Load the prompt, and create a deterministic unique filename for the cache
    prompt = PROMPTS[element_type]
    image_bytes = image_path.read_bytes()
    digest = hashlib.sha256(image_bytes + prompt.encode("utf-8")).hexdigest()
    cache_path = VISION_CACHE_DIR / f"{digest}.json"
    if cache_path.exists():
        return json.loads(cache_path.read_text(encoding="utf-8"))["text"]
    
    # Encode prompt and image to send to the model
    # Source: https://reference.langchain.com/python/langchain-core/messages/content
    mime_type = mimetypes.guess_type(image_path.name)[0] or "image/png" # typically image/jpeg else image/png
    encoded = base64.b64encode(image_bytes).decode("ascii")
    message_with_base64 = HumanMessage(content_blocks=[
        TextContentBlock(type="text", text=prompt),
        ImageContentBlock(
            type="image",
            base64 = encoded,
            mime_type = mime_type
        )
    ])
    response = model.invoke([message_with_base64])
    
    # Cache the result
    text = response.text if isinstance(response.text, str) else str(response.text)
    payload = {"source_image": str(image_path), "text": text}
    cache_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return text

def fill_image_elements(elements:list[Element], model=None) -> list[Element]:
    filled = []
    for element in elements:
        if element.element_type not in PROMPTS:
            filled.append(element)
            continue
        
        text = describe_image(Path(element.metadata["image_path"]), element.element_type, model)
        if not text.strip():
            continue
        
        element.metadata["extraction_method"] = RESULT_TYPES[element.element_type]
        element.element_type = RESULT_TYPES[element.element_type]
        element.text = text
        filled.append(element)
    
    return filled