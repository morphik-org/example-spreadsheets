from __future__ import annotations

import io
from typing import Any, Dict, List, Sequence

from morphik import Morphik
from openai import OpenAI

DEFAULT_PAGE_OUTPUT_FORMAT = "url"
DEFAULT_CHUNK_OUTPUT_FORMAT = "url"


def build_tools(file_ids: Sequence[str]) -> List[Dict[str, Any]]:
    return [
        {
            "type": "function",
            "name": "retrieve_chunks",
            "description": (
                "Retrieve relevant chunks from Morphik using ColPali mode. "
                "Only provide a search query and the number of chunks to fetch."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Search query text."},
                    "k": {"type": "integer", "description": "Number of chunks to retrieve.", "minimum": 1},
                },
                "required": ["query"],
            },
        },
        {
            "type": "function",
            "name": "get_page_range",
            "description": (
                "Get pages or chunks within a specific range. Provide document_id and either "
                "start_page/end_page for page images, or start_chunk/end_chunk for chunk text."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "document_id": {"type": "string", "description": "Morphik document external ID."},
                    "start_page": {"type": "integer", "description": "Start page number (1-indexed)."},
                    "end_page": {"type": "integer", "description": "End page number (1-indexed)."},
                    "start_chunk": {"type": "integer", "description": "Start chunk number (1-indexed)."},
                    "end_chunk": {"type": "integer", "description": "End chunk number (1-indexed)."},
                },
                "required": ["document_id"],
            },
        },
        {
            "type": "function",
            "name": "list_documents",
            "description": "List documents available in Morphik.",
            "parameters": {
                "type": "object",
                "properties": {
                    "skip": {"type": "integer", "description": "Number of documents to skip.", "minimum": 0},
                    "limit": {"type": "integer", "description": "Maximum number of documents to return.", "minimum": 1},
                    "completed_only": {"type": "boolean", "description": "Only return completed documents."},
                    "sort_by": {
                        "type": "string",
                        "description": "Field to sort by.",
                        "enum": ["created_at", "updated_at", "filename", "external_id"],
                    },
                    "sort_direction": {
                        "type": "string",
                        "description": "Sort direction.",
                        "enum": ["asc", "desc"],
                    },
                },
            },
        },
        {
            "type": "function",
            "name": "load_file_for_execution",
            "description": (
                "Load a Morphik document into the code execution environment. "
                "Provide the document external ID."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "document_external_id": {"type": "string", "description": "Morphik document external ID."},
                },
                "required": ["document_external_id"],
            },
        },
        {
            "type": "code_interpreter",
            "container": {
                "type": "auto",
                "memory_limit": "4g",
                "file_ids": list(file_ids),
            },
        },
    ]


def run_tool_call(
    name: str,
    arguments: Dict[str, Any],
    *,
    morphik: Morphik,
    openai_client: OpenAI,
    state: Dict[str, Any],
) -> Dict[str, Any]:
    if name == "retrieve_chunks":
        return _retrieve_chunks(morphik, arguments)
    if name == "get_page_range":
        return _get_page_range(morphik, arguments)
    if name == "list_documents":
        return _list_documents(morphik, arguments)
    if name == "load_file_for_execution":
        return _load_file_for_execution(morphik, openai_client, arguments, state)
    raise ValueError(f"Unknown tool: {name}")


def _retrieve_chunks(morphik: Morphik, arguments: Dict[str, Any]) -> Dict[str, Any]:
    query = arguments.get("query")
    if not query:
        raise ValueError("query is required")
    k = int(arguments.get("k") or 4)
    chunks = morphik.retrieve_chunks(
        query=query,
        k=k,
        use_colpali=True,
        output_format=DEFAULT_CHUNK_OUTPUT_FORMAT,
    )
    return {"query": query, "k": k, "chunks": [_serialize_chunk(chunk) for chunk in chunks]}


def _get_page_range(morphik: Morphik, arguments: Dict[str, Any]) -> Dict[str, Any]:
    document_id = arguments.get("document_id")
    if not document_id:
        raise ValueError("document_id is required")
    start_page = arguments.get("start_page")
    end_page = arguments.get("end_page")
    start_chunk = arguments.get("start_chunk")
    end_chunk = arguments.get("end_chunk")

    if start_page is not None and end_page is not None:
        pages = morphik.extract_document_pages(
            document_id=document_id,
            start_page=int(start_page),
            end_page=int(end_page),
            output_format=DEFAULT_PAGE_OUTPUT_FORMAT,
        )
        return {"type": "pages", **pages.model_dump()}

    if start_chunk is not None and end_chunk is not None:
        start_chunk = int(start_chunk)
        end_chunk = int(end_chunk)
        if end_chunk < start_chunk:
            raise ValueError("end_chunk must be >= start_chunk")
        sources = [
            {"document_id": document_id, "chunk_number": chunk_number}
            for chunk_number in range(start_chunk, end_chunk + 1)
        ]
        chunks = morphik.batch_get_chunks(
            sources=sources,
            use_colpali=True,
            output_format=DEFAULT_CHUNK_OUTPUT_FORMAT,
        )
        return {
            "type": "chunks",
            "document_id": document_id,
            "start_chunk": start_chunk,
            "end_chunk": end_chunk,
            "chunks": [_serialize_chunk(chunk) for chunk in chunks],
        }

    raise ValueError("Provide start_page/end_page or start_chunk/end_chunk")


def _list_documents(morphik: Morphik, arguments: Dict[str, Any]) -> Dict[str, Any]:
    skip = int(arguments.get("skip") or 0)
    limit = int(arguments.get("limit") or 100)
    completed_only = arguments.get("completed_only", False)
    if isinstance(completed_only, str):
        completed_only = completed_only.lower() == "true"
    sort_by = arguments.get("sort_by", "updated_at")
    sort_direction = arguments.get("sort_direction", "desc")
    response = morphik.list_documents(
        skip=skip,
        limit=limit,
        completed_only=completed_only,
        sort_by=sort_by,
        sort_direction=sort_direction,
    )
    return response.model_dump()


def _load_file_for_execution(
    morphik: Morphik,
    openai_client: OpenAI,
    arguments: Dict[str, Any],
    state: Dict[str, Any],
) -> Dict[str, Any]:
    document_id = arguments.get("document_external_id")
    if not document_id:
        raise ValueError("document_external_id is required")

    loaded_files = state.setdefault("loaded_files", {})
    if document_id in loaded_files:
        return {
            "document_id": document_id,
            "file_id": loaded_files[document_id]["file_id"],
            "filename": loaded_files[document_id]["filename"],
            "status": "already_loaded",
        }

    document = morphik.get_document(document_id)
    filename = document.filename or f"{document_id}"
    file_bytes = morphik.get_document_file(document_id)

    file_buffer = io.BytesIO(file_bytes)
    file_buffer.seek(0)
    file_obj = openai_client.files.create(
        file=(filename, file_buffer),
        purpose="assistants",
    )

    state.setdefault("file_ids", set()).add(file_obj.id)
    loaded_files[document_id] = {"file_id": file_obj.id, "filename": filename}

    return {
        "document_id": document_id,
        "file_id": file_obj.id,
        "filename": filename,
        "status": "loaded",
    }


def _serialize_chunk(chunk: Any) -> Dict[str, Any]:
    content = chunk.content
    if not isinstance(content, str):
        if hasattr(content, "size"):
            content = f"<image size={getattr(content, 'size', '')}>"
        else:
            content = str(content)

    return {
        "document_id": chunk.document_id,
        "chunk_number": chunk.chunk_number,
        "score": chunk.score,
        "content": content,
        "metadata": chunk.metadata,
        "content_type": chunk.content_type,
        "filename": chunk.filename,
        "download_url": chunk.download_url,
    }
