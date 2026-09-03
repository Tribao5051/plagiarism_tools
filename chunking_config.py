from chonkie import (
    OverlapRefinery,
    RecursiveChunker,
    SemanticChunker,
    TokenChunker,
)
from project_config import get_chunking_profile, load_project_config


PROJECT_CONFIG = load_project_config()
RECURSIVE_CONFIG = get_chunking_profile(
    "recursive_character", PROJECT_CONFIG)
SEMANTIC_CONFIG = get_chunking_profile("semantic", PROJECT_CONFIG)
TOKEN_CONFIG = get_chunking_profile("token", PROJECT_CONFIG)


def load_chunking_config():
    return get_chunking_profile(config=PROJECT_CONFIG)


def create_chunker_tools(config, include_overlap=True):
    if config["strategy"] == "token":
        chunker = TokenChunker(
            tokenizer=config["tokenizer"],
            chunk_size=config["chunk_size"],
            chunk_overlap=(
                config["chunk_overlap"] if include_overlap else 0
            ),
        )
        return chunker, None

    if config["strategy"] == "semantic":
        chunker = SemanticChunker(
            embedding_model=config["semantic_embedding_model"],
            threshold=config["semantic_threshold"],
            chunk_size=config["chunk_size"],
            similarity_window=config["semantic_similarity_window"],
            min_sentences_per_chunk=config["semantic_min_sentences"],
            min_characters_per_sentence=config["min_chunk_characters"],
        )
        return chunker, None

    chunker = RecursiveChunker(
        tokenizer="character",
        chunk_size=config["chunk_size"],
        min_characters_per_chunk=config["min_chunk_characters"],
    )
    refinery = None
    if include_overlap and config["chunk_overlap"]:
        refinery = OverlapRefinery(
            tokenizer="character",
            context_size=config["chunk_overlap"],
            method="suffix",
            merge=True,
        )
    return chunker, refinery


def chunk_text(text, chunker, refinery=None):
    chunks = chunker.chunk(text)
    if refinery is not None:
        chunks = refinery(chunks)
    return [chunk.text for chunk in chunks if chunk.text.strip()]


def chunk_text_with_native_spans(text, chunker, refinery=None):
    chunks = chunker.chunk(text)
    if refinery is not None:
        chunks = refinery(chunks)

    chunk_texts = []
    spans = []
    for chunk in chunks:
        start = getattr(chunk, "start_index", None)
        end = getattr(chunk, "end_index", None)
        if (
            isinstance(start, int)
            and isinstance(end, int)
            and 0 <= start < end <= len(text)
        ):
            chunk_value = text[start:end]
            span = (start, end)
        else:
            chunk_value = chunk.text
            span = (-1, -1)
        if chunk_value.strip():
            chunk_texts.append(chunk_value)
            spans.append(span)
    return chunk_texts, spans
