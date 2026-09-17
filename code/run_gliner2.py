"""Extract candidate knowledge-graph nodes and edges from a Markdown note."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import time
from pathlib import Path
from typing import Any

from gliner2 import GLiNER2


DEFAULT_MODEL = "fastino/gliner2-base-v1"

ENTITY_SCHEMA = {
    "model": "Named machine learning model or model family",
    "method": "Method, technique, procedure, or computational approach",
    "concept": "Important technical or scientific concept",
    "property": "Desired quality, advantage, limitation, or measurable property",
    "data_source": "Text, note, document, dataset, or other source of information",
}

RELATION_SCHEMA = {
    "causes": "The first entity is explicitly said to cause or produce the second",
    "increases": "The first entity is explicitly said to increase the second",
    "reduces": "The first entity is explicitly said to reduce the second",
    "enables": "The first entity makes the second possible",
    "requires": "The first entity needs the second",
    "uses": "The first entity uses the second as a method or component",
    "supports": "The first entity provides evidence or support for the second",
    "is_part_of": "The first entity is a component or stage of the second",
}


def markdown_to_text(markdown: str) -> str:
    """Remove frontmatter and lightweight Markdown markup without rewriting prose."""
    text = re.sub(r"\A---\n.*?\n---\n", "", markdown, flags=re.DOTALL)
    text = re.sub(r"```(?:\w+)?\n(.*?)```", r"\1", text, flags=re.DOTALL)
    text = re.sub(r"^#{1,6}\s+", "", text, flags=re.MULTILINE)
    text = re.sub(r"[*_`]", "", text)
    text = re.sub(r"\[(.*?)\]\(.*?\)", r"\1", text)
    return re.sub(r"\n{3,}", "\n\n", text).strip()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    args = parser.parse_args()

    source = args.input.read_text(encoding="utf-8")
    text = markdown_to_text(source)

    load_started = time.perf_counter()
    extractor = GLiNER2.from_pretrained(args.model)
    load_seconds = time.perf_counter() - load_started

    schema = (
        extractor.create_schema()
        .entities(ENTITY_SCHEMA)
        .relations(RELATION_SCHEMA)
    )

    inference_started = time.perf_counter()
    result = extractor.extract(
        text,
        schema,
        include_confidence=True,
        include_spans=True,
    )
    inference_seconds = time.perf_counter() - inference_started

    payload: dict[str, Any] = {
        "model": args.model,
        "source": str(args.input),
        "source_sha256": hashlib.sha256(source.encode()).hexdigest(),
        "characters_analyzed": len(text),
        "entity_schema": ENTITY_SCHEMA,
        "relation_schema": RELATION_SCHEMA,
        "load_seconds": round(load_seconds, 3),
        "inference_seconds": round(inference_seconds, 3),
        "result": result,
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, default=str) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(payload, indent=2, ensure_ascii=False, default=str))


if __name__ == "__main__":
    main()
