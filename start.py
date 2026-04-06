from __future__ import annotations

import argparse
import json

from rag4pdf import PdfRagAssistant, Settings


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run RAG over PDFs in ./data using Ollama.")
    parser.add_argument(
        "--query",
        help="Question to ask against the indexed PDF content.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=4,
        help="Number of retrieved chunks to include in context.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    assistant = PdfRagAssistant(settings=Settings())
    assistant.initialize()
    result = assistant.answer(args.query, k=args.top_k)
    print(json.dumps(result, ensure_ascii=True, indent=2))


if __name__ == "__main__":
    main()
