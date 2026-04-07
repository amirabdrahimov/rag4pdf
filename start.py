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
    parser.add_argument(
        "--mlflow-tracking-uri",
        help="MLflow tracking URI. Defaults to file:./mlruns.",
    )
    parser.add_argument(
        "--mlflow-experiment-name",
        help="MLflow experiment name. Defaults to rag4pdf.",
    )
    parser.add_argument(
        "--no-mlflow",
        action="store_true",
        help="Disable MLflow logging for this run.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    default_settings = Settings()
    settings = Settings(
        mlflow_enabled=not args.no_mlflow,
        mlflow_tracking_uri=args.mlflow_tracking_uri or default_settings.mlflow_tracking_uri,
        mlflow_experiment_name=args.mlflow_experiment_name or default_settings.mlflow_experiment_name,
    )
    assistant = PdfRagAssistant(settings=settings)
    assistant.initialize()
    result = assistant.answer(args.query, k=args.top_k)
    print(json.dumps(result, ensure_ascii=True, indent=2))


if __name__ == "__main__":
    main()
