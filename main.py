"""Run the whole analysis + training pipeline:  python main.py"""
import argparse
import logging

from config import Config
from src.pipeline import DrugReviewPipeline


def main() -> None:
    parser = argparse.ArgumentParser(description="Drug review sentiment pipeline")
    parser.add_argument("--transformer", action="store_true", help="also score with a Hugging Face model (slow)")
    parser.add_argument("--n-topics", type=int, default=None, help="topics per review column (default 6)")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    cfg = Config()
    if args.n_topics:
        cfg.n_topics = args.n_topics
    DrugReviewPipeline(cfg, use_transformer=args.transformer).run()


if __name__ == "__main__":
    main()
