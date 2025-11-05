import argparse
import json
import os
import sys
from typing import List

import torch

try:
    from .models import EmbeddingClassifier
    from .datasets import create_sample_sms_dataset
    from .runner import evaluate_with_autoattack
except ImportError:
    # Allow running as a standalone script
    CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
    PARENT_DIR = os.path.abspath(os.path.join(CURRENT_DIR, os.pardir))
    if PARENT_DIR not in sys.path:
        sys.path.append(PARENT_DIR)
    from models import EmbeddingClassifier
    from datasets import create_sample_sms_dataset
    from runner import evaluate_with_autoattack


def main():
    parser = argparse.ArgumentParser(description="AutoAttack CLI for SMS phishing detection")
    parser.add_argument("--model_name", type=str, default="bert-base-uncased")
    parser.add_argument("--num_samples", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--epsilons", nargs="+", type=float, default=[0.1, 0.2, 0.3])
    parser.add_argument("--norms", nargs="+", default=["Linf"])  # Linf default works best here
    parser.add_argument("--version", type=str, default="standard")
    parser.add_argument("--output", type=str, default="adv_attack_results.json")

    args = parser.parse_args()

    model = EmbeddingClassifier(args.model_name)
    texts, labels = create_sample_sms_dataset(args.num_samples)

    embeddings = model.encode_texts(texts)
    labels_tensor = torch.tensor(labels).to(next(model.parameters()).device)

    configs = [{"epsilon": e, "norm": n, "version": args.version} for e in args.epsilons for n in args.norms]
    results = evaluate_with_autoattack(model.forward, embeddings, labels_tensor, configs, batch_size=args.batch_size, log_prefix="adv_eval_log")

    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)

    print(f"Saved results to {args.output}")


if __name__ == "__main__":
    main()


