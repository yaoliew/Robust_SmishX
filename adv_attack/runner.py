import json
from typing import Dict, List

import numpy as np
import torch
from autoattack import AutoAttack


def evaluate_with_autoattack(
    forward_fn,
    embeddings: torch.Tensor,
    labels: torch.Tensor,
    configs: List[Dict],
    batch_size: int = 2,
    log_prefix: str = "autoattack"
) -> Dict:
    results = {
        "evaluations": [],
        "summary": {},
    }

    with torch.no_grad():
        clean_logits = forward_fn(embeddings)
        clean_pred = clean_logits.argmax(dim=1)
        clean_acc = (clean_pred == labels).float().mean().item()
    results["clean_accuracy"] = clean_acc

    adv_accs: List[float] = []

    for cfg in configs:
        def fwd(x: torch.Tensor) -> torch.Tensor:
            return forward_fn(x)

        adversary = AutoAttack(
            fwd,
            norm=cfg["norm"],
            eps=cfg["epsilon"],
            version=cfg.get("version", "standard"),
            log_path=f"{log_prefix}_{cfg['epsilon']}_{cfg['norm']}.txt",
        )

        try:
            x_adv = adversary.run_standard_evaluation(embeddings, labels, bs=batch_size)
            with torch.no_grad():
                adv_logits = forward_fn(x_adv)
                adv_pred = adv_logits.argmax(dim=1)
                adv_acc = (adv_pred == labels).float().mean().item()
            results["evaluations"].append({
                **cfg,
                "adversarial_accuracy": adv_acc,
                "robust_accuracy": adv_acc,
                "num_samples": len(embeddings),
                "success": True,
            })
            adv_accs.append(adv_acc)
        except Exception as e:
            results["evaluations"].append({
                **cfg,
                "adversarial_accuracy": 0.0,
                "robust_accuracy": 0.0,
                "num_samples": len(embeddings),
                "success": False,
                "error": str(e),
            })

    if adv_accs:
        results["summary"] = {
            "min_adversarial_accuracy": float(np.min(adv_accs)),
            "max_adversarial_accuracy": float(np.max(adv_accs)),
            "mean_adversarial_accuracy": float(np.mean(adv_accs)),
            "std_adversarial_accuracy": float(np.std(adv_accs)),
            "robustness_gap": float(clean_acc - np.mean(adv_accs)),
        }

    return results






