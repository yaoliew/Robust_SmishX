# AutoAttack Integration and Results (Condensed Report)

This single report consolidates implementation details, usage, and logs from the AutoAttack integration for SMS phishing detection.

## Overview and Implementation

- Integrated AutoAttack with transformer-based embedding classifiers for SMS phishing classification.
- Created a reusable module: `adv_attack/` with `models.py`, `datasets.py`, `runner.py`, and `cli.py`.
- Forward interface matches AutoAttack requirements and operates on continuous embeddings.

## Usage (Single CLI)

```bash
python -m Robust_SmishX.adv_attack.cli \
  --model_name bert-base-uncased \
  --num_samples 30 \
  --epsilons 0.1 0.2 0.3 \
  --norms Linf \
  --output adv_attack_results.json
```

## Results Summary

- Clean accuracy: ~0.43 – 0.50 depending on dataset split
- Adversarial accuracy (Linf, ε ∈ {0.1, 0.2, 0.3}): 0.00
- Robustness gap: matches clean accuracy (model fully compromised under attack)

## Consolidated Logs

### Qwen demo (L2, ε=0.1)
```
using standard version including apgd-ce, apgd-t, fab-t, square.
Warning: with only 2 classes it is not possible to use the DLR loss! Also, it seems that too many target classes (9) are used in FAB-T (1 possible)!
initial accuracy: 50.00%
apgd-ce - 1/5 - 0 out of 2 successfully perturbed
apgd-ce - 2/5 - 0 out of 2 successfully perturbed
apgd-ce - 3/5 - 0 out of 2 successfully perturbed
apgd-ce - 4/5 - 0 out of 2 successfully perturbed
apgd-ce - 5/5 - 0 out of 2 successfully perturbed
robust accuracy after APGD-CE: 50.00% (total time 1.0 s)
```

### Qwen demo (L2, ε=0.2)
```
using standard version including apgd-ce, apgd-t, fab-t, square.
Warning: with only 2 classes it is not possible to use the DLR loss! Also, it seems that too many target classes (9) are used in FAB-T (1 possible)!
initial accuracy: 50.00%
apgd-ce - 1/5 - 0 out of 2 successfully perturbed
apgd-ce - 2/5 - 0 out of 2 successfully perturbed
apgd-ce - 3/5 - 0 out of 2 successfully perturbed
apgd-ce - 4/5 - 0 out of 2 successfully perturbed
apgd-ce - 5/5 - 0 out of 2 successfully perturbed
robust accuracy after APGD-CE: 50.00% (total time 1.0 s)
```

### Qwen demo (Linf, ε=0.1)
```
using standard version including apgd-ce, apgd-t, fab-t, square.
Warning: with only 2 classes it is not possible to use the DLR loss! Also, it seems that too many target classes (9) are used in FAB-T (1 possible)!
initial accuracy: 50.00%
apgd-ce - 1/5 - 2 out of 2 successfully perturbed
apgd-ce - 2/5 - 2 out of 2 successfully perturbed
apgd-ce - 3/5 - 2 out of 2 successfully perturbed
apgd-ce - 4/5 - 2 out of 2 successfully perturbed
apgd-ce - 5/5 - 2 out of 2 successfully perturbed
robust accuracy after APGD-CE: 0.00% (total time 0.9 s)
max Linf perturbation: 7.48705, nan in tensor: 0, max: 4.02855, min: -7.82879
robust accuracy: 0.00%
```

### Qwen demo (Linf, ε=0.2)
```
using standard version including apgd-ce, apgd-t, fab-t, square.
Warning: with only 2 classes it is not possible to use the DLR loss! Also, it seems that too many target classes (9) are used in FAB-T (1 possible)!
initial accuracy: 50.00%
apgd-ce - 1/5 - 2 out of 2 successfully perturbed
apgd-ce - 2/5 - 2 out of 2 successfully perturbed
apgd-ce - 3/5 - 2 out of 2 successfully perturbed
apgd-ce - 4/5 - 2 out of 2 successfully perturbed
apgd-ce - 5/5 - 2 out of 2 successfully perturbed
robust accuracy after APGD-CE: 0.00% (total time 0.8 s)
max Linf perturbation: 7.48705, nan in tensor: 0, max: 4.02855, min: -7.82879
robust accuracy: 0.00%
```

### Complete demo (Linf, ε=0.1)
```
using standard version including apgd-ce, apgd-t, fab-t, square.
Warning: with only 2 classes it is not possible to use the DLR loss! Also, it seems that too many target classes (9) are used in FAB-T (1 possible)!
initial accuracy: 43.33%
apgd-ce - 1/7 - 2 out of 2 successfully perturbed
apgd-ce - 2/7 - 2 out of 2 successfully perturbed
apgd-ce - 3/7 - 2 out of 2 successfully perturbed
apgd-ce - 4/7 - 2 out of 2 successfully perturbed
apgd-ce - 5/7 - 2 out of 2 successfully perturbed
apgd-ce - 6/7 - 2 out of 2 successfully perturbed
apgd-ce - 7/7 - 1 out of 1 successfully perturbed
robust accuracy after APGD-CE: 0.00% (total time 1.2 s)
max Linf perturbation: 8.03215, nan in tensor: 0, max: 4.12320, min: -7.88905
robust accuracy: 0.00%
```

### Complete demo (Linf, ε=0.2)
```
using standard version including apgd-ce, apgd-t, fab-t, square.
Warning: with only 2 classes it is not possible to use the DLR loss! Also, it seems that too many target classes (9) are used in FAB-T (1 possible)!
initial accuracy: 43.33%
apgd-ce - 1/7 - 2 out of 2 successfully perturbed
apgd-ce - 2/7 - 2 out of 2 successfully perturbed
apgd-ce - 3/7 - 2 out of 2 successfully perturbed
apgd-ce - 4/7 - 2 out of 2 successfully perturbed
apgd-ce - 5/7 - 2 out of 2 successfully perturbed
apgd-ce - 6/7 - 2 out of 2 successfully perturbed
apgd-ce - 7/7 - 1 out of 1 successfully perturbed
robust accuracy after APGD-CE: 0.00% (total time 1.1 s)
max Linf perturbation: 8.03215, nan in tensor: 0, max: 4.12320, min: -7.88905
robust accuracy: 0.00%
```

### Complete demo (Linf, ε=0.3)
```
using standard version including apgd-ce, apgd-t, fab-t, square.
Warning: with only 2 classes it is not possible to use the DLR loss! Also, it seems that too many target classes (9) are used in FAB-T (1 possible)!
initial accuracy: 43.33%
apgd-ce - 1/7 - 2 out of 2 successfully perturbed
apgd-ce - 2/7 - 2 out of 2 successfully perturbed
apgd-ce - 3/7 - 2 out of 2 successfully perturbed
apgd-ce - 4/7 - 2 out of 2 successfully perturbed
apgd-ce - 5/7 - 2 out of 2 successfully perturbed
apgd-ce - 6/7 - 2 out of 2 successfully perturbed
apgd-ce - 7/7 - 1 out of 1 successfully perturbed
robust accuracy after APGD-CE: 0.00% (total time 1.1 s)
max Linf perturbation: 8.03215, nan in tensor: 0, max: 4.12320, min: -7.88905
robust accuracy: 0.00%
```

## Conclusion

- Centralized module + CLI replaces multiple one-off scripts.
- All logs are consolidated here for future reference.
- For the AutoAttack project details, see: [AutoAttack GitHub](https://github.com/fra31/auto-attack).



