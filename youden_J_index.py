import pickle
import numpy as np
from sklearn.metrics import confusion_matrix, roc_auc_score, roc_curve, accuracy_score
import matplotlib.pyplot as plt

# ── Load your saved results ──────────────────────────────────────────
results = pickle.load(open("/content/drive/MyDrive/checkpoints/loo_results_v7.pkl", "rb"))
all_probs  = results['probs']    # dict: {'diru': [...], 'tractable': [...], 'lstm': [...]}
all_labels = results['labels']   # dict: same structure

# ── Re-evaluate with sensitivity-targeted threshold ──────────────────
TARGET_SENS = 0.75   # adjust this to whatever your clinical requirement is

print(f"{'='*65}")
print(f"RE-EVALUATION WITH SENSITIVITY TARGET >= {TARGET_SENS}")
print(f"{'='*65}")

for name in ('diru', 'tractable', 'lstm'):
    probs  = np.array(all_probs[name])
    labels = np.array(all_labels[name])

    # Search for threshold that hits target sensitivity with best specificity
    best_spec, best_thr = 0., 0.5
    best_sens_achieved  = 0.

    for thr in np.linspace(0.01, 0.99, 990):
        preds_t = (probs > thr).astype(int)
        tp = int(((preds_t == 1) & (labels == 1)).sum())
        fp = int(((preds_t == 1) & (labels == 0)).sum())
        fn = int(((preds_t == 0) & (labels == 1)).sum())
        tn = int(((preds_t == 0) & (labels == 0)).sum())
        sens_t = tp / (tp + fn) if (tp + fn) else 0.
        spec_t = tn / (tn + fp) if (tn + fp) else 0.

        if sens_t >= TARGET_SENS and spec_t > best_spec:
            best_spec          = spec_t
            best_thr           = thr
            best_sens_achieved = sens_t

    # Final metrics at chosen threshold
    preds = (probs > best_thr).astype(int)
    tn, fp, fn, tp = confusion_matrix(labels, preds).ravel()
    sens = tp / (tp + fn) if (tp + fn) else 0.
    spec = tn / (tn + fp) if (tn + fp) else 0.
    f1   = 2*tp / (2*tp + fp + fn) if (2*tp + fp + fn) else 0.
    acc  = accuracy_score(labels, preds)
    auc  = roc_auc_score(labels, probs)   # unchanged — threshold doesn't affect AUC

    print(f"\n  {name.upper()}")
    print(f"    Threshold : {best_thr:.3f}")
    print(f"    Sens      : {sens:.3f}   (target was >= {TARGET_SENS})")
    print(f"    Spec      : {spec:.3f}")
    print(f"    AUC       : {auc:.3f}   (identical to before)")
    print(f"    F1        : {f1:.3f}")
    print(f"    Acc       : {acc:.3f}")
    print(f"    TP={tp}  FP={fp}  FN={fn}  TN={tn}")

    if best_thr <= 0.02 or best_thr >= 0.98:
        print(f"    WARNING: threshold hit boundary — "
              f"model may not support target sensitivity of {TARGET_SENS}")
