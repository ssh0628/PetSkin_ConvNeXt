# inference/multi_roi/rule_base/rule_and.py
from __future__ import annotations

from typing import Any

from .rule_or import _coerce_probs, _to_numpy_2d, _top2_stats


def select_and_rule_prediction(
    scores: Any,
    probability_threshold: float,
    gap_threshold: float,
    from_logits: bool | None = None,
    return_details: bool = True,
):
    """
    Apply 3-view AND rule on probabilities or logits.

    Args:
        scores:
            Array-like of shape [3, C]. Can be probabilities or logits.
        probability_threshold:
            Minimum top-1 probability required for each view.
        gap_threshold:
            Minimum top1-top2 gap required for each view.
        from_logits:
            True to always apply softmax, False to always normalize as probs,
            None to auto-detect.
        return_details:
            If True, return a metadata dict together with pred.

    Returns:
        If return_details is False:
            pred (int)
        Else:
            (pred, details)
    """
    scores_2d = _to_numpy_2d(scores)
    probs = _coerce_probs(scores_2d, from_logits=from_logits)

    per_view = []
    top1_classes = []

    for view_idx in range(probs.shape[0]):
        top1_idx, top1_prob, top2_prob, gap = _top2_stats(probs[view_idx])
        passed = (top1_prob >= probability_threshold) and (gap >= gap_threshold)
        per_view.append(
            {
                "view_idx": view_idx,
                "class_idx": top1_idx,
                "top1_prob": top1_prob,
                "top2_prob": top2_prob,
                "gap": gap,
                "passed": passed,
            }
        )
        top1_classes.append(top1_idx)

    all_views_confident = all(item["passed"] for item in per_view)
    same_top1_class = len(set(top1_classes)) == 1
    and_pass = all_views_confident and same_top1_class

    if and_pass:
        pred = int(top1_classes[0])
        fallback_used = False
        probs_mean = None
        mode = "and_rule"
    else:
        probs_mean = probs.mean(axis=0)
        pred = int(probs_mean.argmax())
        fallback_used = True
        mode = "mean_fallback"

    if not return_details:
        return pred

    details = {
        "mode": mode,
        "pred": pred,
        "and_pass": bool(and_pass),
        "fallback_used": bool(fallback_used),
        "all_views_confident": bool(all_views_confident),
        "same_top1_class": bool(same_top1_class),
        "consensus_class": int(top1_classes[0]) if same_top1_class else None,
        "probability_threshold": float(probability_threshold),
        "gap_threshold": float(gap_threshold),
        "per_view": per_view,
        "probs": probs,
        "probs_mean": probs_mean,
    }
    return pred, details


__all__ = ["select_and_rule_prediction"]
