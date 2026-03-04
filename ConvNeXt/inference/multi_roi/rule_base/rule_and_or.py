# inference/multi_roi/rule_base/rule_and_or.py
from __future__ import annotations

from typing import Any

import numpy as np

from .rule_or import _coerce_probs, _to_numpy_2d, _top2_stats


def select_and_or_rule_prediction(
    scores: Any,
    probability_threshold: float,
    gap_threshold: float,
    from_logits: bool | None = None,
    return_details: bool = True,
):
    """
    Apply 3-stage decision rule for 3-view classification:
    1) AND
    2) OR
    3) Mean fallback
    """
    scores_2d = _to_numpy_2d(scores)
    probs = _coerce_probs(scores_2d, from_logits=from_logits)

    per_view = []
    top1_classes = []
    or_candidates = []

    for view_idx in range(probs.shape[0]):
        class_idx, top1_prob, top2_prob, gap = _top2_stats(probs[view_idx])
        passed = (top1_prob >= probability_threshold) and (gap >= gap_threshold)
        info = {
            "view_idx": view_idx,
            "class_idx": class_idx,
            "top1_prob": top1_prob,
            "top2_prob": top2_prob,
            "gap": gap,
            "passed": passed,
        }
        per_view.append(info)
        top1_classes.append(class_idx)
        if passed:
            or_candidates.append(info)

    all_views_confident = all(item["passed"] for item in per_view)
    same_top1_class = len(set(top1_classes)) == 1
    and_pass = all_views_confident and same_top1_class

    chosen_view = None
    chosen = None
    probs_mean = None

    if and_pass:
        stage = "AND"
        pred = int(top1_classes[0])
    else:
        if or_candidates:
            or_candidates.sort(
                key=lambda item: (-item["gap"], -item["top1_prob"], item["view_idx"])
            )
            chosen = or_candidates[0]
            chosen_view = int(chosen["view_idx"])
            pred = int(chosen["class_idx"])
            stage = "OR"
        else:
            probs_mean = probs.mean(axis=0)
            pred = int(np.argmax(probs_mean))
            stage = "MEAN"

    if not return_details:
        return pred

    details = {
        "stage": stage,
        "pred": pred,
        "and_pass": bool(and_pass),
        "or_pass": bool(len(or_candidates) > 0),
        "fallback_used": stage == "MEAN",
        "chosen_view": chosen_view,
        "chosen_class": None if chosen is None else int(chosen["class_idx"]),
        "chosen_top1_prob": None if chosen is None else float(chosen["top1_prob"]),
        "chosen_top2_prob": None if chosen is None else float(chosen["top2_prob"]),
        "chosen_gap": None if chosen is None else float(chosen["gap"]),
        "all_views_confident": bool(all_views_confident),
        "same_top1_class": bool(same_top1_class),
        "consensus_class": int(top1_classes[0]) if same_top1_class else None,
        "probability_threshold": float(probability_threshold),
        "gap_threshold": float(gap_threshold),
        "per_view": per_view,
        "or_candidates": or_candidates,
        "probs": probs,
        "probs_mean": probs_mean,
    }
    return pred, details


__all__ = ["select_and_or_rule_prediction"]
