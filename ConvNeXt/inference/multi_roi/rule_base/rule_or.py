# inference/multi_roi/rule_base/rule_or.py
from __future__ import annotations

from typing import Any

import numpy as np

try:
    import torch
except ImportError:  # pragma: no cover
    torch = None


def _to_numpy_2d(x: Any) -> np.ndarray:
    if torch is not None and isinstance(x, torch.Tensor):
        arr = x.detach().cpu().float().numpy()
    else:
        arr = np.asarray(x, dtype=np.float64)

    if arr.ndim != 2:
        raise ValueError(f"Expected shape [V, C], got {arr.shape}")
    if arr.shape[0] != 3:
        raise ValueError(f"Expected exactly 3 views, got {arr.shape[0]}")
    if arr.shape[1] < 1:
        raise ValueError("Expected at least one class")
    return arr


def _row_softmax(x: np.ndarray) -> np.ndarray:
    safe = np.nan_to_num(x, nan=-np.inf, posinf=np.inf, neginf=-np.inf)
    row_max = np.max(safe, axis=1, keepdims=True)
    row_max[~np.isfinite(row_max)] = 0.0
    shifted = safe - row_max
    exp_x = np.exp(np.clip(shifted, a_min=-80.0, a_max=80.0))
    exp_x[~np.isfinite(exp_x)] = 0.0
    denom = exp_x.sum(axis=1, keepdims=True)
    bad = denom.squeeze(1) <= 0
    if np.any(bad):
        exp_x[bad] = 1.0
        denom[bad] = exp_x.shape[1]
    return exp_x / denom


def _coerce_probs(x: np.ndarray, from_logits: bool | None) -> np.ndarray:
    if from_logits is True:
        return _row_softmax(x)

    safe = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
    row_sums = safe.sum(axis=1, keepdims=True)
    looks_like_probs = (
        np.all(safe >= 0.0)
        and np.all(np.isfinite(safe))
        and np.all(row_sums.squeeze(1) > 0.0)
        and np.allclose(row_sums, 1.0, atol=1e-4)
    )

    if from_logits is False or looks_like_probs:
        probs = np.clip(safe, a_min=0.0, a_max=None)
        denom = probs.sum(axis=1, keepdims=True)
        bad = denom.squeeze(1) <= 0
        if np.any(bad):
            probs[bad] = 1.0
            denom[bad] = probs.shape[1]
        return probs / denom

    return _row_softmax(x)


def _top2_stats(probs_row: np.ndarray) -> tuple[int, float, float, float]:
    top1_idx = int(np.argmax(probs_row))
    top1_prob = float(probs_row[top1_idx])

    if probs_row.shape[0] < 2:
        top2_prob = 0.0
    else:
        top2_idx = int(np.argpartition(probs_row, -2)[-2])
        if top2_idx == top1_idx:
            sorted_idx = np.argsort(probs_row)
            top2_idx = int(sorted_idx[-2])
        top2_prob = float(probs_row[top2_idx])

    gap = top1_prob - top2_prob
    return top1_idx, top1_prob, top2_prob, gap


def select_or_rule_prediction(
    scores: Any,
    probability_threshold: float,
    gap_threshold: float,
    from_logits: bool | None = None,
    return_details: bool = True,
):
    """
    Apply 3-view OR rule on probabilities or logits.

    Args:
        scores:
            Array-like of shape [3, C]. Can be probabilities or logits.
        probability_threshold:
            Minimum top-1 probability required to register a confident view.
        gap_threshold:
            Minimum top1-top2 gap required to register a confident view.
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

    candidates = []
    per_view = []

    for view_idx in range(probs.shape[0]):
        top1_idx, top1_prob, top2_prob, gap = _top2_stats(probs[view_idx])
        passed = (top1_prob >= probability_threshold) and (gap >= gap_threshold)
        info = {
            "view_idx": view_idx,
            "class_idx": top1_idx,
            "top1_prob": top1_prob,
            "top2_prob": top2_prob,
            "gap": gap,
            "passed": passed,
        }
        per_view.append(info)
        if passed:
            candidates.append(info)

    if candidates:
        candidates.sort(
            key=lambda item: (-item["gap"], -item["top1_prob"], item["view_idx"])
        )
        chosen = candidates[0]
        pred = int(chosen["class_idx"])
        mode = "or_rule"
        probs_mean = None
    else:
        probs_mean = probs.mean(axis=0)
        pred = int(np.argmax(probs_mean))
        chosen = None
        mode = "mean_fallback"

    if not return_details:
        return pred

    details = {
        "mode": mode,
        "pred": pred,
        "selected_view": None if chosen is None else int(chosen["view_idx"]),
        "selected_class": None if chosen is None else int(chosen["class_idx"]),
        "selected_top1_prob": None if chosen is None else float(chosen["top1_prob"]),
        "selected_top2_prob": None if chosen is None else float(chosen["top2_prob"]),
        "selected_gap": None if chosen is None else float(chosen["gap"]),
        "probability_threshold": float(probability_threshold),
        "gap_threshold": float(gap_threshold),
        "per_view": per_view,
        "candidates": candidates,
        "probs": probs,
        "probs_mean": probs_mean,
    }
    return pred, details


__all__ = ["select_or_rule_prediction"]
