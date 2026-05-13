"""Paired t-test for per-query significance testing."""
import numpy as np
from scipy.stats import ttest_rel


def paired_ttest(
    per_query_a: dict[str, dict[str, float]],
    per_query_b: dict[str, dict[str, float]],
    metric: str = "ndcg_cut_10",
) -> tuple[float, float, int]:
    """
    Paired two-tailed t-test between system A and system B.

    Args:
        per_query_a / _b : {qid: {metric: value}}
        metric           : which metric to compare

    Returns:
        (t_statistic, p_value, n_queries)
    """
    common = sorted(set(per_query_a) & set(per_query_b))
    if len(common) < 2:
        return 0.0, 1.0, 0

    a = [per_query_a[q].get(metric, 0.0) for q in common]
    b = [per_query_b[q].get(metric, 0.0) for q in common]
    t, p = ttest_rel(a, b)
    return float(t), float(p), len(common)


def is_significant(p_value: float, alpha: float = 0.05) -> bool:
    return p_value < alpha
