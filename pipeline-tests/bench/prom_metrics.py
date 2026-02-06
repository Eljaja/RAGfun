from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class Sample:
    name: str
    labels: dict[str, str]
    value: float


def _parse_labels(s: str) -> dict[str, str]:
    # Very small parser for Prometheus exposition labels: k="v",...
    out: dict[str, str] = {}
    s = s.strip()
    if not s:
        return out
    i = 0
    n = len(s)
    while i < n:
        # key
        j = s.find("=", i)
        if j < 0:
            break
        key = s[i:j].strip()
        i = j + 1
        if i >= n or s[i] != '"':
            break
        i += 1
        # value (supports escaped quotes and backslashes)
        val = []
        while i < n:
            ch = s[i]
            if ch == "\\" and i + 1 < n:
                val.append(s[i + 1])
                i += 2
                continue
            if ch == '"':
                i += 1
                break
            val.append(ch)
            i += 1
        out[key] = "".join(val)
        # skip comma
        while i < n and s[i] in " ,":
            i += 1
    return out


def parse_prometheus_text(text: str) -> list[Sample]:
    samples: list[Sample] = []
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        # examples:
        # metric_name 123
        # metric_name{a="b"} 123
        name_part, _, value_part = line.rpartition(" ")
        if not name_part:
            continue
        try:
            value = float(value_part.strip())
        except Exception:
            continue

        if "{" in name_part and name_part.endswith("}"):
            name, _, rest = name_part.partition("{")
            labels_raw = rest[:-1]
            labels = _parse_labels(labels_raw)
            samples.append(Sample(name=name, labels=labels, value=value))
        else:
            samples.append(Sample(name=name_part.strip(), labels={}, value=value))
    return samples


def _key_for(sample: Sample, *, drop_labels: set[str]) -> tuple[str, tuple[tuple[str, str], ...]]:
    labels = tuple(sorted([(k, v) for k, v in sample.labels.items() if k not in drop_labels]))
    return (sample.name, labels)


def diff_samples(
    before: list[Sample], after: list[Sample], *, drop_labels: set[str] | None = None
) -> dict[tuple[str, tuple[tuple[str, str], ...]], float]:
    """
    Returns (metric_name, labels_without_drop) -> delta_value
    """
    drop = drop_labels or set()
    b: dict[tuple[str, tuple[tuple[str, str], ...]], float] = {}
    a: dict[tuple[str, tuple[tuple[str, str], ...]], float] = {}

    for s in before:
        b[_key_for(s, drop_labels=drop)] = s.value
    for s in after:
        a[_key_for(s, drop_labels=drop)] = s.value

    keys = set(b.keys()) | set(a.keys())
    return {k: a.get(k, 0.0) - b.get(k, 0.0) for k in keys}


def _as_float_le(le: str) -> float:
    if le == "+Inf":
        return float("inf")
    return float(le)


def histogram_quantile_from_cumulative(
    *,
    buckets: list[tuple[str, float]],
    quantile: float,
) -> float | None:
    """
    buckets: list of (le, cumulative_count) sorted by le.
    Returns le boundary for which cumulative_count crosses quantile*total.
    """
    if not buckets:
        return None
    buckets_sorted = sorted(buckets, key=lambda x: _as_float_le(x[0]))
    total = buckets_sorted[-1][1]
    if total <= 0:
        return None
    target = total * quantile
    for le, c in buckets_sorted:
        if c >= target:
            v = _as_float_le(le)
            if v == float("inf"):
                return None
            return v
    return None


def extract_histogram(
    deltas: dict[tuple[str, tuple[tuple[str, str], ...]], float],
    *,
    metric_prefix: str,
    match_labels: dict[str, str],
    group_by: list[str],
) -> dict[tuple[str, ...], dict[str, Any]]:
    """
    Extract histogram deltas for metric_prefix (without _bucket/_sum/_count suffix).

    Returns:
      group_key -> {"buckets": [(le, cnt)], "sum": float, "count": float, "labels": {...}}
    """
    out: dict[tuple[str, ...], dict[str, Any]] = {}

    def labels_to_dict(lbls: tuple[tuple[str, str], ...]) -> dict[str, str]:
        return {k: v for k, v in lbls}

    for (name, lbls_t), dv in deltas.items():
        if not (name == f"{metric_prefix}_bucket" or name == f"{metric_prefix}_sum" or name == f"{metric_prefix}_count"):
            continue
        labels = labels_to_dict(lbls_t)
        ok = True
        for k, v in match_labels.items():
            if labels.get(k) != v:
                ok = False
                break
        if not ok:
            continue

        gk = tuple(labels.get(k, "") for k in group_by)
        slot = out.setdefault(gk, {"buckets": [], "sum": 0.0, "count": 0.0, "labels": labels})
        if name.endswith("_bucket"):
            le = labels.get("le", "")
            if le:
                slot["buckets"].append((le, float(dv)))
        elif name.endswith("_sum"):
            slot["sum"] = float(dv)
        elif name.endswith("_count"):
            slot["count"] = float(dv)

    return out

















