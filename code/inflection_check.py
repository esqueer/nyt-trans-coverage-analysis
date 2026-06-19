#!/usr/bin/env python3
"""Pin down WHEN the framing shift happens.

For each key dimension: build a quarterly series, then scan all quarter
boundaries for the split that maximizes the before/after separation
(Welch t-statistic). If the user's editorial-timeline story is right,
the best split should land around mid-2022 for the framing dims.

Model handling:
  - tone dims (conflict, normalizing): Anthropic+Gemini mean (OpenAI
    doesn't use these scales — including it would just shrink everything)
  - all other dims: mean of all available models per article
"""
import json
import math
from collections import defaultdict
from pathlib import Path

DATA = Path(__file__).parent / "dist" / "analysis_output" / "collected_articles.json"

with open(DATA, encoding="utf-8") as f:
    raw = json.load(f)
articles = raw.get("articles") if isinstance(raw, dict) else raw

# OpenAI was removed from the ensemble entirely (scale non-engagement);
# every dimension now averages Anthropic+Gemini+Qwen.
PROVIDERS = ["anthropic", "gemini", "ollama"]

KEY_DIMS = [
    ("framing_direction", "skepticism_to_affirming"),
    ("framing_direction", "restrictive_to_protective"),
    ("framing_tone", "conflict_framing"),
    ("framing_tone", "normalizing"),
    ("voice", "trans_voice_centrality"),
    ("voice", "opponent_voice_centrality"),
    ("topic", "medical"),
    ("topic", "youth_focus"),
]


def article_score(a, section, dim):
    cons = a.get("llm_consensus") or {}
    pms = cons.get("per_model_scores") or {}
    providers = PROVIDERS
    vals = []
    for p in providers:
        pm = pms.get(p) or {}
        if pm.get("error"):
            continue
        v = (pm.get(section) or {}).get(dim)
        if isinstance(v, (int, float)):
            vals.append(float(v))
    return sum(vals) / len(vals) if vals else None


def quarter(a):
    d = a.get("pub_date") or ""
    try:
        y, m = int(d[:4]), int(d[5:7])
    except (ValueError, IndexError):
        return None
    if y < 2014:
        return None
    return f"{y}Q{(m - 1) // 3 + 1}"


def qsort_key(q):
    return (int(q[:4]), int(q[-1]))


# Volume per quarter
vol = defaultdict(int)
for a in articles:
    q = quarter(a)
    if q:
        vol[q] += 1

quarters = sorted(vol.keys(), key=qsort_key)
print("ARTICLE VOLUME BY QUARTER")
for q in quarters:
    bar = "#" * (vol[q] // 5)
    print(f"  {q}  {vol[q]:4d}  {bar}")

# Per-dim: quarterly means + best changepoint
print("\n\nCHANGEPOINT SCAN (best before/after split per dimension)")
print("Candidates: every quarter boundary 2018Q1..2024Q4. Welch t-stat.\n")

for section, dim in KEY_DIMS:
    # collect (date-sortable quarter, score) pairs
    pairs = []
    for a in articles:
        q = quarter(a)
        if not q:
            continue
        s = article_score(a, section, dim)
        if s is not None:
            pairs.append((qsort_key(q), s))
    pairs.sort()

    # quarterly means for display
    by_q = defaultdict(list)
    for k, s in pairs:
        by_q[k].append(s)

    # changepoint scan
    candidates = [(y, q) for y in range(2018, 2025) for q in (1, 2, 3, 4)]
    best = None
    for cut in candidates:
        before = [s for k, s in pairs if k < cut]
        after = [s for k, s in pairs if k >= cut]
        if len(before) < 100 or len(after) < 100:
            continue
        mb = sum(before) / len(before)
        ma = sum(after) / len(after)
        vb = sum((x - mb) ** 2 for x in before) / (len(before) - 1)
        va = sum((x - ma) ** 2 for x in after) / (len(after) - 1)
        se = math.sqrt(vb / len(before) + va / len(after))
        if se == 0:
            continue
        t = (ma - mb) / se
        if best is None or abs(t) > abs(best[1]):
            best = (cut, t, mb, ma)

    cut, t, mb, ma = best
    cut_label = f"{cut[0]}Q{cut[1]}"
    print(f"{section}.{dim}")
    print(f"  best split: {cut_label}   t={t:+.1f}   mean {mb:+.2f} -> {ma:+.2f}   delta={ma - mb:+.2f}")
    # show 2021-2023 quarterly detail around the likely inflection
    detail = []
    for y in (2021, 2022, 2023):
        for qq in (1, 2, 3, 4):
            k = (y, qq)
            if k in by_q:
                v = sum(by_q[k]) / len(by_q[k])
                detail.append(f"{y}Q{qq}={v:.2f}(n={len(by_q[k])})")
    print(f"  detail: {'  '.join(detail)}")
    print()
