#!/usr/bin/env python3
"""One-off diagnostic for the methodology review.

Computes, from per_model_scores on collected_articles.json:
  1. Coverage per provider
  2. Pairwise Pearson correlation per dimension (do models rank articles the same way?)
  3. Krippendorff's alpha (interval) per dimension (the content-analysis standard)
  4. Mean absolute pairwise difference per dimension (raw disagreement size)
  5. Era-bucket means per provider for key dims (robustness: same trend regardless of model?)
"""
import json
import math
from collections import defaultdict
from pathlib import Path

DATA = Path(__file__).parent / "dist" / "analysis_output" / "collected_articles.json"

with open(DATA, encoding="utf-8") as f:
    raw = json.load(f)
articles = raw.get("articles") if isinstance(raw, dict) else raw

# OpenAI removed from the ensemble June 2026 (scale non-engagement);
# its raw scores live in dist/analysis_output/openai_scores_archive.json.
PROVIDERS = ["anthropic", "gemini", "ollama"]

SECTIONS = {
    "topic": ["medical", "legal", "cultural", "violence", "personal",
              "science", "politics", "international", "youth_focus"],
    "framing_direction": ["skepticism_to_affirming", "restrictive_to_protective"],
    "framing_tone": ["conflict_framing", "pathologizing", "both_sides_treatment",
                     "alarmist_tone", "normalizing"],
    "voice": ["trans_voice_centrality", "expert_voice_centrality",
              "advocate_voice_centrality", "opponent_voice_centrality"],
}


def get_score(article, provider, section, dim):
    cons = article.get("llm_consensus") or {}
    pm = (cons.get("per_model_scores") or {}).get(provider) or {}
    if pm.get("error"):
        return None
    sec = pm.get(section) or {}
    v = sec.get(dim)
    return float(v) if isinstance(v, (int, float)) else None


def era(article):
    d = (article.get("pub_date") or "")[:4]
    try:
        y = int(d)
    except ValueError:
        return None
    if 2014 <= y <= 2017:
        return "2014-2017"
    if 2018 <= y <= 2021:
        return "2018-2021"
    if y >= 2022:
        return "2022+"
    return None


def pearson(xs, ys):
    n = len(xs)
    if n < 3:
        return None
    mx, my = sum(xs) / n, sum(ys) / n
    sx = math.sqrt(sum((x - mx) ** 2 for x in xs))
    sy = math.sqrt(sum((y - my) ** 2 for y in ys))
    if sx == 0 or sy == 0:
        return None
    return sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / (sx * sy)


def kripp_alpha_interval(rows):
    """Krippendorff's alpha for interval data.
    rows: list of dicts {rater: value} (missing raters allowed, need >=2/row)."""
    units = [r for r in rows if len(r) >= 2]
    if not units:
        return None
    # Observed disagreement
    do_num, do_den = 0.0, 0
    all_vals = []
    for u in units:
        vals = list(u.values())
        m = len(vals)
        all_vals.extend(vals)
        for i in range(m):
            for j in range(m):
                if i != j:
                    do_num += (vals[i] - vals[j]) ** 2
        do_den += m * (m - 1)
    if do_den == 0:
        return None
    do = do_num / do_den
    # Expected disagreement across all values
    n = len(all_vals)
    if n < 2:
        return None
    de_num = 0.0
    mean_all = sum(all_vals) / n
    var_all = sum((v - mean_all) ** 2 for v in all_vals) / n
    de = 2 * var_all * n / (n - 1)
    if de == 0:
        return 1.0
    return 1 - do / de


# --- 1. Coverage ---
total = len(articles)
cov = defaultdict(int)
full3 = 0
for a in articles:
    ok = [p for p in PROVIDERS
          if get_score(a, p, "framing_direction", "skepticism_to_affirming") is not None]
    for p in ok:
        cov[p] += 1
    if len(ok) == 3:
        full3 += 1

print(f"TOTAL ARTICLES: {total}")
print(f"Full 3-provider coverage: {full3}")
for p in PROVIDERS:
    print(f"  {p:10s}: {cov[p]}")

# --- 2-4. Agreement stats per dimension (3 raters: A, G, Qwen) ---
print("\n=== AGREEMENT BY DIMENSION (n = full-coverage articles) ===")
print(f"{'dimension':32s} {'alpha':>6s} {'r(A-G)':>7s} {'r(A-Q)':>7s} {'r(G-Q)':>7s} {'scale':>7s}")

results = {}
for section, dims in SECTIONS.items():
    for dim in dims:
        rows = []
        per = {p: [] for p in PROVIDERS}
        for a in articles:
            vals = {}
            for p in PROVIDERS:
                v = get_score(a, p, section, dim)
                if v is not None:
                    vals[p] = v
            if len(vals) == len(PROVIDERS):
                rows.append(vals)
                for p in PROVIDERS:
                    per[p].append(vals[p])
        if not rows:
            continue
        alpha = kripp_alpha_interval(rows)
        r_ag = pearson(per["anthropic"], per["gemini"])
        r_aq = pearson(per["anthropic"], per["ollama"])
        r_gq = pearson(per["gemini"], per["ollama"])
        scale = "0-10" if section != "framing_direction" else "-5..5"
        fmt = lambda v: f"{v:.2f}" if v is not None else "  n/a"
        print(f"{section + '.' + dim:32s} {fmt(alpha):>6s} {fmt(r_ag):>7s} {fmt(r_aq):>7s} {fmt(r_gq):>7s} {scale:>7s}")
        results[f"{section}.{dim}"] = alpha

# --- 5. Era trends per provider (robustness check) ---
print("\n=== ERA TRENDS PER PROVIDER (mean score; does the TREND survive model choice?) ===")
KEY_DIMS = [
    ("framing_direction", "skepticism_to_affirming"),
    ("framing_direction", "restrictive_to_protective"),
    ("framing_tone", "conflict_framing"),
    ("framing_tone", "both_sides_treatment"),
    ("framing_tone", "normalizing"),
    ("framing_tone", "pathologizing"),
    ("voice", "trans_voice_centrality"),
    ("voice", "opponent_voice_centrality"),
    ("topic", "medical"),
    ("topic", "youth_focus"),
]
ERAS = ["2014-2017", "2018-2021", "2022+"]
for section, dim in KEY_DIMS:
    print(f"\n{section}.{dim}")
    print(f"  {'provider':10s} {'2014-2017':>10s} {'2018-2021':>10s} {'2022+':>8s}  trend")
    for p in PROVIDERS:
        means = []
        for e in ERAS:
            vals = [get_score(a, p, section, dim) for a in articles
                    if era(a) == e]
            vals = [v for v in vals if v is not None]
            means.append(sum(vals) / len(vals) if vals else None)
        cells = [f"{m:10.2f}" if m is not None else f"{'n/a':>10s}" for m in means]
        if all(m is not None for m in means):
            d1 = means[1] - means[0]
            d2 = means[2] - means[1]
            arrow = lambda d: "+" if d > 0.3 else ("-" if d < -0.3 else "=")
            trend = f"{arrow(d1)}{arrow(d2)}"
        else:
            trend = "?"
        print(f"  {p:10s} {cells[0]} {cells[1]} {cells[2]}  {trend}")

# Era article counts
print("\nEra article counts (any provider scored):")
for e in ERAS:
    n = sum(1 for a in articles if era(a) == e)
    print(f"  {e}: {n}")
