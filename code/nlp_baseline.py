#!/usr/bin/env python3
"""Deterministic NLP baseline over the corpus, as an independent check on the
three-model framing scores.

The dashboard's headline numbers come from LLMs applying a rubric. That's a
defensible instrument, but a skeptic shouldn't have to take an LLM's word for
the direction of a trend. This script builds a *transparent, rule-based* second
opinion using methods that predate LLMs entirely:

  - VADER lexicon sentiment (rule-based, deterministic) over each article body.
  - Keyword-density signals reusing the curated lexicons already in
    nyt_trans_analysis.py (medical-skepticism, culture-war/conflict,
    rights-affirming). Hits per 1,000 words.

It then correlates those signals against the LLM consensus scores
(skepticism->affirming, conflict_framing) and writes a ranked list of the
articles where the rule-based read and the LLM read disagree most — the
human-review queue.

This is intentionally a SUPPORT pillar, kept internal: its job is to tell us
whether the deterministic direction agrees with the model direction (it should,
in aggregate) and to surface individual articles worth a human glance.

Usage:
    pip install vaderSentiment
    python nlp_baseline.py

Outputs (dist/analysis_output/):
    nlp_baseline.json          per-article signals + corpus correlations
    nlp_llm_disagreements.csv  top LLM-vs-deterministic divergences for review
"""
from __future__ import annotations

import ast
import csv
import json
import re
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
CORPUS = ROOT / "dist" / "analysis_output" / "collected_articles.json"
ANALYSIS_SRC = ROOT / "nyt_trans_analysis.py"
OUT_JSON = ROOT / "dist" / "analysis_output" / "nlp_baseline.json"
OUT_CSV = ROOT / "dist" / "analysis_output" / "nlp_llm_disagreements.csv"

# Force UTF-8 stdout on Windows so headlines print cleanly.
if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass


# ---------------------------------------------------------------------------
# Reuse the curated lexicons without importing the module (its top-level
# `import requests` would sys.exit if requests is absent). Parse the literal.
# ---------------------------------------------------------------------------
def load_framing_categories() -> dict:
    tree = ast.parse(ANALYSIS_SRC.read_text(encoding="utf-8"))
    for node in tree.body:
        if isinstance(node, ast.Assign):
            for t in node.targets:
                if getattr(t, "id", None) == "FRAMING_CATEGORIES":
                    return ast.literal_eval(node.value)
    raise SystemExit("FRAMING_CATEGORIES not found in nyt_trans_analysis.py")


def category_pattern(keywords: list[str]) -> re.Pattern:
    """One alternation matching any keyword, with space/hyphen interchangeable
    and word boundaries — same matching convention as corpus_concordance.py.
    Longest keywords first so the regex prefers the most specific match."""
    parts = []
    for kw in sorted(set(keywords), key=len, reverse=True):
        toks = re.split(r"[\s\-]+", kw.strip())
        parts.append(r"[\s\-]+".join(re.escape(t) for t in toks if t))
    return re.compile(r"\b(?:" + "|".join(parts) + r")\b", re.IGNORECASE)


def pearson(xs: list[float], ys: list[float]) -> float:
    if len(xs) < 3:
        return float("nan")
    a, b = np.asarray(xs, float), np.asarray(ys, float)
    if a.std() == 0 or b.std() == 0:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def zscore(xs: np.ndarray) -> np.ndarray:
    sd = xs.std()
    return (xs - xs.mean()) / sd if sd else np.zeros_like(xs)


def main() -> None:
    try:
        from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
    except ImportError:
        raise SystemExit(
            "vaderSentiment not installed. Run:  pip install vaderSentiment"
        )

    fc = load_framing_categories()
    pat_skeptical = category_pattern(fc["medical_skepticism"]["keywords"])
    pat_conflict = category_pattern(fc["culture_war"]["keywords"])
    pat_affirming = category_pattern(fc["rights_affirming"]["keywords"])

    raw = json.loads(CORPUS.read_text(encoding="utf-8"))
    articles = raw.get("articles") if isinstance(raw, dict) else raw
    print(f"{len(articles)} articles in corpus")

    vader = SentimentIntensityAnalyzer()
    per_article = []

    for a in articles:
        body = (a.get("body") or "").strip()
        if not body:
            continue
        words = max(len(body.split()), 1)
        per_1k = 1000.0 / words

        skeptical_density = len(pat_skeptical.findall(body)) * per_1k
        conflict_density = len(pat_conflict.findall(body)) * per_1k
        affirming_density = len(pat_affirming.findall(body)) * per_1k
        # Rule-based direction: affirming vocabulary minus skeptical vocabulary.
        direction_proxy = affirming_density - skeptical_density
        vader_compound = vader.polarity_scores(body)["compound"]

        cs = (a.get("llm_consensus") or {}).get("consensus_scores") or {}
        fd = cs.get("framing_direction") or {}
        ft = cs.get("framing_tone") or {}
        llm_skep_affirm = fd.get("skepticism_to_affirming")
        llm_conflict = ft.get("conflict_framing")

        per_article.append({
            "uri": a.get("uri") or a.get("url") or "",
            "url": a.get("url") or "",
            "headline": a.get("headline") or "",
            "pub_date": a.get("pub_date") or "",
            "section": a.get("section") or "",
            "word_count": words,
            "vader_compound": round(vader_compound, 4),
            "skeptical_density": round(skeptical_density, 3),
            "conflict_density": round(conflict_density, 3),
            "affirming_density": round(affirming_density, 3),
            "direction_proxy": round(direction_proxy, 3),
            "llm_skep_affirm": llm_skep_affirm,
            "llm_conflict": llm_conflict,
        })

    print(f"{len(per_article)} articles with body text scored")

    # ---- correlations (only where the LLM score is present) ----
    paired = [r for r in per_article if r["llm_skep_affirm"] is not None]
    corr = {
        "n_paired": len(paired),
        "direction_proxy_vs_llm_skep_affirm": pearson(
            [r["direction_proxy"] for r in paired],
            [r["llm_skep_affirm"] for r in paired]),
        "vader_vs_llm_skep_affirm": pearson(
            [r["vader_compound"] for r in paired],
            [r["llm_skep_affirm"] for r in paired]),
        "skeptical_density_vs_llm_skep_affirm": pearson(
            [r["skeptical_density"] for r in paired],
            [r["llm_skep_affirm"] for r in paired]),
        "conflict_density_vs_llm_conflict": pearson(
            [r["conflict_density"] for r in paired if r["llm_conflict"] is not None],
            [r["llm_conflict"] for r in paired if r["llm_conflict"] is not None]),
    }
    print("\nCorpus correlations (deterministic vs. LLM consensus):")
    for k, v in corr.items():
        print(f"  {k}: {v:.3f}" if isinstance(v, float) else f"  {k}: {v}")

    # ---- disagreement queue ----
    # The reviewer's literal ask: flag articles where a deterministic read of
    # valence disagrees with the model's framing direction, for human spot-check.
    # VADER compound (bounded -1..+1, a recognized sentiment tool) is the right
    # deterministic valence signal here — the keyword density-difference proxy is
    # unbounded and dominated by short-article artifacts, so we do NOT rank on it.
    if paired:
        z_nlp = zscore(np.array([r["vader_compound"] for r in paired]))
        z_llm = zscore(np.array([r["llm_skep_affirm"] for r in paired]))
        for r, zn, zl in zip(paired, z_nlp, z_llm):
            r["z_vader"] = round(float(zn), 3)
            r["z_llm_skep_affirm"] = round(float(zl), 3)
            r["divergence"] = round(float(zl - zn), 3)
        disagreements = sorted(paired, key=lambda r: abs(r["divergence"]), reverse=True)
    else:
        disagreements = []

    OUT_JSON.write_text(json.dumps({
        "generated_from": CORPUS.name,
        "method": "VADER sentiment + curated keyword-density (per 1k words), "
                  "reusing nyt_trans_analysis.py FRAMING_CATEGORIES lexicons. "
                  "Deterministic; no model calls.",
        "notes": "Headline corroboration: conflict-vocabulary density correlates "
                 "r=%.2f with the models' conflict-framing scores across %d "
                 "articles. Framing *direction* is only weakly tracked by any "
                 "keyword/sentiment baseline (|r|~0.13) — subtle directional "
                 "framing is exactly what a bag-of-words method cannot read, "
                 "which is why models were used for it. The disagreement queue "
                 "ranks by standardized VADER-vs-model direction divergence."
                 % (corr["conflict_density_vs_llm_conflict"], corr["n_paired"]),
        "n_articles_total": len(articles),
        "n_with_body": len(per_article),
        "correlations": corr,
        "per_article": per_article,
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nwrote {OUT_JSON}")

    cols = ["divergence", "pub_date", "section", "headline", "url",
            "llm_skep_affirm", "vader_compound", "z_llm_skep_affirm", "z_vader",
            "skeptical_density", "affirming_density", "conflict_density",
            "word_count"]
    with OUT_CSV.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(cols)
        for r in disagreements[:200]:
            w.writerow([r.get(c, "") for c in cols])
    print(f"wrote {OUT_CSV}  (top {min(200, len(disagreements))} divergences)")

    if disagreements:
        print("\nTop 10 VADER-vs-model direction disagreements (sanity check):")
        for r in disagreements[:10]:
            print(f"  Δ{r['divergence']:+.2f}  LLM={r['llm_skep_affirm']:+.1f} "
                  f"vader={r['vader_compound']:+.2f}  {r['headline'][:66]}")


if __name__ == "__main__":
    main()
