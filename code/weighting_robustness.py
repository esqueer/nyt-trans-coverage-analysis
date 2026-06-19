#!/usr/bin/env python3
"""Robustness check cited in the methods box: do corpus-level era trends
change if articles are weighted by cross-model agreement (consensus_certainty)
instead of counted equally?

Answer (June 2026, 3-model corpus): no — every E3-E2 trend is identical to
within +/-0.04. Also reports the correlation between score magnitude and
agreement (positive: models agree MORE on strongly framed articles, so
disagreement concentrates in the ambiguous middle).
"""
import json
import statistics
from pathlib import Path

DATA = Path(__file__).parent / "dist" / "analysis_output" / "collected_articles.json"

with open(DATA, encoding="utf-8") as f:
    raw = json.load(f)
articles = raw.get("articles") if isinstance(raw, dict) else raw


def era(a):
    try:
        y = int((a.get("pub_date") or "")[:4])
    except ValueError:
        return None
    if 2014 <= y <= 2017:
        return "E1"
    if 2018 <= y <= 2021:
        return "E2"
    if y >= 2022:
        return "E3"
    return None


def cons(a, sec, dim):
    cs = ((a.get("llm_consensus") or {}).get("consensus_scores") or {})
    v = (cs.get(sec) or {}).get(dim)
    return float(v) if isinstance(v, (int, float)) else None


def cert(a):
    return ((a.get("llm_consensus") or {}).get("consensus_certainty")) or 0.0


DIMS = [
    ("framing_direction", "skepticism_to_affirming"),
    ("framing_direction", "restrictive_to_protective"),
    ("framing_tone", "conflict_framing"),
    ("framing_tone", "normalizing"),
    ("voice", "trans_voice_centrality"),
    ("voice", "opponent_voice_centrality"),
]

print("%-44s %-4s %7s %7s %6s" % ("dimension", "era", "plain", "wghtd", "diff"))
for sec, dim in DIMS:
    deltas = {}
    for e in ["E1", "E2", "E3"]:
        sub = [(cons(a, sec, dim), cert(a)) for a in articles if era(a) == e]
        sub = [(v, w) for v, w in sub if v is not None]
        plain = sum(v for v, _ in sub) / len(sub)
        wghtd = sum(v * w for v, w in sub) / sum(w for _, w in sub)
        deltas[e] = (plain, wghtd)
        print("%-44s %-4s %7.2f %7.2f %+6.2f"
              % (f"{sec}.{dim}", e, plain, wghtd, wghtd - plain))
    p_trend = deltas["E3"][0] - deltas["E2"][0]
    w_trend = deltas["E3"][1] - deltas["E2"][1]
    print("%-44s %-4s %+7.2f %+7.2f   <- E3-E2 trend under both schemes"
          % ("", "d32", p_trend, w_trend))
    print()

# Composition check: does agreement correlate with how strongly framed
# an article is? Positive r = strongly framed articles are the EASY calls.
mags, certs = [], []
for a in articles:
    v = cons(a, "framing_direction", "restrictive_to_protective")
    if v is None:
        continue
    mags.append(abs(v))
    certs.append(cert(a))
n = len(mags)
mm, mc = sum(mags) / n, sum(certs) / n
cov = sum((m - mm) * (c - mc) for m, c in zip(mags, certs)) / n
r = cov / (statistics.pstdev(mags) * statistics.pstdev(certs))
print("corr(|rights-stance score|, certainty) = %+.3f  (n=%d)" % (r, n))
