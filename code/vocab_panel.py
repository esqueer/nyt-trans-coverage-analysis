#!/usr/bin/env python3
"""Terminology register panel over the corpus body text.

For a fixed panel of framing-coded terms, computes per-year occurrences
split into the paper's narrative voice vs. quoted sources, normalizes by
per-year corpus volume, aggregates by register, and charts:

  vocab_register_rates.png  affirming vs skeptic vs clinical register,
                            paper-voice rate per 100 corpus articles
  vocab_skeptic_share.png   skeptic share of (affirming+skeptic) paper-voice
                            mentions per year — does the clinical-skeptic
                            register gain ground in the paper's own voice?
  vocab_terms_small.png     small multiples, one term per panel (own y-scale)

Paper-voice-only is the headline cut: it strips what sources say and isolates
the Times' own editorial register. Counts are reported alongside rates so the
reader sees the n behind each point.
"""
import json
import re
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).parent
CORPUS = ROOT / "dist" / "analysis_output" / "collected_articles.json"
OUT_DIR = ROOT / "article"
OUT_DIR.mkdir(exist_ok=True)

# term -> register.  A trailing "*" means stem match (detransition -> -ed/-er).
PANEL = [
    ("gender-affirming", "affirming"),
    ("transphobic",      "affirming"),
    ("transphobia",      "affirming"),
    ("anti-trans",       "affirming"),
    ("cross-sex hormones", "skeptic"),
    ("biological sex",   "skeptic"),
    ("sex change",       "skeptic"),
    ("gender ideology",  "skeptic"),
    ("detransition*",    "skeptic"),
    ("puberty blockers", "clinical"),
    ("gender dysphoria", "clinical"),
    ("hormone therapy",  "clinical"),
]

REGISTER_COLORS = {"affirming": "#2e6f7e", "skeptic": "#b03a48", "clinical": "#b8860b"}
YEARS = list(range(2014, 2027))


def term_pattern(term: str) -> re.Pattern:
    stem = term.endswith("*")
    term = term.rstrip("*")
    parts = re.split(r"[\s\-]+", term.strip())
    body = r"[\s\-]+".join(re.escape(p) for p in parts)
    tail = r"\w*" if stem else r"\b"
    lead = r"\b"
    return re.compile(rf"{lead}{body}{tail}", re.IGNORECASE)


def in_quote(text: str, pos: int) -> bool:
    open_i = text.rfind("“", 0, pos)
    close_i = text.rfind("”", 0, pos)
    if open_i == -1 and close_i == -1:
        return text.count('"', 0, pos) % 2 == 1
    return open_i > close_i


raw = json.loads(CORPUS.read_text(encoding="utf-8"))
articles = raw.get("articles") if isinstance(raw, dict) else raw

# corpus volume per year (denominator)
corpus_year = defaultdict(int)
for a in articles:
    y = (a.get("pub_date") or "")[:4]
    if y.isdigit():
        corpus_year[int(y)] += 1

# voice[term][year], quote[term][year]
voice = {t: defaultdict(int) for t, _ in PANEL}
quote = {t: defaultdict(int) for t, _ in PANEL}
pats = {t: term_pattern(t) for t, _ in PANEL}

for a in articles:
    body = a.get("body") or ""
    if not body:
        continue
    y = (a.get("pub_date") or "")[:4]
    if not y.isdigit():
        continue
    y = int(y)
    for t, _ in PANEL:
        for m in pats[t].finditer(body):
            if in_quote(body, m.start()):
                quote[t][y] += 1
            else:
                voice[t][y] += 1


def rate(term, yr):  # paper-voice occ per 100 corpus articles that year
    denom = corpus_year.get(yr, 0)
    return (voice[term][yr] / denom * 100) if denom else 0.0


# ---- console table -------------------------------------------------------
print("PAPER-VOICE occurrences per year (quoted uses excluded)\n")
hdr = "term".ljust(20) + " reg   " + " ".join(f"{y%100:>3d}" for y in YEARS) + "  voice/quote"
print(hdr)
for t, reg in PANEL:
    cells = " ".join(f"{voice[t][y]:>3d}" for y in YEARS)
    vtot = sum(voice[t].values())
    qtot = sum(quote[t].values())
    print(f"{t:<20} {reg[:5]:<5} {cells}   {vtot}/{qtot}")

# ---- register aggregates -------------------------------------------------
reg_voice = defaultdict(lambda: defaultdict(int))
for t, reg in PANEL:
    for y in YEARS:
        reg_voice[reg][y] += voice[t][y]


def reg_rate(reg, yr):
    denom = corpus_year.get(yr, 0)
    return (reg_voice[reg][yr] / denom * 100) if denom else 0.0


print("\nREGISTER paper-voice rate per 100 corpus articles\n")
print("year  " + "  ".join(f"{r:>9s}" for r in ("affirming", "skeptic", "clinical")) + "   skeptic_share%")
for y in YEARS:
    aff, ske, cli = reg_rate("affirming", y), reg_rate("skeptic", y), reg_rate("clinical", y)
    base = reg_voice["affirming"][y] + reg_voice["skeptic"][y]
    share = (reg_voice["skeptic"][y] / base * 100) if base else 0
    print(f"{y}  {aff:9.2f}  {ske:9.2f}  {cli:9.2f}   {share:5.1f}%")

# ---- styling -------------------------------------------------------------
plt.rcParams.update({
    "figure.facecolor": "white", "axes.facecolor": "white",
    "axes.edgecolor": "#cccccc", "axes.grid": True, "grid.color": "#ececec",
    "axes.spines.top": False, "axes.spines.right": False,
    "font.size": 10, "axes.titlesize": 12, "axes.titleweight": "bold",
})
KAHN = 2022.45


def mark_kahn(ax):
    ax.axvline(KAHN, color="#b03a48", lw=0.9, ls=":", alpha=0.7)


# ---- Chart 1: register rates ---------------------------------------------
fig, ax = plt.subplots(figsize=(10, 5.5))
for reg in ("affirming", "skeptic", "clinical"):
    ys = [reg_rate(reg, y) for y in YEARS]
    ax.plot(YEARS, ys, color=REGISTER_COLORS[reg], lw=2.4, marker="o", ms=3,
            label=f"{reg} register")
mark_kahn(ax)
ax.annotate("Kahn / editorial shift\n(mid-2022)", xy=(KAHN, ax.get_ylim()[1]*0.92),
            fontsize=8, color="#b03a48", ha="left")
ax.set_title("The Times' own vocabulary: register rate over time")
ax.set_ylabel("paper-voice mentions per 100 trans-coverage articles")
ax.set_xlim(2014, 2026)
ax.legend(fontsize=9)
fig.text(0.01, -0.01, "Quoted uses excluded. Affirming = gender-affirming, transphobic/-ia, anti-trans. "
         "Skeptic = cross-sex hormones, biological sex, sex change, gender ideology, detransition. "
         "Clinical = puberty blockers, gender dysphoria, hormone therapy. 2026 partial.",
         fontsize=7, color="#777")
fig.tight_layout()
fig.savefig(OUT_DIR / "vocab_register_rates.png", dpi=200, bbox_inches="tight")
plt.close(fig)
print(f"\nwrote {OUT_DIR / 'vocab_register_rates.png'}")

# ---- Chart 2: skeptic share ----------------------------------------------
fig, ax = plt.subplots(figsize=(10, 5))
shares, ns = [], []
for y in YEARS:
    base = reg_voice["affirming"][y] + reg_voice["skeptic"][y]
    shares.append((reg_voice["skeptic"][y] / base * 100) if base else 0)
    ns.append(base)
ax.plot(YEARS, shares, color="#b03a48", lw=2.6, marker="o", ms=4)
for y, s, n in zip(YEARS, shares, ns):
    ax.annotate(f"n={n}", xy=(y, s), xytext=(0, 7), textcoords="offset points",
                fontsize=6.5, color="#999", ha="center")
mark_kahn(ax)
ax.set_title("Clinical-skeptic share of the Times' affirming+skeptic vocabulary")
ax.set_ylabel("skeptic-register % of paper-voice mentions")
ax.set_xlim(2014, 2026)
ax.set_ylim(0, max(shares) * 1.25 + 5)
fig.text(0.01, -0.01, "Skeptic mentions / (affirming + skeptic) mentions, paper voice only. "
         "n = combined mention base per year (small early-year n is noisy). 2026 partial.",
         fontsize=7, color="#777")
fig.tight_layout()
fig.savefig(OUT_DIR / "vocab_skeptic_share.png", dpi=200, bbox_inches="tight")
plt.close(fig)
print(f"wrote {OUT_DIR / 'vocab_skeptic_share.png'}")

# ---- Chart 3: small multiples per term -----------------------------------
ncol = 3
nrow = (len(PANEL) + ncol - 1) // ncol
fig, axes = plt.subplots(nrow, ncol, figsize=(12, 2.3 * nrow), sharex=True)
for i, (t, reg) in enumerate(PANEL):
    ax = axes[i // ncol][i % ncol]
    ys = [rate(t, y) for y in YEARS]
    ax.fill_between(YEARS, ys, color=REGISTER_COLORS[reg], alpha=0.25)
    ax.plot(YEARS, ys, color=REGISTER_COLORS[reg], lw=1.8)
    ax.axvline(KAHN, color="#b03a48", lw=0.7, ls=":", alpha=0.6)
    ax.set_title(f"{t}  ({reg})", fontsize=9.5)
    ax.set_xlim(2014, 2026)
    ax.tick_params(labelsize=7)
for j in range(len(PANEL), nrow * ncol):
    axes[j // ncol][j % ncol].axis("off")
fig.suptitle("Selected terms — paper-voice rate per 100 articles (each panel its own scale)",
             fontsize=13, fontweight="bold")
fig.text(0.01, 0.005, "Dotted line = Kahn / mid-2022 editorial shift. Each panel y-scaled independently "
         "so trajectory shape is visible despite very different magnitudes.", fontsize=7, color="#777")
fig.tight_layout(rect=[0, 0.02, 1, 0.98])
fig.savefig(OUT_DIR / "vocab_terms_small.png", dpi=200, bbox_inches="tight")
plt.close(fig)
print(f"wrote {OUT_DIR / 'vocab_terms_small.png'}")

# ---- save data -----------------------------------------------------------
data = {
    "corpus_year": dict(corpus_year),
    "terms": {t: {"register": reg,
                  "voice": {str(y): voice[t][y] for y in YEARS},
                  "quote": {str(y): quote[t][y] for y in YEARS}}
              for t, reg in PANEL},
}
(OUT_DIR.parent / "dist" / "analysis_output" / "vocab_panel.json").write_text(
    json.dumps(data, indent=2), encoding="utf-8")
print(f"wrote vocab_panel.json")
