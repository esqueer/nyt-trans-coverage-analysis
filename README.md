# How the New York Times covers transgender people: data & code

This repository contains the rubric, the code, and every per-article score behind the
analysis of the New York Times' transgender coverage. It is published so that anyone can
inspect the method, re-run it, and check the numbers without my cooperation.

## What the analysis does

Every Times article on transgender topics published since January 2014 was collected using
two of the paper's own public data feeds (the Article Search API, queried for 22 terms, and
the Archive API, filtered against the Times' own editor-applied subject tags such as
*"Transgender and Transsexuals"*). After de-duplication that is **3,242 articles**; full text
was recovered for 3,200 from public web archives.

Each article was then scored against a single human-written rubric applied **independently by
three large language models from three different companies**:

- Anthropic — Claude Haiku 4.5
- Google — Gemini 2.5 Flash-Lite
- A local **open-weights** model — Qwen 2.5 32B, run on a home GPU via Ollama (so the
  ensemble does not depend on any paid API to reproduce)

The three models' scores are averaged with **equal weight**. A fourth model (OpenAI GPT-5
nano) was evaluated but excluded from the ensemble because it declines to engage the
framing/tone scales; its raw scores are archived in `data/openai_scores_archive.json`.

As a sanity check, the whole corpus was also run through an old-fashioned NLP pass (VADER,
plain word counts + rule-based sentiment) in `code/nlp_baseline.py`.

## Repository layout

```
rubric/      scoring_prompt.md   — the exact system prompt + per-dimension schema
code/        the analysis pipeline (see "Code" below)
data/        per-article scores and all derived/aggregate data (see "Data" below)
dashboard/   index.html — self-contained interactive dashboard (open in a browser)
figures/     the charts used in the piece (PNG)
```

## The rubric

`rubric/scoring_prompt.md` is the verbatim system prompt every model received, plus the
scoring schema. The scales:

- **Topic** (0–10, multi-label): medical, legal, cultural, violence, personal, science,
  politics, international, youth_focus
- **Framing direction** (−5…+5, signed): `skepticism_to_affirming`,
  `restrictive_to_protective` — the article's *own* stance, explicitly separated from the
  outcome of the event it reports
- **Framing tone** (0–10): conflict_framing, pathologizing, both_sides_treatment,
  alarmist_tone, normalizing
- **Voice** (0–10): centrality of trans / expert / advocate / opponent voices

## Data

### `data/article_scores.json` — the per-article scores

One record per article. Each keeps the NYT URL, headline, byline, dates, section, material
type, NYT subject keywords, word count, and the scores:

- `llm_consensus.consensus_scores` — the 3-model average (the headline numbers)
- `llm_consensus.per_model_scores` — each model's raw scores, so you can see disagreement
- `llm_consensus.consensus_sd` / `consensus_certainty` — spread and confidence
- `llm_consensus.models_used`, `had_body`, `prompt_version`
- `llm_scores` — a legacy single-model score, kept for completeness

> **Article text is intentionally not included.** The full body text of NYT articles is
> copyrighted, so it has been stripped from every record (`body`, `body_meta`,
> `lead_paragraph` removed). Each record keeps its `url`, so any original article can be
> looked up directly. The scores are what this project makes public.

### Other data files

| File | What it is |
|------|------------|
| `analysis_results.json` | Summary tables (framing counts, breakdowns) |
| `nlp_baseline.json` | VADER baseline scores per article |
| `nlp_llm_disagreements.csv` | Where the word-counter and the models diverge |
| `openai_scores_archive.json` | Raw GPT-5 nano scores (excluded from the ensemble) |

## Code

| File | Role |
|------|------|
| `llm_consensus.py` | The ensemble: rubric/system prompt, scoring, equal-weight aggregation |
| `model_adapters.py` | Provider adapters (Anthropic, Gemini, Ollama, OpenAI) |
| `llm_classify.py` | Single-model classifier the ensemble builds on |
| `run_qwen_consensus.py` | Standalone runner that adds the local Qwen model |
| `fetch_bodies.py` | Recovers article full text from public web archives |
| `nlp_baseline.py` | VADER / rule-based sentiment baseline |
| `nyt_trans_analysis.py` | Corpus collection + the main analysis |
| `make_article_charts.py` | Generates the figures |
| `weighting_robustness.py`, `inflection_check.py`, `methodology_review.py` | Robustness checks |

### Reproducing

The scoring scripts read API keys from local files (e.g. `ANTHROPIC_KEY.txt`,
`GEMINI_KEY.txt`) and a local Ollama server for the open-weights model — **none of those
secrets are in this repository**, and `.gitignore` is set up to keep it that way. The
open-weights leg (Qwen via Ollama) can be re-run with no paid API at all. See comments at the
top of each script and the per-dimension definitions in `code/llm_consensus.py`.

## License & use

Code is provided for inspection and reproduction. The scores are derived data about the
coverage; original article text belongs to The New York Times and is not redistributed here.
