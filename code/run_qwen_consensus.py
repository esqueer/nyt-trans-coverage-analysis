#!/usr/bin/env python3
"""Add the Qwen (Ollama) 4th-model scores to the existing 3-cloud-model
consensus over the full corpus.

The 3 cloud providers (Claude Haiku, GPT-5 nano, Gemini Flash-Lite) are already
scored on all articles; their per-model scores are REUSED (no fresh API calls).
Only Qwen runs locally on the GPU, and the consensus is re-aggregated across
all 4 models.

Resumable: skip_existing=True skips any article that already has all 4 models,
so re-running after an interruption picks up where it left off.

Writes back to collected_articles.json with checkpointing every 25 articles,
then regenerates dashboard_data.json.
"""
import json
import os
import sys
import time
from pathlib import Path

HERE = r"C:\Users\aleja\OneDrive\Documents\Claude Projects\NYT Analyzer - Consensus"
sys.path.insert(0, HERE)

import model_adapters as ma
import llm_consensus as consensus
import fetch_bodies as bodies
import nyt_trans_analysis as analyzer

DIST = os.path.join(HERE, "dist")
OUTPUT_DIR = os.path.join(DIST, "analysis_output")
# Path (not str): bodies.update_articles_file calls path.with_suffix(...)
COLLECTED = Path(OUTPUT_DIR) / "collected_articles.json"
OLLAMA_URL = "http://localhost:11434"
OLLAMA_MODEL = "qwen2.5:32b-instruct-q4_K_M"
CHECKPOINT_EVERY = 25

anthropic_key = open(os.path.join(DIST, "ANTHROPIC_KEY.txt"), encoding="utf-8").read().strip()
openai_key = open(os.path.join(DIST, "OPENAI_KEY.txt"), encoding="utf-8").read().strip()
gemini_key = open(os.path.join(DIST, "GEMINI_KEY.txt"), encoding="utf-8").read().strip()

# Probe Ollama before starting
ok, info = ma.probe_ollama(OLLAMA_URL, timeout=5.0)
if not ok:
    print(f"ABORT: Ollama not reachable: {info}", flush=True)
    sys.exit(1)
print(f"Ollama OK: {info}", flush=True)

with open(COLLECTED, encoding="utf-8") as f:
    data = json.load(f)
articles = data["articles"] if isinstance(data, dict) else data
total = len(articles)

# How many already have qwen (resume awareness)
already = sum(
    1 for a in articles
    if "ollama" in ((a.get("llm_consensus") or {}).get("models_used") or [])
)
print(f"Total articles: {total} | already have Qwen: {already} | to process: {total - already}", flush=True)

models = dict(consensus.DEFAULT_MODELS)
models["ollama"] = OLLAMA_MODEL

start_ts = time.time()
processed_since_start = [0]  # mutable for closure

def progress(i, tot, article, cons):
    pm = cons.get("per_model_scores") or {}
    qerr = (pm.get("ollama") or {}).get("error")
    used = cons.get("models_used") or []
    if qerr:
        tag = f"QWEN-ERR: {qerr[:50]}"
    else:
        processed_since_start[0] += 1
        tag = f"4-model OK ({len(used)} models)"
    # ETA based on articles actually run (not skipped) this session
    done_run = processed_since_start[0]
    if done_run > 0:
        rate = (time.time() - start_ts) / done_run
        remaining = (tot - i) * rate  # rough; assumes rest need running
        eta_min = remaining / 60
    else:
        eta_min = 0
    print(f"[{i}/{tot}] {tag} | {(article.get('headline') or '')[:45]!r} "
          f"| ~{eta_min:.0f}m left", flush=True)

    if i % CHECKPOINT_EVERY == 0:
        try:
            bodies.update_articles_file(COLLECTED, articles)
            print(f"  -- checkpoint saved at {i} --", flush=True)
        except Exception as e:
            print(f"  -- checkpoint FAILED: {e} --", flush=True)

print("Starting Qwen pass (cloud models reused)...", flush=True)
stats = consensus.classify_all(
    articles,
    anthropic_key=anthropic_key,
    openai_key=openai_key,
    gemini_key=gemini_key,
    models=models,
    ollama_url=OLLAMA_URL,
    progress_cb=progress,
    skip_existing=True,  # resumable: skips articles that already have all 4
)

# Final save
bodies.update_articles_file(COLLECTED, articles)
print(f"\nFinal stats: {json.dumps(stats, indent=2)}", flush=True)

# Regenerate dashboard_data.json
print("Regenerating dashboard_data.json...", flush=True)
import io
from contextlib import redirect_stdout, redirect_stderr
sink = io.StringIO()
try:
    with redirect_stdout(sink), redirect_stderr(sink):
        results = analyzer.analyze_articles(articles, OUTPUT_DIR)
        analyzer.export_dashboard_data(results, articles, OUTPUT_DIR)
    print("Dashboard data regenerated.", flush=True)
except Exception as e:
    print(f"Dashboard regen warning: {e}", flush=True)

# Verify final coverage
qwen_final = sum(
    1 for a in articles
    if "ollama" in ((a.get("llm_consensus") or {}).get("models_used") or [])
)
print(f"\nDONE. Articles with Qwen score: {qwen_final}/{total}", flush=True)
