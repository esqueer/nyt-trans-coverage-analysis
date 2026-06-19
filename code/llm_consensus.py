#!/usr/bin/env python3
"""Multi-model consensus classifier for NYT trans-coverage articles.

For each article, calls 3 cloud providers (Claude Haiku, GPT-5 nano, Gemini
2.5 Flash-Lite) in parallel — and optionally a 4th local model via Ollama —
then aggregates the per-model scores using a SIMPLE UNWEIGHTED MEAN.

Why simple mean and not inverse-variance weighting (which earlier versions
used): when 2 of 3 models agree, the agreement often reflects correlated
training-data biases (e.g. topic-valence leaking into framing scores) rather
than independent confirmation. Downweighting the dissenter then pulls the
aggregate toward the wrong answer when the minority opinion is actually
more faithful to the prompt. Equal weighting avoids this failure mode and
makes the aggregation honest about the inherent subjectivity of the task.

The aggregated record stored on each article includes:
    - consensus_scores: unweighted-mean values per dimension
    - consensus_sd: per-dimension standard deviation across models — the
                    disagreement signal, surfaced for the dashboard's
                    sort-by-disagreement view but NOT used to weight scores
    - consensus_certainty: legacy 0-1 metric (1 / (1 + var)) kept for
                           backwards compatibility with existing UI code
    - per_model_scores: raw output from each provider (kept for inspection)
    - models_used: which providers actually returned a valid score
    - prompt_version, had_body: same semantics as single-model classifier
"""
from __future__ import annotations

import statistics
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import Callable, Optional

import model_adapters as ma


# ---------------------------------------------------------------------------
# Schema (mirrors the original llm_classify.py — kept compatible so downstream
# dashboard code can read consensus_scores the same way it reads llm_scores)
# ---------------------------------------------------------------------------

PROMPT_VERSION = 3  # carries over from the single-model module

# Default cloud models per provider — adjustable from the launcher.
# Ollama is added as a provider at run-time when the local server is
# reachable (see launcher._run_consensus_classify). Not listed here because
# the model name depends on what's pulled to the user's Ollama instance.
#
# OpenAI was removed from the ensemble (June 2026): GPT-5 nano declines to
# engage the framing/tone scales (scores ~0 regardless of content, pairwise
# r ≈ 0 with every other model on those dims), so it added noise rather
# than signal. Its historical raw scores are archived in
# analysis_output/openai_scores_archive.json. The adapter code remains in
# model_adapters.py if it's ever re-enabled.
DEFAULT_MODELS = {
    "anthropic": "claude-haiku-4-5",
    "gemini":    "gemini-2.5-flash-lite",
}

# Providers whose scores participate in consensus aggregation/reaggregation.
ACTIVE_PROVIDERS = ("anthropic", "gemini", "ollama")

# Suggested default for the 4090 — Qwen 2.5 32B at Q4 fits comfortably in
# 24GB VRAM and is the strongest open-weights model in that class for
# structured output. User overrides via OLLAMA.json in their workspace.
DEFAULT_OLLAMA_MODEL = "qwen2.5:32b-instruct-q4_K_M"

MAX_OUTPUT_TOKENS = 1200
# Generous ceiling so a local 32B model (Ollama on a single GPU) has room to
# process a full-article prompt and emit up to MAX_OUTPUT_TOKENS of JSON. Cloud
# providers normally respond in a few seconds, so a higher ceiling is harmless
# for them and only matters as headroom for the slower local model.
REQUEST_TIMEOUT = 180
MAX_BODY_CHARS = 12000


TOPIC_DIMS = [
    ("medical",        0, 10, "Medical care, hormone therapy, puberty blockers, surgery, gender clinics"),
    ("legal",          0, 10, "Bans, lawsuits, court rulings, executive orders, legislation"),
    ("cultural",       0, 10, "Arts, books, film, theater, profiles, cultural reception"),
    ("violence",       0, 10, "Hate crimes, attacks, threats, harassment, safety concerns"),
    ("personal",       0, 10, "Individual narratives, family stories, journeys"),
    ("science",        0, 10, "Studies, surveys, research data, scientific publications"),
    ("politics",       0, 10, "Electoral politics, campaigns, partisan conflict"),
    ("international",  0, 10, "Trans coverage outside the United States"),
    ("youth_focus",    0, 10, "Specifically focused on minors / children / teens"),
]

FRAMING_DIRECTION_DIMS = [
    ("skepticism_to_affirming",   -5, 5, "Position on gender-affirming care: -5 strongly skeptical/critical, 0 neutral, +5 strongly affirming"),
    ("restrictive_to_protective", -5, 5, "Position on trans rights: -5 favors restriction, 0 neutral, +5 favors protection"),
]

FRAMING_TONE_DIMS = [
    ("conflict_framing",      0, 10, "Frames the issue as battle/controversy/clash between sides"),
    ("pathologizing",         0, 10, "Treats trans identity as problem, disorder, or aberration"),
    ("both_sides_treatment",  0, 10, "Artificial balance / false equivalence between trans rights and skepticism"),
    ("alarmist_tone",         0, 10, "Catastrophizing, urgent, crisis-coded language"),
    ("normalizing",           0, 10, "Treats trans existence as ordinary/everyday"),
]

VOICE_DIMS = [
    ("trans_voice_centrality",     0, 10, "How substantively trans people's own voices SHAPE the article — not just whether they appear. 0 = absent or only spoken about. 2-3 = a single short quote used as reaction. 5 = provides material content but article's frame is set elsewhere. 7-8 = trans voices structure the narrative arc. 10 = the article is built around trans subjects' own framing and experience"),
    ("expert_voice_centrality",    0, 10, "How substantively medical/scientific expert voices shape the article. Same scale as above — count narrative weight, not just presence of quotes"),
    ("advocate_voice_centrality",  0, 10, "How substantively trans rights advocates / civil-rights orgs shape the article. Same scale — narrative weight, not just presence"),
    ("opponent_voice_centrality",  0, 10, "How substantively critics/opponents of gender-affirming care or trans rights shape the article. Same scale — narrative weight, not just presence"),
]


def _format_schema_for_prompt() -> str:
    lines = ["TOPIC (0-10, multi-label, several can be high simultaneously):"]
    for key, _lo, _hi, defn in TOPIC_DIMS:
        lines.append(f"  - {key}: {defn}")
    lines.append("")
    lines.append("FRAMING_DIRECTION (-5 to +5, signed; 0 = neutral or balanced):")
    for key, _lo, _hi, defn in FRAMING_DIRECTION_DIMS:
        lines.append(f"  - {key}: {defn}")
    lines.append("")
    lines.append("FRAMING_TONE (0-10):")
    for key, _lo, _hi, defn in FRAMING_TONE_DIMS:
        lines.append(f"  - {key}: {defn}")
    lines.append("")
    lines.append("VOICE (0-10):")
    for key, _lo, _hi, defn in VOICE_DIMS:
        lines.append(f"  - {key}: {defn}")
    return "\n".join(lines)


SYSTEM_PROMPT = f"""You are an objective media-analysis assistant scoring NYT articles on transgender coverage across multiple dimensions. Your goal is consistent, evidence-based scoring — not advocacy.

For each article, return a JSON object with these dimensions:

{_format_schema_for_prompt()}

Plus:
  - confidence: 0-10 (how sure you are about your scores given available text)
  - rationale: ONE OR TWO SENTENCES maximum (do not exceed 400 characters)
    naming the strongest signal that drove the directional scores. Be terse —
    rationales over 400 chars cause output truncation and corrupt the JSON.

CRITICAL RULE — separate WHAT HAPPENED from HOW IT'S COVERED:

The framing_direction scores measure THE ARTICLE'S OWN FRAMING — both tonal and
structural — NOT the outcome of the event being reported.

The OUTCOME of the news is not the article's stance:
- A news report about a court ruling that restricts trans rights does not
  automatically score restrictive. The ruling was restrictive; the article may
  or may not take a stance about it.
- Same for protective rulings, restrictive bills, hate crimes, conservative
  campaigns, or affirming studies. The event is what it is — score only the
  article's handling.

Score outside ±1 when the article takes a stance through EITHER tonal OR
structural choices.

TONAL framing (the obvious kind):
  · Loaded language ("alarmist", "discredited", "concerning", "common-sense",
    "experimental", "lifesaving") applied to one side but not the other.
  · Adjectives and verbs that color the description of one side.
  · Editorial verdicts: explicit endorsement, criticism, or pathologizing.

STRUCTURAL framing (the harder-to-spot kind that ALSO counts):
  · Which side's terminology becomes the article's unmarked default — e.g.,
    "gender-affirming care" vs "transition treatment" vs "experimental
    interventions for minors". The choice between these IS a stance.
  · Source ordering and emphasis: who gets quoted first, who gets the kicker
    quote, whose framing organizes the lead, whose paragraph-length quotes
    drive the narrative vs whose single-sentence rebuttal lands near the end.
  · Source-type mix: is the article structured around government officials,
    medical institutions, advocacy groups, opponents, or affected subjects —
    and does that mix line up with the story or skew it?
  · What the article treats as ESTABLISHED background vs CONTESTED foreground.
    Presenting "gender-affirming care reduces suicidality" as settled fact
    (vs. as an advocacy claim) is a structural choice; so is presenting "the
    Cass Review found weak evidence" as settled fact (vs. as a contested
    finding).
  · Lead framing: does the opening paragraph adopt one side's perspective as
    the lens for understanding the event?
  · Omissions: which voices, facts, or counter-perspectives are conspicuously
    absent given the topic?

A tonally measured article that consistently makes these structural choices in
one direction IS taking a stance — score it ±2 to ±4 even if the language stays
calm.

DO NOT default to "neutrality." Some pieces are genuinely balanced and score 0.
Some are tonally measured but structurally skewed and score ±2 to ±4. Some are
loudly polemical and score ±5. Measure what the article actually does, not what
"most reporting" supposedly does. There is no presumption of neutrality.

If you can't name the specific language, voice, source-selection, framing
emphasis, default-terminology, or omission choice the article made (as opposed
to the news event it covered), the direction scores should stay near 0.

Anchoring rules:
- Score based on textual evidence in the article body, not on what the headline
  or topic implies.
- 0-10 fields: 0 = absent, 5 = present but not dominant, 10 = dominant.
- -5 to +5 fields: 0 = balanced or not applicable; magnitude reflects how
  strongly the article positions itself, NOT the strength of the news event.
- Be conservative on extreme scores: don't assign ±5 (or 10) unless the
  dimension is unmistakably the dominant signal across the entire article.
- Respond with JSON only — no preface, no markdown fences."""


# ---------------------------------------------------------------------------
# Article → user message (identical structure to single-model version)
# ---------------------------------------------------------------------------

def _truncate_body(body: str, limit: int = MAX_BODY_CHARS) -> str:
    body = (body or "").strip()
    if len(body) <= limit:
        return body
    head = body[: int(limit * 0.7)]
    tail = body[-int(limit * 0.3):]
    return f"{head}\n\n[... article truncated ...]\n\n{tail}"


def build_user_message(article: dict) -> str:
    parts = []
    parts.append(f"HEADLINE: {article.get('headline') or '(none)'}")
    parts.append(f"BYLINE: {article.get('byline') or '(none)'}")
    parts.append(f"SECTION: {article.get('section') or '(unknown)'}")
    parts.append(f"PUB_DATE: {(article.get('pub_date') or '')[:10]}")
    if article.get("subsection"):
        parts.append(f"SUBSECTION: {article.get('subsection')}")
    if article.get("type_of_material"):
        parts.append(f"TYPE: {article.get('type_of_material')}")
    parts.append("")
    if article.get("abstract"):
        parts.append(f"ABSTRACT: {article['abstract']}")
    if article.get("lead_paragraph"):
        parts.append(f"LEAD: {article['lead_paragraph']}")
    keywords = article.get("keywords") or []
    subject_tags = [
        kw.get("value", "") for kw in keywords
        if isinstance(kw, dict) and (kw.get("name") or "").lower() == "subject"
    ]
    if subject_tags:
        parts.append(f"NYT SUBJECT TAGS: {', '.join(subject_tags[:8])}")
    body = (article.get("body") or "").strip()
    if body:
        parts.append("")
        parts.append("FULL ARTICLE TEXT:")
        parts.append(_truncate_body(body))
    else:
        parts.append("")
        parts.append("(No full body text available — score from headline + abstract + lead + keywords.)")
    return "\n".join(parts)


# ---------------------------------------------------------------------------
# Per-model parsing into a normalized Score
# ---------------------------------------------------------------------------

def _clamp(v, lo, hi) -> int:
    try:
        v = int(round(float(v)))
    except (TypeError, ValueError):
        return 0
    return max(lo, min(hi, v))


def _section(raw: dict, dims: list, default_lo: int = 0) -> dict:
    out = {}
    if not isinstance(raw, dict):
        raw = {}
    for key, lo, hi, _defn in dims:
        out[key] = _clamp(raw.get(key, default_lo), lo, hi)
    return out


@dataclass
class ModelScore:
    """One model's score for one article."""
    provider: str
    model: str
    topic: dict
    framing_direction: dict
    framing_tone: dict
    voice: dict
    confidence: int
    rationale: str
    error: Optional[str] = None

    def to_dict(self) -> dict:
        return {
            "provider": self.provider,
            "model": self.model,
            "topic": self.topic,
            "framing_direction": self.framing_direction,
            "framing_tone": self.framing_tone,
            "voice": self.voice,
            "confidence": self.confidence,
            "rationale": self.rationale,
            "error": self.error,
        }


def _from_raw(provider: str, model: str, raw: dict) -> ModelScore:
    return ModelScore(
        provider=provider,
        model=model,
        topic=_section(raw.get("topic"), TOPIC_DIMS),
        framing_direction=_section(raw.get("framing_direction"), FRAMING_DIRECTION_DIMS),
        framing_tone=_section(raw.get("framing_tone"), FRAMING_TONE_DIMS),
        voice=_section(raw.get("voice"), VOICE_DIMS),
        confidence=_clamp(raw.get("confidence"), 0, 10),
        rationale=str(raw.get("rationale") or "").strip()[:500],
    )


def _error_score(provider: str, model: str, err: str) -> ModelScore:
    return ModelScore(
        provider=provider, model=model,
        topic=_section({}, TOPIC_DIMS),
        framing_direction=_section({}, FRAMING_DIRECTION_DIMS),
        framing_tone=_section({}, FRAMING_TONE_DIMS),
        voice=_section({}, VOICE_DIMS),
        confidence=0, rationale="", error=err,
    )


# ---------------------------------------------------------------------------
# Per-article: run all 3 models in parallel
# ---------------------------------------------------------------------------

def _call_one_provider(provider: str, model: str, clients: ma.Clients,
                       user_msg: str) -> ModelScore:
    try:
        raw = ma.classify(
            provider=provider, clients=clients, model=model,
            system=SYSTEM_PROMPT, user=user_msg,
            max_tokens=MAX_OUTPUT_TOKENS, timeout=REQUEST_TIMEOUT,
        )
        return _from_raw(provider, model, raw)
    except ma.ProviderError as e:
        return _error_score(provider, model, str(e))
    except Exception as e:
        return _error_score(provider, model, f"unexpected: {type(e).__name__}: {e}")


def _reusable_provider_score(article: dict, provider: str,
                              model: str) -> Optional[ModelScore]:
    """If the article already has a successful per_model score from a prior
    consensus run for this provider+model+prompt_version combination, lift it
    into a ModelScore so the consensus run doesn't pay for a fresh call.

    For Anthropic, we also fall back to the single-model llm_scores field
    (which stores results from prior llm_classify single-model runs).
    """
    cons = article.get("llm_consensus") or {}
    per_model = (cons.get("per_model_scores") or {}).get(provider) or {}
    if per_model and not per_model.get("error"):
        if ((per_model.get("model") or "") == model
                and (cons.get("prompt_version", 0) or 0) >= PROMPT_VERSION):
            return ModelScore(
                provider=provider,
                model=model,
                topic=per_model.get("topic", {}),
                framing_direction=per_model.get("framing_direction", {}),
                framing_tone=per_model.get("framing_tone", {}),
                voice=per_model.get("voice", {}),
                confidence=per_model.get("confidence", 0),
                rationale=per_model.get("rationale", ""),
                error=None,
            )

    # Anthropic-only fallback: single-model llm_scores from prior llm_classify runs
    if provider == "anthropic":
        scores = article.get("llm_scores") or {}
        if not scores or scores.get("error"):
            return None
        if (scores.get("model") or "") != model:
            return None
        if (scores.get("prompt_version", 0) or 0) < PROMPT_VERSION:
            return None
        return ModelScore(
            provider="anthropic",
            model=model,
            topic=scores.get("topic", {}),
            framing_direction=scores.get("framing_direction", {}),
            framing_tone=scores.get("framing_tone", {}),
            voice=scores.get("voice", {}),
            confidence=scores.get("confidence", 0),
            rationale=scores.get("rationale", ""),
            error=None,
        )
    return None


# Kept for backwards-compat — same behavior, restricted to anthropic
def _reusable_anthropic_score(article: dict, anthropic_model: str) -> Optional[ModelScore]:
    return _reusable_provider_score(article, "anthropic", anthropic_model)


def classify_one(article: dict, clients: ma.Clients,
                 models: dict = DEFAULT_MODELS) -> tuple[dict, dict]:
    """Run configured providers on one article. Returns (consensus_record, call_stats)
    where call_stats counts {"reused": int, "called": int} so the bulk caller can
    track how many API calls were actually saved by reuse."""
    user_msg = build_user_message(article)
    had_body = bool((article.get("body") or "").strip())

    results: dict[str, ModelScore] = {}
    call_stats = {"reused": 0, "called": 0}

    # Try to reuse a prior successful score for each provider before paying
    # for a fresh call. Anthropic also looks at single-model llm_scores;
    # OpenAI/Gemini only look at prior consensus per_model_scores.
    providers_to_call = dict(models)
    for provider, model in list(models.items()):
        reused = _reusable_provider_score(article, provider, model)
        if reused:
            results[provider] = reused
            providers_to_call.pop(provider, None)
            call_stats["reused"] += 1

    if providers_to_call:
        with ThreadPoolExecutor(max_workers=len(providers_to_call)) as pool:
            futures = {
                pool.submit(_call_one_provider, provider, model, clients, user_msg): provider
                for provider, model in providers_to_call.items()
            }
            for fut in as_completed(futures):
                provider = futures[fut]
                results[provider] = fut.result()
                call_stats["called"] += 1

    return _aggregate(results, had_body), call_stats


# ---------------------------------------------------------------------------
# Consensus aggregation: simple mean across providers
#
# Each dimension's consensus value is the unweighted mean of all valid model
# scores. Per-dimension standard deviation is computed alongside and exposed
# as `consensus_sd` so the dashboard can surface disagreement without it
# biasing the aggregate.
#
# `consensus_certainty` (1 / (1 + var)) is kept as a 0-1 headline number for
# backwards-compat with existing UI code that filters/sorts by it, but it no
# longer reweights anything inside the aggregator.
# ---------------------------------------------------------------------------

def _mean_and_dispersion(values: list[float]) -> tuple[float, float, float]:
    """Returns (mean, stdev, certainty 0-1).

    With <2 values, falls back to the lone value (or 0), sd=0, and a low
    certainty so single-model articles aren't mistaken for high agreement.
    With identical values, certainty = 1.0 (zero variance).
    """
    if not values:
        return 0.0, 0.0, 0.0
    if len(values) == 1:
        return values[0], 0.0, 0.3  # one model alone — modest certainty, no SD
    mean = sum(values) / len(values)
    var = statistics.pvariance(values)
    sd = var ** 0.5
    certainty = 1.0 / (1.0 + var)
    return mean, sd, certainty


def _aggregate(results: dict[str, ModelScore], had_body: bool) -> dict:
    """Combine per-model scores into a consensus record using simple mean."""
    valid = {p: s for p, s in results.items() if not s.error}
    failed = {p: s.error for p, s in results.items() if s.error}

    per_model = {p: s.to_dict() for p, s in results.items()}

    if not valid:
        # All providers failed — return empty consensus + error info
        return {
            "consensus_scores": _empty_consensus_scores(),
            "consensus_sd": _empty_consensus_sd(),
            "consensus_certainty": 0.0,
            "per_model_scores": per_model,
            "models_used": [],
            "models_failed": failed,
            "had_body": had_body,
            "prompt_version": PROMPT_VERSION,
            "error": "all providers failed",
        }

    def merge_dim_set(dims: list, section: str) -> tuple[dict, dict, list[float]]:
        merged = {}
        sd_out = {}
        per_dim_certainty = []
        for key, _lo, _hi, _defn in dims:
            values = [getattr(s, section).get(key, 0) for s in valid.values()]
            mean, sd, cert = _mean_and_dispersion(values)
            merged[key] = round(mean, 2)
            sd_out[key] = round(sd, 2)
            per_dim_certainty.append(cert)
        return merged, sd_out, per_dim_certainty

    topic, topic_sd, _ = merge_dim_set(TOPIC_DIMS, "topic")
    fd, fd_sd, fd_cert = merge_dim_set(FRAMING_DIRECTION_DIMS, "framing_direction")
    ft, ft_sd, ft_cert = merge_dim_set(FRAMING_TONE_DIMS, "framing_tone")
    voice, voice_sd, _ = merge_dim_set(VOICE_DIMS, "voice")

    # Headline certainty: mean across the most-bias-sensitive dims
    # (framing_direction + framing_tone). Reported for UI use only — it does
    # NOT reweight the aggregate.
    headline_certainty = (sum(fd_cert) + sum(ft_cert)) / (len(fd_cert) + len(ft_cert))

    # Per-section certainty also exposed for downstream filtering
    section_certainty = {
        "framing_direction": sum(fd_cert) / len(fd_cert),
        "framing_tone": sum(ft_cert) / len(ft_cert),
    }

    consensus_scores = {
        "topic": topic,
        "framing_direction": fd,
        "framing_tone": ft,
        "voice": voice,
        # Confidence: simple mean across models (each model's self-rated
        # certainty in its own scoring, not the cross-model agreement).
        "confidence": round(sum(s.confidence for s in valid.values()) / len(valid), 2),
        # Pick one rationale to surface (longest from the highest-confidence model).
        # All rationales remain in per_model_scores.
        "rationale": max(valid.values(), key=lambda s: (s.confidence, len(s.rationale))).rationale,
    }

    consensus_sd = {
        "topic": topic_sd,
        "framing_direction": fd_sd,
        "framing_tone": ft_sd,
        "voice": voice_sd,
    }

    return {
        "consensus_scores": consensus_scores,
        "consensus_sd": consensus_sd,
        "consensus_certainty": round(headline_certainty, 3),
        "section_certainty": {k: round(v, 3) for k, v in section_certainty.items()},
        "per_model_scores": per_model,
        "models_used": sorted(valid.keys()),
        "models_failed": failed,
        "had_body": had_body,
        "prompt_version": PROMPT_VERSION,
        "error": None,
    }


def _empty_consensus_scores() -> dict:
    return {
        "topic": _section({}, TOPIC_DIMS),
        "framing_direction": _section({}, FRAMING_DIRECTION_DIMS),
        "framing_tone": _section({}, FRAMING_TONE_DIMS),
        "voice": _section({}, VOICE_DIMS),
        "confidence": 0,
        "rationale": "",
    }


def _empty_consensus_sd() -> dict:
    """Empty SD record — same shape as consensus_scores minus confidence/rationale."""
    return {
        "topic": {k: 0.0 for k, *_ in TOPIC_DIMS},
        "framing_direction": {k: 0.0 for k, *_ in FRAMING_DIRECTION_DIMS},
        "framing_tone": {k: 0.0 for k, *_ in FRAMING_TONE_DIMS},
        "voice": {k: 0.0 for k, *_ in VOICE_DIMS},
    }


# ---------------------------------------------------------------------------
# Bulk classification
# ---------------------------------------------------------------------------

def classify_all(
    articles: list,
    anthropic_key: str,
    openai_key: str,
    gemini_key: str,
    models: dict = DEFAULT_MODELS,
    ollama_url: Optional[str] = None,
    progress_cb: Optional[Callable[[int, int, dict, dict], None]] = None,
    skip_existing: bool = True,
    cancel_check: Optional[Callable[[], bool]] = None,
    inter_request_delay: float = 0.0,
) -> dict:
    """Classify a list of articles in place, mutating each with
    `consensus_scores` (the new field) alongside per_model_scores.

    `models` keys identify which providers to call. Include "ollama" only
    when `ollama_url` is also set; the launcher decides this at runtime
    based on a reachability probe.

    Smart-skip: skips if existing consensus has all required providers under
    the current prompt_version AND (already had body OR no body now).
    """
    clients = ma.build_clients(anthropic_key, openai_key, gemini_key,
                                ollama_url=ollama_url)

    total = len(articles)
    stats = {"total": total, "classified": 0, "skipped": 0,
             "upgraded": 0, "errors": 0, "error_types": {},
             # Tracks how many per-article Anthropic calls were saved by
             # reusing a prior single-model Haiku score (or a prior consensus
             # per_model.anthropic record).
             "anthropic_reused": 0}

    required_providers = set(models.keys())

    for i, article in enumerate(articles, start=1):
        if cancel_check and cancel_check():
            break

        existing = article.get("llm_consensus") or {}
        existing_models = set(existing.get("models_used") or [])
        existing_prompt = existing.get("prompt_version", 0) or 0
        had_body_then = bool(existing.get("had_body"))
        has_body_now = bool((article.get("body") or "").strip())

        # Skip when existing record covers all required providers under the
        # current prompt AND body availability is at least as good
        if skip_existing and existing.get("consensus_scores"):
            body_ok = had_body_then or not has_body_now
            providers_ok = required_providers.issubset(existing_models)
            prompt_ok = existing_prompt >= PROMPT_VERSION
            if body_ok and providers_ok and prompt_ok:
                stats["skipped"] += 1
                if progress_cb:
                    progress_cb(i, total, article, existing)
                continue
            stats["upgraded"] += 1

        consensus, call_stats = classify_one(article, clients, models=models)
        article["llm_consensus"] = consensus
        stats["anthropic_reused"] += call_stats.get("reused", 0)

        if consensus.get("error"):
            stats["errors"] += 1
            err = consensus["error"]
            stats["error_types"][err] = stats["error_types"].get(err, 0) + 1
        elif len(consensus.get("models_used") or []) < len(required_providers):
            # Partial success — some providers failed but consensus has at
            # least one valid score
            for err in (consensus.get("models_failed") or {}).values():
                key = f"partial:{err[:60]}"
                stats["error_types"][key] = stats["error_types"].get(key, 0) + 1
            stats["classified"] += 1
        else:
            stats["classified"] += 1

        if progress_cb:
            progress_cb(i, total, article, consensus)

        if inter_request_delay:
            time.sleep(inter_request_delay)

    return stats


# ---------------------------------------------------------------------------
# Reaggregation: apply the current aggregation rule to existing per-model
# scores without making any API calls. Use after the aggregation logic
# changes (e.g. switching from inverse-variance weighting to simple mean)
# so already-classified articles inherit the new rule without paying for
# fresh LLM calls.
# ---------------------------------------------------------------------------

def reaggregate_all(articles: list,
                    include_providers: Optional[set] = None) -> dict:
    """Recompute consensus_scores / consensus_sd / consensus_certainty from
    each article's existing per_model_scores. Returns a stats dict.

    include_providers: when given, only these providers' scores survive into
    the rebuilt record — both the aggregation AND the stored per_model_scores.
    Used to drop a provider from the corpus (archive its raw scores first)."""
    stats = {"reaggregated": 0, "no_per_model": 0, "all_errors": 0}
    for article in articles:
        cons = article.get("llm_consensus") or {}
        per_model = cons.get("per_model_scores") or {}
        if not per_model:
            stats["no_per_model"] += 1
            continue

        results: dict[str, ModelScore] = {}
        for provider, data in per_model.items():
            if include_providers is not None and provider not in include_providers:
                continue
            if not isinstance(data, dict):
                continue
            model = data.get("model", "") or ""
            err = data.get("error")
            if err:
                results[provider] = _error_score(provider, model, err)
            else:
                results[provider] = ModelScore(
                    provider=provider,
                    model=model,
                    topic=data.get("topic", {}) or {},
                    framing_direction=data.get("framing_direction", {}) or {},
                    framing_tone=data.get("framing_tone", {}) or {},
                    voice=data.get("voice", {}) or {},
                    confidence=data.get("confidence", 0) or 0,
                    rationale=data.get("rationale", "") or "",
                    error=None,
                )

        new_cons = _aggregate(results, had_body=bool(cons.get("had_body")))
        # Preserve the original prompt_version — re-aggregation isn't a re-classify
        new_cons["prompt_version"] = cons.get("prompt_version", PROMPT_VERSION)
        article["llm_consensus"] = new_cons

        if new_cons.get("error") == "all providers failed":
            stats["all_errors"] += 1
        else:
            stats["reaggregated"] += 1
    return stats


# ---------------------------------------------------------------------------
# Aggregation helpers used by the dashboard exporter
# ---------------------------------------------------------------------------

def summary_matrix(articles: list, weight_by_certainty: bool = False) -> dict:
    """Compute averages across dimensions. By default uses simple unweighted
    means so every classified article counts equally — high-disagreement
    articles aren't downweighted because cross-model agreement is not a
    reliable proxy for accuracy on subjective dimensions.

    Pass weight_by_certainty=True to upweight high-agreement articles (legacy
    behavior — note this can amplify correlated-error bias when multiple
    models share the same blind spot)."""
    classified = [a for a in articles
                  if a.get("llm_consensus")
                  and not a["llm_consensus"].get("error")
                  and a["llm_consensus"].get("consensus_scores")]
    n = len(classified)
    if n == 0:
        return {"n": 0, "weighted": weight_by_certainty}

    def avg(section_key: str, dims: list) -> dict:
        out = {}
        for key, _lo, _hi, _defn in dims:
            total_w = 0.0
            total_wv = 0.0
            for a in classified:
                v = (a["llm_consensus"]["consensus_scores"].get(section_key) or {}).get(key, 0)
                w = a["llm_consensus"].get("consensus_certainty", 0.5) if weight_by_certainty else 1.0
                total_w += w
                total_wv += w * v
            out[key] = round(total_wv / total_w, 2) if total_w else 0
        return out

    mean_certainty = sum(a["llm_consensus"].get("consensus_certainty", 0) for a in classified) / n

    return {
        "n": n,
        "weighted": weight_by_certainty,
        "mean_certainty": round(mean_certainty, 3),
        "topic": avg("topic", TOPIC_DIMS),
        "framing_direction": avg("framing_direction", FRAMING_DIRECTION_DIMS),
        "framing_tone": avg("framing_tone", FRAMING_TONE_DIMS),
        "voice": avg("voice", VOICE_DIMS),
        "schema": {
            "topic": [{"key": k, "min": lo, "max": hi, "label": d} for k, lo, hi, d in TOPIC_DIMS],
            "framing_direction": [{"key": k, "min": lo, "max": hi, "label": d} for k, lo, hi, d in FRAMING_DIRECTION_DIMS],
            "framing_tone": [{"key": k, "min": lo, "max": hi, "label": d} for k, lo, hi, d in FRAMING_TONE_DIMS],
            "voice": [{"key": k, "min": lo, "max": hi, "label": d} for k, lo, hi, d in VOICE_DIMS],
        },
    }
