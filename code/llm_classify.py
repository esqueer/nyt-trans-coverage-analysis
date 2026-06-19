#!/usr/bin/env python3
"""LLM-based multi-dimensional classifier for NYT trans-coverage articles.

Each article is scored across ~20 dimensions covering what it covers (topic),
how it frames the issue (direction + tone), and whose voices are present
(sourcing). Replaces / supplements the keyword classifier with something
that can read implicit framing, irony, and journalistic tone.

Output schema (per article):
    topic: dict[str, int 0-10]   - what the article covers
    framing_direction: dict[str, int -5..+5]  - which way it leans
    framing_tone: dict[str, int 0-10]  - rhetorical mode
    voice: dict[str, int 0-10]   - who's quoted / centered
    confidence: int 0-10
    rationale: str

The full schema is fixed at module level so downstream code can compute
matrix views without re-deriving keys.
"""
from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass, field
from typing import Callable, Optional

try:
    from anthropic import Anthropic, APIStatusError, APIConnectionError, RateLimitError
except ImportError as e:
    raise ImportError(
        "anthropic SDK required. Install with: pip install anthropic"
    ) from e


# Default model — Haiku 4.5 is fast and cheap for classification work.
DEFAULT_MODEL = "claude-haiku-4-5"
MAX_OUTPUT_TOKENS = 1200  # v3 needs more headroom for detailed rationales
REQUEST_TIMEOUT = 60
MAX_BODY_CHARS = 12000  # cap input length so bills stay predictable


# ---------------------------------------------------------------------------
# Score schema
# ---------------------------------------------------------------------------

# Each dimension has (key, scale_min, scale_max, definition).
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

ALL_DIMS = TOPIC_DIMS + FRAMING_DIRECTION_DIMS + FRAMING_TONE_DIMS + VOICE_DIMS


# ---------------------------------------------------------------------------
# Prompt construction
# ---------------------------------------------------------------------------

def _format_schema_for_prompt() -> str:
    lines = ["TOPIC (0-10, multi-label, several can be high simultaneously):"]
    for key, lo, hi, defn in TOPIC_DIMS:
        lines.append(f"  - {key}: {defn}")
    lines.append("")
    lines.append("FRAMING_DIRECTION (-5 to +5, signed; 0 = neutral or balanced):")
    for key, lo, hi, defn in FRAMING_DIRECTION_DIMS:
        lines.append(f"  - {key}: {defn}")
    lines.append("")
    lines.append("FRAMING_TONE (0-10):")
    for key, lo, hi, defn in FRAMING_TONE_DIMS:
        lines.append(f"  - {key}: {defn}")
    lines.append("")
    lines.append("VOICE (0-10):")
    for key, lo, hi, defn in VOICE_DIMS:
        lines.append(f"  - {key}: {defn}")
    return "\n".join(lines)


# Bump this when SYSTEM_PROMPT changes meaningfully so classify_all can
# detect stale scores and re-classify under skip_existing.
PROMPT_VERSION = 3

SYSTEM_PROMPT = f"""You are an objective media-analysis assistant scoring NYT articles on transgender coverage across multiple dimensions. Your goal is consistent, evidence-based scoring — not advocacy.

For each article, return a JSON object with these dimensions:

{_format_schema_for_prompt()}

Plus:
  - confidence: 0-10 (how sure you are about your scores given available text)
  - rationale: ONE OR TWO SENTENCES maximum (do not exceed 400 characters)
    naming the strongest signal that drove the directional scores. Be terse —
    rationales over 400 chars cause output truncation and corrupt the JSON.

CRITICAL RULE — separate WHAT HAPPENED from HOW IT'S COVERED:

The framing_direction scores (skepticism_to_affirming and restrictive_to_protective)
measure THE ARTICLE'S OWN FRAMING — both tonal and structural — NOT the outcome of
the event being reported.

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


def _truncate_body(body: str, limit: int = MAX_BODY_CHARS) -> str:
    body = (body or "").strip()
    if len(body) <= limit:
        return body
    # Keep beginning + end (where the conclusion / kicker often lives)
    head = body[: int(limit * 0.7)]
    tail = body[-int(limit * 0.3):]
    return f"{head}\n\n[... article truncated ...]\n\n{tail}"


def build_user_message(article: dict) -> str:
    """Assemble the article payload sent to the LLM."""
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

    # NYT subject keywords — useful framing context
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
# Response parsing
# ---------------------------------------------------------------------------

@dataclass
class Score:
    """Validated score record for a single article."""
    topic: dict
    framing_direction: dict
    framing_tone: dict
    voice: dict
    confidence: int
    rationale: str
    model: str
    error: Optional[str] = None
    # Whether the article body text was available at classification time.
    # Lets us automatically re-score articles whose bodies got fetched
    # between LLM runs without redoing already-best-effort calls.
    had_body: bool = False
    # Records the prompt schema version used. Bumped whenever SYSTEM_PROMPT
    # changes meaningfully (e.g., new scoring rules). Smart-skip can detect
    # older versions and re-classify for consistency.
    prompt_version: int = 0

    def to_dict(self) -> dict:
        return {
            "topic": self.topic,
            "framing_direction": self.framing_direction,
            "framing_tone": self.framing_tone,
            "voice": self.voice,
            "confidence": self.confidence,
            "rationale": self.rationale,
            "model": self.model,
            "error": self.error,
            "had_body": self.had_body,
            "prompt_version": self.prompt_version,
        }


def _clamp(v, lo, hi) -> int:
    try:
        v = int(round(float(v)))
    except (TypeError, ValueError):
        return 0
    return max(lo, min(hi, v))


def _section(raw: dict, dims: list, default_lo: int = 0) -> dict:
    """Extract one nested section from the LLM output, clamping each value."""
    out = {}
    if not isinstance(raw, dict):
        raw = {}
    for key, lo, hi, _defn in dims:
        out[key] = _clamp(raw.get(key, default_lo), lo, hi)
    return out


def parse_response(text: str, model: str) -> Score:
    """Parse the LLM JSON response into a validated Score."""
    # Strip markdown fences if the model added them despite instructions
    text = text.strip()
    text = re.sub(r"^```(?:json)?\s*", "", text)
    text = re.sub(r"\s*```\s*$", "", text)

    try:
        data = json.loads(text)
    except json.JSONDecodeError as e:
        # Try to extract the first {...} block as a fallback
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if match:
            try:
                data = json.loads(match.group(0))
            except json.JSONDecodeError:
                return _empty_score(model, error=f"json parse: {e}")
        else:
            return _empty_score(model, error=f"json parse: {e}")

    return Score(
        topic=_section(data.get("topic"), TOPIC_DIMS),
        framing_direction=_section(data.get("framing_direction"), FRAMING_DIRECTION_DIMS),
        framing_tone=_section(data.get("framing_tone"), FRAMING_TONE_DIMS),
        voice=_section(data.get("voice"), VOICE_DIMS),
        confidence=_clamp(data.get("confidence"), 0, 10),
        rationale=str(data.get("rationale") or "").strip()[:500],
        model=model,
        error=None,
    )


def _empty_score(model: str, error: str) -> Score:
    return Score(
        topic=_section({}, TOPIC_DIMS),
        framing_direction=_section({}, FRAMING_DIRECTION_DIMS),
        framing_tone=_section({}, FRAMING_TONE_DIMS),
        voice=_section({}, VOICE_DIMS),
        confidence=0,
        rationale="",
        model=model,
        error=error,
    )


# ---------------------------------------------------------------------------
# Classification
# ---------------------------------------------------------------------------

def classify_one(client: Anthropic, article: dict, model: str = DEFAULT_MODEL) -> Score:
    """Classify a single article. Retries transient errors a couple of times."""
    user_msg = build_user_message(article)
    had_body = bool((article.get("body") or "").strip())

    def _stamp(score: Score) -> Score:
        score.had_body = had_body
        score.prompt_version = PROMPT_VERSION
        return score

    for attempt, backoff in enumerate([0, 4, 12], start=1):
        if backoff:
            time.sleep(backoff)
        try:
            resp = client.messages.create(
                model=model,
                max_tokens=MAX_OUTPUT_TOKENS,
                temperature=0.0,  # deterministic-ish
                system=SYSTEM_PROMPT,
                messages=[{"role": "user", "content": user_msg}],
                timeout=REQUEST_TIMEOUT,
            )
            text = "".join(
                block.text for block in resp.content
                if getattr(block, "type", None) == "text"
            )
            return _stamp(parse_response(text, model))
        except RateLimitError:
            continue
        except (APIConnectionError, APIStatusError) as e:
            if attempt == 3:
                return _stamp(_empty_score(model, error=f"api: {type(e).__name__}"))
            continue
        except Exception as e:
            return _stamp(_empty_score(model, error=f"unexpected: {type(e).__name__}: {e}"))

    return _stamp(_empty_score(model, error="rate-limited after retries"))


def classify_all(
    articles: list,
    api_key: str,
    progress_cb: Optional[Callable[[int, int, dict, Score], None]] = None,
    model: str = DEFAULT_MODEL,
    skip_existing: bool = True,
    cancel_check: Optional[Callable[[], bool]] = None,
    inter_request_delay: float = 0.0,
    require_model_match: bool = False,
) -> dict:
    """Classify a list of articles in place, mutating each with `llm_scores`.

    `progress_cb(i, total, article, score)` is called after every attempt.
    `cancel_check()` should return True if the caller wants to stop early.
    `require_model_match`: when True and skip_existing is True, articles whose
    existing scores were generated by a DIFFERENT model than the requested one
    are re-classified so the corpus can be made model-consistent.
    """
    client = Anthropic(api_key=api_key)
    total = len(articles)
    stats = {"total": total, "classified": 0, "skipped": 0, "upgraded": 0,
             "model_upgraded": 0, "prompt_upgraded": 0,
             "errors": 0, "error_types": {}}

    for i, article in enumerate(articles, start=1):
        if cancel_check and cancel_check():
            break

        existing = article.get("llm_scores") or {}
        existing_ok = bool(existing) and not existing.get("error")
        had_body_then = bool(existing.get("had_body"))
        has_body_now = bool((article.get("body") or "").strip())
        prev_model = (existing.get("model") or "").strip()
        prev_prompt_version = existing.get("prompt_version", 0) or 0

        # Smart skip: only skip if the existing classification is at least as
        # good as what a fresh call would produce. Re-classify when ANY of:
        #   1. Old call was metadata-only AND body has since been fetched
        #   2. require_model_match is on AND the existing score was made with
        #      a different model than the current request
        #   3. Existing score was made with an older prompt version
        if skip_existing and existing_ok:
            body_optimal = had_body_then or not has_body_now
            model_optimal = (not require_model_match) or (prev_model == model)
            prompt_optimal = prev_prompt_version >= PROMPT_VERSION
            if body_optimal and model_optimal and prompt_optimal:
                stats["skipped"] += 1
                if progress_cb:
                    progress_cb(i, total, article, Score(
                        topic=existing.get("topic", {}),
                        framing_direction=existing.get("framing_direction", {}),
                        framing_tone=existing.get("framing_tone", {}),
                        voice=existing.get("voice", {}),
                        confidence=existing.get("confidence", 0),
                        rationale=existing.get("rationale", ""),
                        model=prev_model,
                        had_body=had_body_then,
                        prompt_version=prev_prompt_version,
                    ))
                continue
            # Fall through: body / model / prompt upgrade
            if not body_optimal:
                stats["upgraded"] += 1
            elif not model_optimal:
                stats["model_upgraded"] += 1
            elif not prompt_optimal:
                stats["prompt_upgraded"] += 1

        score = classify_one(client, article, model=model)
        article["llm_scores"] = score.to_dict()

        if score.error:
            stats["errors"] += 1
            stats["error_types"][score.error] = stats["error_types"].get(score.error, 0) + 1
        else:
            stats["classified"] += 1

        if progress_cb:
            progress_cb(i, total, article, score)

        if inter_request_delay:
            time.sleep(inter_request_delay)

    return stats


# ---------------------------------------------------------------------------
# Aggregation helpers (used by the dashboard exporter)
# ---------------------------------------------------------------------------

def summary_matrix(articles: list) -> dict:
    """Compute average scores across each dimension for use in dashboard."""
    classified = [a for a in articles if a.get("llm_scores") and not a["llm_scores"].get("error")]
    n = len(classified)
    if n == 0:
        return {"n": 0}

    def avg(section_key, dims):
        out = {}
        for key, _lo, _hi, _defn in dims:
            vals = [
                (a["llm_scores"].get(section_key) or {}).get(key, 0)
                for a in classified
            ]
            out[key] = round(sum(vals) / n, 2) if vals else 0
        return out

    return {
        "n": n,
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
