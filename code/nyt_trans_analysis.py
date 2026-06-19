#!/usr/bin/env python3
"""
NYT Trans Coverage Analysis Pipeline
=====================================
Collects and analyzes New York Times article metadata related to transgender
coverage, measuring editorial placement hierarchy, byline concentration,
framing patterns, and sourcing asymmetries.

Requires: NYT Developer API key (free at developer.nytimes.com)
Rate limits: 500 requests/day, 5 requests/minute

Usage:
    # Collect data
    python nyt_trans_analysis.py collect --api-key YOUR_KEY --start 2022-01-01 --end 2025-12-31

    # Analyze previously collected data
    python nyt_trans_analysis.py analyze --input collected_articles.json

    # Full pipeline
    python nyt_trans_analysis.py full --api-key YOUR_KEY --start 2022-01-01 --end 2025-12-31
"""

import json
import time
import argparse
import re
import os
import sys
from datetime import datetime, timedelta
from collections import Counter, defaultdict
from pathlib import Path

try:
    import requests
except ImportError:
    print("Install requests: pip install requests")
    sys.exit(1)


# Force UTF-8 stdout on Windows so unicode bar chars and headlines print cleanly
if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass


SCRIPT_DIR = Path(__file__).resolve().parent
# Mutable so the launcher (or anything embedding this module) can repoint it
# to the user's workspace dir, which differs from __file__ when running as
# a PyInstaller --onefile exe.
API_KEY_FILE = SCRIPT_DIR / "API.txt"


def load_api_key(cli_value: str = None) -> str:
    """Resolve API key from CLI arg, then API.txt, then env var."""
    if cli_value:
        return cli_value.strip()
    if API_KEY_FILE.exists():
        key = API_KEY_FILE.read_text(encoding="utf-8").strip()
        if key:
            return key
    env_key = os.environ.get("NYT_API_KEY", "").strip()
    if env_key:
        return env_key
    raise RuntimeError(
        "No NYT API key found. Pass --api-key, write it to "
        f"{API_KEY_FILE}, or set NYT_API_KEY. Get a free key at "
        "https://developer.nytimes.com"
    )


# ---------------------------------------------------------------------------
# SEARCH TERMS AND CLASSIFICATION
# ---------------------------------------------------------------------------

# Primary search queries to capture trans-related coverage
SEARCH_QUERIES = [
    '"transgender"',
    '"gender-affirming care"',
    '"gender affirming care"',
    '"gender identity"',
    '"trans youth"',
    '"trans kids"',
    '"trans children"',
    '"trans women"',
    '"trans men"',
    '"gender dysphoria"',
    '"puberty blockers"',
    '"sex reassignment"',
    '"gender transition"',
    '"detransition"',
    '"cisgender"',
    '"nonbinary"',
    '"trans rights"',
    '"transgender military"',
    '"transgender athlete"',
    '"transgender bathroom"',
    '"gender clinic"',
    '"WPATH"',
]

# Framing classification keywords
# Each article gets classified by dominant framing based on headline + abstract
FRAMING_CATEGORIES = {
    "medical_skepticism": {
        "keywords": [
            # Treatment vocabulary — broad enough to catch the entire
            # medical-skepticism beat (Azeen Ghorayshi, Emily Bazelon, etc.)
            # since NYT coverage of trans medicine is overwhelmingly framed
            # through doubt or contestation even when individual headlines
            # don't read as explicitly skeptical.
            "puberty blocker", "puberty blockers", "puberty-blocking",
            "puberty suppression", "puberty suppressors",
            "hormone therapy", "hormone treatment", "hormone blocker",
            "cross-sex hormone", "cross sex hormone",
            "gender medicine", "gender treatment", "gender treatments",
            "gender care", "gender therapy", "gender-affirming care",
            "gender affirming care", "gender clinic",
            "youth gender", "youth medical", "youth transition",
            "child gender transition", "transgender care",
            "trans youth medical", "medical care for transgender",
            "treatment for transgender", "treatment for trans",
            "transition surgery", "top surgery",
            # Detransition / regret narratives
            "detransition", "detransitioner", "detransitioners",
            "regret transitioning", "regret their transition",
            "desisting", "desisters", "watchful waiting",
            "wait-and-see approach",
            # Skepticism / evidence / debate vocabulary
            "side effect", "side effects", "serious complication",
            "long-term risk", "long term risk", "health risk",
            "medical risk", "lifelong risk",
            "safety concern", "safety concerns",
            "raised concerns about", "growing concern about",
            "scientific debate", "medical debate", "doctors debate",
            "scientific evidence", "medical evidence",
            "evidence base", "evidence-base",
            "lack of evidence", "low quality evidence",
            "low-quality evidence", "quality of evidence",
            "weak evidence", "limited evidence", "no evidence",
            "study casts doubt", "raises questions about treatment",
            "raises questions about care",
            "reversible", "irreversible",
            "out of date", "outdated", "denounce", "denounces",
            "denounced", "remove age limit", "remove age limits",
            "age limit", "age limits", "age restriction",
            # Specific reports / named figures
            "cass review", "cass report", "hilary cass", "dr. cass",
            "systematic review", "exploratory therapy",
            "too far", "too fast",
            "the protocol",  # Azeen's NYT podcast/series
            "investigate medical", "investigate gender",
            "medical research on", "research on transgender",
            "stop covering", "halt", "pause",  # institutional pull-back vocabulary
            # Health-research vocabulary that pulls in the broader beat
            "rise in transgender", "increase in transgender",
            "sharp rise", "suicide risk", "suicide rate", "suicide rates",
            "national survey", "national study", "first national",
            "trans health", "transgender health",
            "change their minds", "changed their minds",
            "identify as transgender", "transgender adolescents",
            "youth identification",
        ],
        "description": "Frames coverage of gender-affirming care through skepticism — the contested medical/scientific debate beat (Cass Report fallout, detransitioner discourse, evidence-quality questions, state restrictions on care)",
    },
    "legislative_neutral": {
        "keywords": [
            "ban", "bill", "legislation", "law", "governor sign",
            "state legislature", "republican", "democrat",
            "executive order", "policy", "regulation", "rule",
            "supreme court", "court ruling", "appeals court",
            "challenge", "lawsuit", "constitutional",
            # Institutional-policy vocabulary that doesn't always trip
            # legislation-keywords but represents the same framing
            "title ix", "pentagon", "military", "soldier", "recruit",
            "prison", "jail", "inmate", "prisoner",
            "school district", "federal court", "agency",
            "ruling", "restriction", "mandate", "guidance",
            "department of", "biden administration", "trump administration",
        ],
        "description": "Legislative/legal/institutional-policy coverage without centering affected communities",
    },
    "rights_affirming": {
        "keywords": [
            "rights", "protection", "discrimination", "equality",
            "support", "affirm", "celebrate", "community",
            "pride", "visibility", "resilience", "family support",
            "acceptance", "wellbeing", "mental health benefit",
            "positive outcome", "thriving",
        ],
        "description": "Centers trans people's rights, dignity, or positive outcomes",
    },
    "culture_war": {
        "keywords": [
            "culture war", "controversy", "backlash", "divide",
            "battle", "fight", "clash", "both sides",
            "parents' rights", "parental rights", "religious liberty",
            "conservative", "liberal", "woke", "ideology",
            "bathroom", "locker room", "sports", "athlete",
            "fairness", "biological",
            # NOTE: "debate" intentionally excluded — it disproportionately fires
            # on "doctors debate" / "scientific debate" articles which are
            # actually medical_debate framings, not bathroom-bill culture-war
            # framings. Specific "bathroom debate" / "sports debate" still
            # match via the other keywords above.
        ],
        "description": "Frames trans issues as political/cultural conflict",
    },
    "human_interest": {
        "keywords": [
            "family", "parent", "child", "teenager", "story",
            "journey", "experience", "life", "personal",
            "navigate", "struggle", "cope", "coming out",
        ],
        "description": "Individual stories and personal narratives",
    },
    "international": {
        "keywords": [
            "britain", "uk", "nhs", "england", "europe",
            "canada", "australia", "finland", "sweden",
            "international", "global",
        ],
        "description": "International trans policy/coverage",
    },
    "arts_culture": {
        "keywords": [
            # Books / publishing. Bare "review" excluded — "Cass review",
            # "policy review" etc.
            "novel", "novelist", "memoir", "memoirs",
            "autobiography", "paperback", "fiction writer",
            "nonfiction writer", "author of", "poet", "poetry",
            "book review", "best books", "new book",
            # Film / TV. Bare "show" / "series" REMOVED — they match
            # "study shows" and "series of bills" in non-arts contexts.
            # Multi-word forms still match real arts coverage.
            "film", "movie", "documentary",
            "oscar nomination", "oscar-winning",
            "emmy nomination", "emmy-winning", "sundance",
            "netflix series", "netflix film", "hbo series",
            "hbo documentary", "amazon prime",
            "tv show", "tv series", "television series",
            "television show", "miniseries",
            "season premiere", "series premiere",
            "starring", "stars in", "stars as", "starring role",
            "cast as", "casting", "title role",
            # Stage
            "play", "playwright", "theater", "theatre", "broadway",
            "off-broadway", "off broadway", "stage production",
            "musical theater",
            # Music
            "album", "musician", "singer", "songwriter",
            "concert tour", "rapper", "pop star",
            # Visual art / fashion / culture
            "artist", "painting", "sculpture", "exhibition",
            "art exhibit", "museum", "gallery",
            "photography exhibit", "photographer",
            "fashion week", "runway show", "fashion designer",
            "red carpet", "hollywood", "celebrity profile",
            "drag queen", "drag race", "drag show", "in drag",
        ],
        "description": "Cultural reception: reviews, profiles, and criticism of trans-related artists/works",
    },
    "violence_safety": {
        "keywords": [
            "violence", "violent", "attack", "attacked", "attacker",
            "assault", "assaulted", "murder", "murdered", "killing",
            "killed", "slain", "stabbed", "shot", "shooting",
            "hate crime", "hate-crime", "beating", "beaten",
            "victim", "victims", "fear", "threat", "threatened",
            "harassment", "harassed", "abuse", "abused",
            "transphobic", "transphobia", "anti-trans attack",
        ],
        "description": "Coverage of violence, hate crimes, attacks, or threats against trans people",
    },
    "obituary_memorial": {
        "keywords": [
            "dies at", "dead at", "died at",
            "obituary", "obituaries", "memorial", "memorialize",
            "remembering", "remembrance", "tribute", "eulogy",
            "funeral", "passed away", "in memory",
            "transgender day of remembrance", "tdor",
            "first transgender", "trailblazer", "pioneer",
        ],
        "description": "Obituaries, memorials, and posthumous tributes",
    },
}

# Bylines of interest: reporters frequently assigned to trans coverage
# These get populated dynamically during analysis
KNOWN_REPORTERS = set()


# ---------------------------------------------------------------------------
# NYT API CLIENT
# ---------------------------------------------------------------------------

class NYTClient:
    """Rate-limited client for NYT Article Search and Archive APIs."""

    BASE_URL = "https://api.nytimes.com/svc"
    SEARCH_URL = f"{BASE_URL}/search/v2/articlesearch.json"
    ARCHIVE_URL = f"{BASE_URL}/archive/v1"
    POPULAR_URL = f"{BASE_URL}/mostpopular/v2"

    def __init__(self, api_key: str, rate_limit_delay: float = 12.0):
        self.api_key = api_key
        self.delay = rate_limit_delay  # seconds between requests
        self.last_request_time = 0
        self.request_count = 0
        self.session = requests.Session()

    def _throttle(self):
        """Enforce rate limiting: 5 requests/minute."""
        elapsed = time.time() - self.last_request_time
        if elapsed < self.delay:
            time.sleep(self.delay - elapsed)
        self.last_request_time = time.time()
        self.request_count += 1

    def _get(self, url: str, params: dict) -> dict:
        """Make a throttled GET request."""
        self._throttle()
        params["api-key"] = self.api_key
        try:
            resp = self.session.get(url, params=params, timeout=30)
            resp.raise_for_status()
            return resp.json()
        except requests.exceptions.HTTPError as e:
            if resp.status_code == 429:
                print("  Rate limited. Waiting 60 seconds...")
                time.sleep(60)
                return self._get(url, params)
            raise
        except requests.exceptions.RequestException as e:
            print(f"  Request error: {e}")
            return {}

    def search_articles(
        self,
        query: str,
        begin_date: str,
        end_date: str,
        page: int = 0,
        fq: str = None,
    ) -> dict:
        """
        Search articles via Article Search API.

        Args:
            query: Search query string
            begin_date: YYYYMMDD format
            end_date: YYYYMMDD format
            page: Results page (0-indexed, 10 results per page)
            fq: Filter query (e.g., 'section_name:("U.S." "Health")')

        Returns:
            API response dict with .response.docs[] and .response.meta
        """
        params = {
            "q": query,
            "begin_date": begin_date,
            "end_date": end_date,
            "page": page,
            "sort": "newest",
            "fl": ",".join([
                "web_url", "snippet", "lead_paragraph", "abstract",
                "headline", "keywords", "pub_date", "document_type",
                "type_of_material", "section_name", "subsection_name",
                "byline", "word_count", "print_page", "print_section",
                "source", "multimedia", "_id", "uri",
            ]),
        }
        if fq:
            params["fq"] = fq

        return self._get(self.SEARCH_URL, params)

    def get_archive(self, year: int, month: int) -> dict:
        """
        Get all article metadata for a given month via Archive API.

        Returns ~thousands of articles per month. Large JSON response.
        """
        url = f"{self.ARCHIVE_URL}/{year}/{month}.json"
        return self._get(url, {})

    def get_most_popular(
        self, period: int = 7, share_type: str = "viewed"
    ) -> dict:
        """
        Get most popular articles.

        Args:
            period: 1, 7, or 30 days
            share_type: 'emailed', 'shared', or 'viewed'
        """
        url = f"{self.POPULAR_URL}/{share_type}/{period}.json"
        return self._get(url, {})


# ---------------------------------------------------------------------------
# DATA COLLECTION
# ---------------------------------------------------------------------------

def collect_articles(
    client: NYTClient,
    start_date: str,
    end_date: str,
    output_path: str = "collected_articles.json",
    method: str = "search",
) -> list:
    """
    Collect all trans-related articles within date range.

    Two methods:
    - 'search': Use Article Search API with targeted queries (precise, slower)
    - 'archive': Use Archive API month-by-month, filter locally (comprehensive, faster per article but large downloads)
    """
    articles = {}  # dedupe by URI

    if method == "search":
        articles = _collect_via_search(client, start_date, end_date)
    elif method == "archive":
        articles = _collect_via_archive(
            client, start_date, end_date, checkpoint_path=output_path
        )
    else:
        raise ValueError(f"Unknown method: {method}")

    article_list = list(articles.values())
    print(f"\nTotal unique articles collected: {len(article_list)}")

    # Save to disk
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "metadata": {
                    "collected_at": datetime.now().isoformat(),
                    "start_date": start_date,
                    "end_date": end_date,
                    "method": method,
                    "total_articles": len(article_list),
                    "search_queries": SEARCH_QUERIES,
                },
                "articles": article_list,
            },
            f,
            indent=2,
            default=str,
            ensure_ascii=False,
        )
    print(f"Saved to {output_path}")
    return article_list


def _collect_via_search(
    client: NYTClient, start_date: str, end_date: str
) -> dict:
    """Collect articles using Article Search API with multiple queries."""
    articles = {}
    begin = start_date.replace("-", "")
    end = end_date.replace("-", "")

    for query in SEARCH_QUERIES:
        print(f"\nSearching: {query}")
        page = 0
        total_hits = None

        while True:
            data = client.search_articles(query, begin, end, page=page)
            response = data.get("response", {})
            docs = response.get("docs", [])
            meta = response.get("meta") or {}

            if total_hits is None:
                total_hits = meta.get("hits", 0)
                print(f"  Total hits: {total_hits}")

            if not docs:
                break

            for doc in docs:
                uri = doc.get("uri") or doc.get("_id", "")
                if uri and uri not in articles:
                    articles[uri] = _normalize_article(doc)

            page += 1
            if page >= 100:  # API limit: 1000 results per query
                print(f"  Reached page limit (100). {len(docs)} articles on last page.")
                break

            print(f"  Page {page}, collected {len(articles)} unique articles so far")

    return articles


def _collect_via_archive(
    client: NYTClient, start_date: str, end_date: str,
    checkpoint_path: str = None,
) -> dict:
    """Collect articles using Archive API, filtering locally for trans content.

    If checkpoint_path is given, writes the partial article list after each
    month so a crash mid-collection doesn't lose progress.
    """
    articles = {}

    start = datetime.strptime(start_date, "%Y-%m-%d")
    end = datetime.strptime(end_date, "%Y-%m-%d")

    # Build trans keyword pattern for local filtering
    trans_patterns = [
        r"\btransgender\b", r"\btrans\s+(woman|man|women|men|youth|kid|child|people|person|athlete)\b",
        r"\bgender[- ]affirming\b", r"\bgender identity\b", r"\bgender dysphoria\b",
        r"\bpuberty blocker\b", r"\bdetransition\b", r"\bnonbinary\b",
        r"\btrans rights\b", r"\bgender clinic\b", r"\bWPATH\b",
        r"\bgender transition\b", r"\bsex reassignment\b",
        r"\btranssexual\b", r"\bcisgender\b",
    ]
    pattern = re.compile("|".join(trans_patterns), re.IGNORECASE)

    current = start.replace(day=1)
    while current <= end:
        print(f"\nFetching archive: {current.year}-{current.month:02d}")
        data = client.get_archive(current.year, current.month)
        response = data.get("response") or {}
        docs = response.get("docs") or []
        print(f"  Total articles in month: {len(docs)}")

        matched = 0
        skipped = 0
        for doc in docs:
            try:
                # NYT archive sometimes returns None for keys that exist —
                # `dict.get(k, default)` only uses the default when the key is
                # missing, not when it exists with value None. Use `or` to
                # coerce None → safe empty value.
                kw_values = " ".join(
                    (kw.get("value") or "") for kw in (doc.get("keywords") or [])
                )
                searchable = " ".join(filter(None, [
                    _get_headline_text(doc),
                    doc.get("abstract") or "",
                    doc.get("lead_paragraph") or "",
                    doc.get("snippet") or "",
                    kw_values,
                ]))

                if pattern.search(searchable):
                    uri = doc.get("uri") or doc.get("_id") or ""
                    if uri and uri not in articles:
                        articles[uri] = _normalize_article(doc)
                        matched += 1
            except Exception as e:
                skipped += 1
                continue

        print(f"  Trans-related articles found: {matched}")
        if skipped:
            print(f"  Skipped {skipped} malformed docs")
        print(f"  Running total: {len(articles)}")

        # Save partial progress after each month
        if checkpoint_path:
            try:
                _save_checkpoint(checkpoint_path, list(articles.values()),
                                 start_date, end_date, "archive")
            except Exception as e:
                print(f"  (warn) checkpoint save failed: {e}")

        # Advance to next month
        if current.month == 12:
            current = current.replace(year=current.year + 1, month=1)
        else:
            current = current.replace(month=current.month + 1)

    return articles


def _save_checkpoint(path: str, article_list: list, start_date: str,
                     end_date: str, method: str):
    """Write the in-progress article list to the same JSON path collect_articles uses."""
    with open(path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "metadata": {
                    "collected_at": datetime.now().isoformat(),
                    "start_date": start_date,
                    "end_date": end_date,
                    "method": method,
                    "total_articles": len(article_list),
                    "search_queries": SEARCH_QUERIES,
                    "checkpoint": True,
                },
                "articles": article_list,
            },
            f, indent=2, default=str, ensure_ascii=False,
        )


def _normalize_article(doc: dict) -> dict:
    """Normalize an article document into a consistent schema."""
    headline = _get_headline_text(doc)
    byline = _get_byline_text(doc)

    return {
        "uri": doc.get("uri") or doc.get("_id") or "",
        "url": doc.get("web_url") or "",
        "headline": headline,
        "abstract": doc.get("abstract") or "",
        "snippet": doc.get("snippet") or "",
        "lead_paragraph": doc.get("lead_paragraph") or "",
        "pub_date": doc.get("pub_date") or "",
        "section": doc.get("section_name") or "",
        "subsection": doc.get("subsection_name") or "",
        "document_type": doc.get("document_type") or "",
        "type_of_material": doc.get("type_of_material") or "",
        "byline": byline,
        "word_count": doc.get("word_count") or 0,
        "print_page": doc.get("print_page") or "",
        "print_section": doc.get("print_section") or "",
        "source": doc.get("source") or "",
        "keywords": [
            {"name": kw.get("name") or "", "value": kw.get("value") or ""}
            for kw in (doc.get("keywords") or [])
        ],
        "multimedia_count": len(doc.get("multimedia") or []),
    }


def _get_headline_text(doc: dict) -> str:
    """Extract headline text from various formats."""
    headline = doc.get("headline") or {}
    if isinstance(headline, dict):
        return (headline.get("main") or headline.get("print_headline") or "").strip()
    return str(headline).strip()


def _get_byline_text(doc: dict) -> str:
    """Extract byline text from various formats. Returns '' if nothing usable."""
    byline = doc.get("byline") or {}
    if isinstance(byline, dict):
        original = (byline.get("original") or "").strip()
        if original:
            return original
        persons = byline.get("person") or []
        names = []
        for p in persons:
            if not isinstance(p, dict):
                continue
            parts = [(p.get(k) or "").strip()
                     for k in ("firstname", "middlename", "lastname")]
            full = " ".join(part for part in parts if part).strip()
            if full:
                names.append(full)
        if names:
            return "By " + ", ".join(names)
        # Dict had no extractable name — return empty, not the str(dict) repr
        return ""
    if isinstance(byline, str):
        return byline.strip()
    return ""


# ---------------------------------------------------------------------------
# ANALYSIS
# ---------------------------------------------------------------------------

def analyze_articles(articles: list, output_dir: str = "analysis_output") -> dict:
    """Run full analysis pipeline on collected articles."""
    os.makedirs(output_dir, exist_ok=True)

    # Classify each article
    for article in articles:
        article["framing"] = classify_framing(article)
        article["placement_tier"] = classify_placement(article)
        article["focus_tier"] = classify_focus(article)

    results = {
        "summary": generate_summary(articles),
        "framing_analysis": analyze_framing(articles),
        "placement_analysis": analyze_placement(articles),
        "byline_analysis": analyze_bylines(articles),
        "section_analysis": analyze_sections(articles),
        "temporal_analysis": analyze_temporal(articles),
        "word_count_analysis": analyze_word_counts(articles),
        "cross_tabulation": cross_tabulate(articles),
        "focus_analysis": analyze_focus(articles),
        "era_comparison": analyze_by_era(articles),
    }

    # Save full results
    output_path = os.path.join(output_dir, "analysis_results.json")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, default=str, ensure_ascii=False)
    print(f"\nAnalysis saved to {output_path}")

    # Save classified articles
    classified_path = os.path.join(output_dir, "classified_articles.json")
    with open(classified_path, "w", encoding="utf-8") as f:
        json.dump(articles, f, indent=2, default=str, ensure_ascii=False)
    print(f"Classified articles saved to {classified_path}")

    # Print summary to console
    print_analysis_summary(results)

    return results


def classify_framing(article: dict) -> list:
    """
    Classify article framing based on headline + abstract keyword matching.

    Returns list of (category, score) tuples, sorted by score descending.
    A single article can match multiple framings.
    """
    text = " ".join(filter(None, [
        article.get("headline", ""),
        article.get("abstract", ""),
        article.get("snippet", ""),
    ])).lower()

    scores = {}
    for category, config in FRAMING_CATEGORIES.items():
        score = 0
        for keyword in config["keywords"]:
            if keyword.lower() in text:
                score += 1
        if score > 0:
            scores[category] = score

    if not scores:
        return [("unclassified", 0)]

    sorted_scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return sorted_scores


# Compiled once: trans-term regex used for focus classification.
# Same vocabulary as the archive collector, but kept separate so we can tune
# focus detection without affecting collection.
_FOCUS_TRANS_RE = re.compile(
    r"\b(transgender|transsexual|cisgender|nonbinary|detransition"
    r"|trans\s+(?:woman|man|women|men|youth|kid|kids|child|children|people|person|athlete|teen|teens|community)"
    r"|gender[- ]affirming|gender identity|gender dysphoria|gender transition|gender clinic"
    r"|puberty blocker[s]?|sex reassignment|wpath)\b",
    re.IGNORECASE,
)

# NYT subject keyword fragments that strongly indicate a trans-focused article.
# NYT uses tags like "Transgender and Transsexuals", "Gender Identity",
# "Gender-Affirming Care", etc. Matching as substrings against the lowercased
# value catches variants without needing the exact full string.
_FOCUS_SUBJECT_TAGS = (
    "transgender", "transsexual", "gender identity",
    "gender dysphoria", "gender-affirming", "gender affirming",
)


def classify_focus(article: dict) -> str:
    """Classify how central trans coverage is to this article.

    Returns one of: 'focus', 'substantial', 'passing'.

    Signals (strongest → weakest):
      - NYT subject keyword explicitly tags trans topic → focus
      - Trans term in headline → focus
      - Trans term in lead paragraph or abstract → substantial
      - Trans term only in snippet or non-subject keyword → passing

    The article was already collected by trans-pattern matching, so we
    assume at least one of these signals is present somewhere.
    """
    headline = article.get("headline") or ""
    abstract = article.get("abstract") or ""
    lead = article.get("lead_paragraph") or ""
    snippet = article.get("snippet") or ""
    keywords = article.get("keywords") or []

    # Strong: explicit subject tag
    has_subject_tag = False
    for kw in keywords:
        if not isinstance(kw, dict):
            continue
        if (kw.get("name") or "").lower() != "subject":
            continue
        value_lower = (kw.get("value") or "").lower()
        if any(tag in value_lower for tag in _FOCUS_SUBJECT_TAGS):
            has_subject_tag = True
            break

    if has_subject_tag or _FOCUS_TRANS_RE.search(headline):
        return "focus"
    if _FOCUS_TRANS_RE.search(lead) or _FOCUS_TRANS_RE.search(abstract):
        return "substantial"
    return "passing"


def classify_placement(article: dict) -> str:
    """
    Classify editorial placement into tiers.

    Tier 1: Front page (print_page == '1' or print_section == 'A' page 1)
    Tier 2: Section front (first few pages of a section)
    Tier 3: Interior pages
    Tier 4: Online only / no print placement data
    """
    page = article.get("print_page", "")
    section = article.get("print_section", "")

    if not page and not section:
        return "online_only"

    try:
        page_num = int(page)
    except (ValueError, TypeError):
        return "unknown"

    if page_num == 1 and section in ("A", ""):
        return "front_page"
    elif page_num == 1:
        return "section_front"
    elif page_num <= 3:
        return "prominent"
    else:
        return "interior"


def analyze_framing(articles: list) -> dict:
    """Analyze framing distribution across articles."""
    primary_framing = Counter()
    all_framings = Counter()
    framing_over_time = defaultdict(lambda: Counter())

    for article in articles:
        framings = article.get("framing", [("unclassified", 0)])
        primary = framings[0][0]
        primary_framing[primary] += 1

        for category, score in framings:
            all_framings[category] += 1

        # Monthly aggregation
        pub_date = article.get("pub_date", "")
        if pub_date:
            month = pub_date[:7]  # YYYY-MM
            framing_over_time[month][primary] += 1

    return {
        "primary_framing_counts": dict(primary_framing.most_common()),
        "all_framing_mentions": dict(all_framings.most_common()),
        "monthly_framing": {
            month: dict(counts)
            for month, counts in sorted(framing_over_time.items())
        },
    }


def analyze_placement(articles: list) -> dict:
    """Analyze placement hierarchy and its correlation with framing."""
    placement_counts = Counter()
    placement_by_framing = defaultdict(Counter)

    for article in articles:
        tier = article.get("placement_tier", "unknown")
        placement_counts[tier] += 1

        primary_framing = article.get("framing", [("unclassified", 0)])[0][0]
        placement_by_framing[primary_framing][tier] += 1

    # Calculate front-page rate by framing type
    front_page_rates = {}
    for framing, placements in placement_by_framing.items():
        total = sum(placements.values())
        front = placements.get("front_page", 0)
        prominent = placements.get("prominent", 0) + placements.get("section_front", 0)
        front_page_rates[framing] = {
            "total_articles": total,
            "front_page": front,
            "front_page_rate": round(front / total, 4) if total else 0,
            "prominent_placement": front + prominent,
            "prominent_rate": round((front + prominent) / total, 4) if total else 0,
        }

    return {
        "overall_placement": dict(placement_counts.most_common()),
        "placement_by_framing": {
            k: dict(v) for k, v in placement_by_framing.items()
        },
        "front_page_rates_by_framing": front_page_rates,
    }


def analyze_bylines(articles: list) -> dict:
    """Analyze byline concentration in trans coverage."""
    byline_counts = Counter()
    byline_framings = defaultdict(Counter)
    byline_placements = defaultdict(Counter)

    for article in articles:
        byline = (article.get("byline") or "").strip()
        # Older runs stored str(dict) when the API returned an empty byline
        # struct — strip those out so they don't show up as a "reporter".
        if byline.startswith(("{'", '{"')) and byline.endswith("}"):
            byline = ""
        if not byline:
            byline = "No byline"

        # Normalize: remove "By " prefix
        normalized = re.sub(r"^By\s+", "", byline, flags=re.IGNORECASE).strip()
        if not normalized:
            normalized = "No byline"

        byline_counts[normalized] += 1

        primary_framing = article.get("framing", [("unclassified", 0)])[0][0]
        byline_framings[normalized][primary_framing] += 1

        tier = article.get("placement_tier", "unknown")
        byline_placements[normalized][tier] += 1

    # Top reporters
    top_reporters = byline_counts.most_common(20)

    # Reporter framing profiles
    reporter_profiles = {}
    for reporter, count in top_reporters:
        if count >= 3:  # minimum threshold
            reporter_profiles[reporter] = {
                "article_count": count,
                "framing_distribution": dict(byline_framings[reporter]),
                "placement_distribution": dict(byline_placements[reporter]),
            }

    return {
        "top_reporters": top_reporters,
        "reporter_profiles": reporter_profiles,
        "total_unique_bylines": len(byline_counts),
        "concentration": {
            "top_5_share": round(
                sum(c for _, c in byline_counts.most_common(5))
                / max(len(articles), 1),
                4,
            ),
            "top_10_share": round(
                sum(c for _, c in byline_counts.most_common(10))
                / max(len(articles), 1),
                4,
            ),
        },
    }


def analyze_sections(articles: list) -> dict:
    """Analyze section assignment patterns."""
    section_counts = Counter()
    section_by_framing = defaultdict(Counter)

    for article in articles:
        section = article.get("section", "") or "Unknown"
        section_counts[section] += 1

        primary_framing = article.get("framing", [("unclassified", 0)])[0][0]
        section_by_framing[section][primary_framing] += 1

    return {
        "section_distribution": dict(section_counts.most_common()),
        "section_by_framing": {
            k: dict(v) for k, v in section_by_framing.items()
        },
    }


def analyze_temporal(articles: list) -> dict:
    """Analyze coverage volume and patterns over time."""
    monthly_counts = Counter()
    weekly_counts = Counter()
    daily_counts = Counter()
    material_type_over_time = defaultdict(Counter)

    for article in articles:
        pub_date = article.get("pub_date", "")
        if not pub_date:
            continue

        month = pub_date[:7]
        monthly_counts[month] += 1

        # Parse for day-of-week
        try:
            dt = datetime.fromisoformat(pub_date.replace("Z", "+00:00"))
            day_name = dt.strftime("%A")
            daily_counts[day_name] += 1

            # ISO week
            week = dt.strftime("%Y-W%W")
            weekly_counts[week] += 1
        except (ValueError, TypeError):
            pass

        mat_type = article.get("type_of_material", "") or "Unknown"
        material_type_over_time[month][mat_type] += 1

    return {
        "monthly_volume": dict(sorted(monthly_counts.items())),
        "day_of_week_distribution": dict(daily_counts.most_common()),
        "material_type_over_time": {
            month: dict(counts)
            for month, counts in sorted(material_type_over_time.items())
        },
    }


def analyze_word_counts(articles: list) -> dict:
    """Analyze word count distribution by framing and placement."""
    wc_by_framing = defaultdict(list)
    wc_by_placement = defaultdict(list)

    for article in articles:
        wc = article.get("word_count", 0)
        if not wc:
            continue
        try:
            wc = int(wc)
        except (ValueError, TypeError):
            continue

        primary_framing = article.get("framing", [("unclassified", 0)])[0][0]
        wc_by_framing[primary_framing].append(wc)

        tier = article.get("placement_tier", "unknown")
        wc_by_placement[tier].append(wc)

    def stats(values):
        if not values:
            return {"count": 0, "mean": 0, "median": 0, "min": 0, "max": 0}
        sorted_v = sorted(values)
        n = len(sorted_v)
        return {
            "count": n,
            "mean": round(sum(sorted_v) / n),
            "median": sorted_v[n // 2],
            "min": sorted_v[0],
            "max": sorted_v[-1],
        }

    return {
        "by_framing": {k: stats(v) for k, v in wc_by_framing.items()},
        "by_placement": {k: stats(v) for k, v in wc_by_placement.items()},
    }


def cross_tabulate(articles: list) -> dict:
    """
    Cross-tabulate framing x placement to identify the core pattern:
    do skepticism/culture-war articles get systematically higher placement
    than rights-affirming or human-interest stories?
    """
    cross_tab = defaultdict(lambda: defaultdict(int))
    for article in articles:
        primary_framing = article.get("framing", [("unclassified", 0)])[0][0]
        tier = article.get("placement_tier", "unknown")
        cross_tab[primary_framing][tier] += 1

    # Calculate a "prominence score" for each framing category
    # Front page = 4, section front = 3, prominent = 2, interior = 1, online = 0
    tier_weights = {
        "front_page": 4,
        "section_front": 3,
        "prominent": 2,
        "interior": 1,
        "online_only": 0,
        "unknown": 0,
    }

    prominence_scores = {}
    for framing, placements in cross_tab.items():
        total = sum(placements.values())
        if total == 0:
            continue
        weighted = sum(
            tier_weights.get(tier, 0) * count
            for tier, count in placements.items()
        )
        prominence_scores[framing] = {
            "mean_prominence": round(weighted / total, 3),
            "total_articles": total,
            "distribution": dict(placements),
        }

    return {
        "cross_tab": {k: dict(v) for k, v in cross_tab.items()},
        "prominence_scores": prominence_scores,
    }


def analyze_focus(articles: list) -> dict:
    """Distribution and cross-tabs for focus tier (focus / substantial / passing)."""
    tier_counts = Counter()
    framing_by_tier = defaultdict(Counter)
    placement_by_tier = defaultdict(Counter)
    section_by_tier = defaultdict(Counter)
    wc_by_tier = defaultdict(list)
    monthly_by_tier = defaultdict(lambda: Counter())

    for article in articles:
        tier = article.get("focus_tier") or "passing"
        tier_counts[tier] += 1

        primary_framing = article.get("framing", [("unclassified", 0)])[0][0]
        framing_by_tier[tier][primary_framing] += 1

        placement_by_tier[tier][article.get("placement_tier") or "unknown"] += 1
        section_by_tier[tier][article.get("section") or "Unknown"] += 1

        try:
            wc = int(article.get("word_count") or 0)
            if wc:
                wc_by_tier[tier].append(wc)
        except (ValueError, TypeError):
            pass

        pub_date = article.get("pub_date") or ""
        if pub_date:
            monthly_by_tier[pub_date[:7]][tier] += 1

    def stats(values):
        if not values:
            return {"count": 0, "mean": 0, "median": 0}
        s = sorted(values)
        return {
            "count": len(s),
            "mean": round(sum(s) / len(s)),
            "median": s[len(s) // 2],
        }

    return {
        "tier_counts": dict(tier_counts.most_common()),
        "framing_by_tier": {k: dict(v) for k, v in framing_by_tier.items()},
        "placement_by_tier": {k: dict(v) for k, v in placement_by_tier.items()},
        "section_by_tier": {
            k: dict(v.most_common(10)) for k, v in section_by_tier.items()
        },
        "word_count_by_tier": {k: stats(v) for k, v in wc_by_tier.items()},
        "monthly_by_tier": {
            month: dict(counts) for month, counts in sorted(monthly_by_tier.items())
        },
    }


# Three-bucket era split: early Obama/Trump-1 (pre-state-ban wave),
# late Trump-1 / Biden, then post-2022 state-bill / SCOTUS / Cass-Review era.
# Edit this list to change boundaries; everything downstream reads from it.
# `end` is exclusive; None = open-ended.
ERAS = [
    {"key": "era_2014_2017", "label": "2014–2017", "start": "2014-01-01", "end": "2018-01-01"},
    {"key": "era_2018_2021", "label": "2018–2021", "start": "2018-01-01", "end": "2022-01-01"},
    {"key": "era_2022_plus", "label": "2022–present", "start": "2022-01-01", "end": None},
]


def _era_for(article: dict) -> str:
    """Return the era key an article falls into, or None if outside any era."""
    pub = (article.get("pub_date") or "")[:10]
    if not pub:
        return None
    for era in ERAS:
        if pub < era["start"]:
            continue
        if era["end"] is None or pub < era["end"]:
            return era["key"]
    return None


def _era_metrics(articles: list) -> dict:
    """Compute the comparison-relevant metrics for one era's article subset."""
    if not articles:
        return {
            "count": 0, "framing_counts": {}, "framing_pct": {},
            "focus_counts": {}, "focus_pct": {},
            "section_top": [], "reporter_top": [],
            "placement_counts": {}, "word_count_mean": 0,
            "monthly_volume": {}, "front_page_rate": 0,
        }

    framing_counts = Counter()
    focus_counts = Counter()
    section_counts = Counter()
    reporter_counts = Counter()
    placement_counts = Counter()
    monthly = Counter()
    word_counts = []
    front_page = 0
    placement_total = 0

    for a in articles:
        primary = a.get("framing", [("unclassified", 0)])[0][0]
        framing_counts[primary] += 1
        focus_counts[a.get("focus_tier") or "passing"] += 1
        section_counts[a.get("section") or "Unknown"] += 1

        # Skip blank bylines and the cleaned-up dict-as-string case
        byline = (a.get("byline") or "").strip()
        if byline.startswith(("{'", '{"')) and byline.endswith("}"):
            byline = ""
        if byline:
            normalized = re.sub(r"^By\s+", "", byline, flags=re.IGNORECASE).strip()
            if normalized:
                reporter_counts[normalized] += 1

        tier = a.get("placement_tier") or "unknown"
        placement_counts[tier] += 1
        if tier == "front_page":
            front_page += 1
        placement_total += 1

        try:
            wc = int(a.get("word_count") or 0)
            if wc:
                word_counts.append(wc)
        except (ValueError, TypeError):
            pass

        pub = (a.get("pub_date") or "")[:7]
        if pub:
            monthly[pub] += 1

    total = len(articles)

    def pct(counter):
        return {k: round(100 * v / total, 2) for k, v in counter.items()}

    return {
        "count": total,
        "framing_counts": dict(framing_counts),
        "framing_pct": pct(framing_counts),
        "focus_counts": dict(focus_counts),
        "focus_pct": pct(focus_counts),
        "section_top": section_counts.most_common(10),
        "reporter_top": reporter_counts.most_common(10),
        "placement_counts": dict(placement_counts),
        "word_count_mean": round(sum(word_counts) / len(word_counts)) if word_counts else 0,
        "word_count_median": sorted(word_counts)[len(word_counts) // 2] if word_counts else 0,
        "monthly_volume": dict(sorted(monthly.items())),
        "front_page_rate": round(front_page / placement_total, 4) if placement_total else 0,
    }


def analyze_by_era(articles: list) -> dict:
    """Bucket articles by ERAS and compute metrics for each. Result keys mirror
    the era keys in ERAS (so the dashboard can iterate them dynamically).
    `eras` is the ordered metadata list so the UI knows the display order."""
    buckets = {era["key"]: [] for era in ERAS}
    for a in articles:
        key = _era_for(a)
        if key in buckets:
            buckets[key].append(a)

    result = {
        "eras": [{"key": e["key"], "label": e["label"],
                  "start": e["start"], "end": e["end"]} for e in ERAS],
        "labels": {e["key"]: e["label"] for e in ERAS},
    }
    for era in ERAS:
        result[era["key"]] = _era_metrics(buckets[era["key"]])
    return result


def generate_summary(articles: list) -> dict:
    """Generate high-level summary statistics."""
    date_range = [a.get("pub_date", "")[:10] for a in articles if a.get("pub_date")]
    date_range = [d for d in date_range if d]

    material_types = Counter(
        a.get("type_of_material", "Unknown") or "Unknown" for a in articles
    )

    return {
        "total_articles": len(articles),
        "date_range": {
            "earliest": min(date_range) if date_range else None,
            "latest": max(date_range) if date_range else None,
        },
        "material_types": dict(material_types.most_common()),
    }


# ---------------------------------------------------------------------------
# OUTPUT
# ---------------------------------------------------------------------------

def print_analysis_summary(results: dict):
    """Print a readable summary to console."""
    summary = results["summary"]
    print("\n" + "=" * 70)
    print("NYT TRANS COVERAGE ANALYSIS")
    print("=" * 70)
    print(f"Total articles: {summary['total_articles']}")
    print(f"Date range: {summary['date_range']['earliest']} to {summary['date_range']['latest']}")

    focus = results.get("focus_analysis", {}).get("tier_counts", {})
    if focus:
        print("\n--- FOCUS DISTRIBUTION ---")
        ftotal = sum(focus.values()) or 1
        for tier in ("focus", "substantial", "passing"):
            count = focus.get(tier, 0)
            pct = round(100 * count / ftotal, 1)
            bar = "█" * int(pct / 2)
            print(f"  {tier:15s} {count:5d} ({pct:5.1f}%) {bar}")

    print("\n--- FRAMING DISTRIBUTION ---")
    framing = results["framing_analysis"]["primary_framing_counts"]
    total = sum(framing.values())
    for category, count in sorted(framing.items(), key=lambda x: x[1], reverse=True):
        pct = round(100 * count / total, 1)
        bar = "█" * int(pct / 2)
        print(f"  {category:25s} {count:4d} ({pct:5.1f}%) {bar}")

    print("\n--- PLACEMENT HIERARCHY ---")
    placement = results["placement_analysis"]["overall_placement"]
    for tier, count in sorted(placement.items(), key=lambda x: x[1], reverse=True):
        print(f"  {tier:20s} {count:4d}")

    print("\n--- PROMINENCE BY FRAMING ---")
    prominence = results["cross_tabulation"]["prominence_scores"]
    for framing, data in sorted(
        prominence.items(), key=lambda x: x[1]["mean_prominence"], reverse=True
    ):
        score = data["mean_prominence"]
        n = data["total_articles"]
        bar = "█" * int(score * 5)
        print(f"  {framing:25s} score={score:.2f}  n={n:3d}  {bar}")

    print("\n--- TOP REPORTERS ---")
    bylines = results["byline_analysis"]["top_reporters"][:10]
    for reporter, count in bylines:
        print(f"  {reporter:40s} {count:3d} articles")

    concentration = results["byline_analysis"]["concentration"]
    print(f"\n  Top 5 reporters wrote {concentration['top_5_share']:.1%} of all coverage")
    print(f"  Top 10 reporters wrote {concentration['top_10_share']:.1%} of all coverage")

    print("\n--- SECTION DISTRIBUTION ---")
    sections = results["section_analysis"]["section_distribution"]
    for section, count in list(sorted(sections.items(), key=lambda x: x[1], reverse=True))[:10]:
        print(f"  {section:25s} {count:4d}")

    print("\n--- WORD COUNT BY FRAMING ---")
    wc = results["word_count_analysis"]["by_framing"]
    for framing, stats in sorted(wc.items(), key=lambda x: x[1]["mean"], reverse=True):
        print(f"  {framing:25s} mean={stats['mean']:5d}  median={stats['median']:5d}  n={stats['count']:3d}")

    print("\n" + "=" * 70)


# ---------------------------------------------------------------------------
# DASHBOARD DATA EXPORT
# ---------------------------------------------------------------------------

def export_dashboard_data(results: dict, articles: list, output_dir: str = "analysis_output"):
    """Export analysis data in a format optimized for the React dashboard.

    LLM classifier outputs are the primary data source.  Legacy keyword-based
    keys are preserved under ``legacy_*`` prefixes for backward compatibility.
    """
    from collections import defaultdict

    # ------------------------------------------------------------------
    # Normalize llm_consensus → llm_scores so every downstream view that
    # already reads llm_scores keeps working. We do this NON-DESTRUCTIVELY
    # by promoting consensus_scores into the same article's llm_scores field
    # only when llm_scores is missing or stale. The original llm_consensus
    # record is preserved for the Consensus tab and per-model breakdowns.
    # ------------------------------------------------------------------
    for a in articles:
        cons = a.get("llm_consensus") or {}
        if not cons or cons.get("error"):
            continue
        scores = cons.get("consensus_scores")
        if not scores:
            continue
        # Build an llm_scores-shaped view from the consensus
        a["llm_scores"] = {
            **scores,
            "model": "consensus(" + ",".join(cons.get("models_used") or []) + ")",
            "error": None,
            "had_body": cons.get("had_body", False),
            "prompt_version": cons.get("prompt_version", 0),
            # Pass through certainty so downstream views can weight by it
            "consensus_certainty": cons.get("consensus_certainty"),
        }

    # ------------------------------------------------------------------
    # Helper: collect articles that have valid (non-error) llm_scores
    # ------------------------------------------------------------------
    classified = [
        a for a in articles
        if a.get("llm_scores") and not a["llm_scores"].get("error")
    ]

    def _monthly_key(article):
        return (article.get("pub_date") or "")[:7]  # "YYYY-MM"

    def _safe_mean(values):
        if not values:
            return 0.0
        return round(sum(values) / len(values), 4)

    # ------------------------------------------------------------------
    # 1. summary (kept as-is)
    # ------------------------------------------------------------------
    dashboard_data = {
        "summary": results["summary"],
    }

    # ------------------------------------------------------------------
    # 2. llm_matrix (uses llm_classify.summary_matrix)
    # ------------------------------------------------------------------
    try:
        import llm_classify  # type: ignore
        if classified:
            dashboard_data["llm_matrix"] = llm_classify.summary_matrix(articles)
        else:
            dashboard_data["llm_matrix"] = {}
    except ImportError:
        dashboard_data["llm_matrix"] = {}

    # ------------------------------------------------------------------
    # 2b. consensus_stats — corpus-level agreement metrics (only populated
    # when 3-model ensemble has been run on at least some articles)
    # ------------------------------------------------------------------
    consensus_articles = [
        a for a in articles
        if a.get("llm_consensus") and not a["llm_consensus"].get("error")
        and a["llm_consensus"].get("consensus_scores")
    ]
    if consensus_articles:
        certainties = [
            a["llm_consensus"].get("consensus_certainty", 0)
            for a in consensus_articles
        ]
        # Histogram bins for certainty distribution
        bins = {"high (>0.85)": 0, "medium (0.6-0.85)": 0,
                "low (0.3-0.6)": 0, "very_low (<0.3)": 0}
        for c in certainties:
            if c is None:
                continue
            if c > 0.85:
                bins["high (>0.85)"] += 1
            elif c > 0.6:
                bins["medium (0.6-0.85)"] += 1
            elif c > 0.3:
                bins["low (0.3-0.6)"] += 1
            else:
                bins["very_low (<0.3)"] += 1

        # Provider coverage — how many articles each model scored
        from collections import Counter
        provider_counts = Counter()
        for a in consensus_articles:
            for p in (a["llm_consensus"].get("models_used") or []):
                provider_counts[p] += 1

        # Full-success rate: an article is "full" when it has a score from
        # every provider that appears anywhere in the corpus. This adapts
        # automatically as models are added (e.g. the local Qwen 4th model) and
        # correctly surfaces articles a provider refused (partial coverage).
        n_providers = len(provider_counts)
        all_providers = set(provider_counts.keys())
        full_success = sum(
            1 for a in consensus_articles
            if all_providers.issubset(set(a["llm_consensus"].get("models_used") or []))
        )

        dashboard_data["consensus_stats"] = {
            "n": len(consensus_articles),
            "n_providers": n_providers,
            "mean_certainty": round(sum(certainties) / len(certainties), 3),
            "median_certainty": round(sorted(certainties)[len(certainties) // 2], 3),
            "certainty_bins": bins,
            "provider_counts": dict(provider_counts),
            "full_coverage_count": full_success,
            # Kept for backward-compat with older dashboard builds.
            "full_3model_count": full_success,
            "partial_count": len(consensus_articles) - full_success,
        }

        # Also use the consensus-aware matrix (certainty-weighted averages)
        try:
            import llm_consensus
            dashboard_data["llm_matrix"] = llm_consensus.summary_matrix(
                articles, weight_by_certainty=True)
        except ImportError:
            pass  # Stick with single-model matrix
    else:
        dashboard_data["consensus_stats"] = {}

    # ------------------------------------------------------------------
    # 3. llm_direction_timeseries - monthly averages of framing_direction
    # ------------------------------------------------------------------
    direction_dims = ["skepticism_to_affirming", "restrictive_to_protective"]
    direction_by_month = defaultdict(lambda: defaultdict(list))
    for a in classified:
        month = _monthly_key(a)
        if not month:
            continue
        fd = a["llm_scores"].get("framing_direction", {})
        for dim in direction_dims:
            val = fd.get(dim)
            if val is not None:
                direction_by_month[month][dim].append(val)

    dashboard_data["llm_direction_timeseries"] = [
        {"month": m, **{dim: _safe_mean(direction_by_month[m].get(dim, [])) for dim in direction_dims}}
        for m in sorted(direction_by_month.keys())
    ]

    # ------------------------------------------------------------------
    # 4. llm_tone_timeseries - monthly averages of framing_tone
    # ------------------------------------------------------------------
    tone_dims = ["conflict_framing", "pathologizing", "both_sides_treatment", "alarmist_tone", "normalizing"]
    tone_by_month = defaultdict(lambda: defaultdict(list))
    for a in classified:
        month = _monthly_key(a)
        if not month:
            continue
        ft = a["llm_scores"].get("framing_tone", {})
        for dim in tone_dims:
            val = ft.get(dim)
            if val is not None:
                tone_by_month[month][dim].append(val)

    dashboard_data["llm_tone_timeseries"] = [
        {"month": m, **{dim: _safe_mean(tone_by_month[m].get(dim, [])) for dim in tone_dims}}
        for m in sorted(tone_by_month.keys())
    ]

    # ------------------------------------------------------------------
    # 5. llm_voice_timeseries - monthly averages of voice dimensions
    # ------------------------------------------------------------------
    voice_dims = ["trans_voice_centrality", "expert_voice_centrality",
                  "advocate_voice_centrality", "opponent_voice_centrality"]
    voice_by_month = defaultdict(lambda: defaultdict(list))
    for a in classified:
        month = _monthly_key(a)
        if not month:
            continue
        v = a["llm_scores"].get("voice", {})
        for dim in voice_dims:
            val = v.get(dim)
            if val is not None:
                voice_by_month[month][dim].append(val)

    dashboard_data["llm_voice_timeseries"] = [
        {"month": m, **{dim: _safe_mean(voice_by_month[m].get(dim, [])) for dim in voice_dims}}
        for m in sorted(voice_by_month.keys())
    ]

    # ------------------------------------------------------------------
    # 6. llm_topic_distribution - average score per topic dimension
    # ------------------------------------------------------------------
    topic_dims = ["medical", "legal", "cultural", "violence", "personal",
                  "science", "politics", "international", "youth_focus"]
    topic_accum = defaultdict(list)
    # Two count-based views readers actually expect, alongside the avg score:
    #   primary_count    — # of articles where this topic is the highest-scoring
    #                      one (mutually exclusive; sums to the article total)
    #   substantial_count — # of articles scoring >= 5 on this topic (overlapping;
    #                      an article can be "substantially about" several topics)
    primary_count = defaultdict(int)
    substantial_count = defaultdict(int)
    n_topic_articles = 0
    for a in classified:
        t = a["llm_scores"].get("topic", {})
        present = {dim: t[dim] for dim in topic_dims
                   if t.get(dim) is not None}
        for dim, val in present.items():
            topic_accum[dim].append(val)
            if val >= 5:
                substantial_count[dim] += 1
        if present:
            n_topic_articles += 1
            primary_count[max(present, key=lambda d: present[d])] += 1

    dashboard_data["llm_topic_distribution"] = [
        {"topic": dim, "avg_score": _safe_mean(topic_accum[dim]),
         "count": len(topic_accum[dim]),
         "primary_count": primary_count[dim],
         "substantial_count": substantial_count[dim]}
        for dim in topic_dims
    ]
    dashboard_data["llm_topic_total"] = n_topic_articles

    # ------------------------------------------------------------------
    # 7. llm_placement_by_direction - placement stats per direction bucket
    # ------------------------------------------------------------------
    def _direction_bucket(article):
        fd = article["llm_scores"].get("framing_direction", {})
        val = fd.get("skepticism_to_affirming")
        if val is None:
            return None
        if val < -1:
            return "skeptical"
        elif val > 1:
            return "affirming"
        else:
            return "neutral"

    placement_buckets = defaultdict(lambda: {"total": 0, "front_page": 0, "pages": []})
    for a in classified:
        bucket = _direction_bucket(a)
        if bucket is None:
            continue
        placement_buckets[bucket]["total"] += 1
        page = a.get("print_page", "")
        if page:
            placement_buckets[bucket]["pages"].append(page)
        if str(page) == "1":
            placement_buckets[bucket]["front_page"] += 1

    dashboard_data["llm_placement_by_direction"] = []
    for bucket in ("skeptical", "neutral", "affirming"):
        info = placement_buckets[bucket]
        total = info["total"]
        dashboard_data["llm_placement_by_direction"].append({
            "direction_bucket": bucket,
            "total": total,
            "front_page": info["front_page"],
            "front_page_rate": round(info["front_page"] / max(total, 1), 4),
        })

    # ------------------------------------------------------------------
    # 8. llm_word_count_by_direction - word count stats per direction bucket
    # ------------------------------------------------------------------
    wc_buckets = defaultdict(list)
    for a in classified:
        bucket = _direction_bucket(a)
        if bucket is None:
            continue
        wc = a.get("word_count", 0) or 0
        wc_buckets[bucket].append(wc)

    dashboard_data["llm_word_count_by_direction"] = []
    for bucket in ("skeptical", "neutral", "affirming"):
        vals = wc_buckets[bucket]
        if vals:
            dashboard_data["llm_word_count_by_direction"].append({
                "direction_bucket": bucket,
                "count": len(vals),
                "mean": round(sum(vals) / len(vals), 1),
                "median": round(sorted(vals)[len(vals) // 2], 1),
                "min": min(vals),
                "max": max(vals),
            })
        else:
            dashboard_data["llm_word_count_by_direction"].append({
                "direction_bucket": bucket,
                "count": 0, "mean": 0, "median": 0, "min": 0, "max": 0,
            })

    # ------------------------------------------------------------------
    # 9. llm_byline_profiles - top 15 reporters with avg LLM dimension scores
    # ------------------------------------------------------------------
    reporter_articles = defaultdict(list)
    for a in classified:
        byline = a.get("byline", "").strip()
        if byline:
            reporter_articles[byline].append(a)

    top_reporters = sorted(reporter_articles.items(), key=lambda x: -len(x[1]))[:15]
    dashboard_data["llm_byline_profiles"] = []
    for reporter, r_articles in top_reporters:
        profile = {"reporter": reporter, "count": len(r_articles)}
        # Average direction
        dir_avgs = {}
        for dim in direction_dims:
            vals = [a["llm_scores"].get("framing_direction", {}).get(dim)
                    for a in r_articles]
            vals = [v for v in vals if v is not None]
            dir_avgs[dim] = _safe_mean(vals)
        profile["framing_direction"] = dir_avgs
        # Average tone
        tone_avgs = {}
        for dim in tone_dims:
            vals = [a["llm_scores"].get("framing_tone", {}).get(dim)
                    for a in r_articles]
            vals = [v for v in vals if v is not None]
            tone_avgs[dim] = _safe_mean(vals)
        profile["framing_tone"] = tone_avgs
        # Average voice
        voice_avgs = {}
        for dim in voice_dims:
            vals = [a["llm_scores"].get("voice", {}).get(dim)
                    for a in r_articles]
            vals = [v for v in vals if v is not None]
            voice_avgs[dim] = _safe_mean(vals)
        profile["voice"] = voice_avgs
        # Average topic
        topic_avgs = {}
        for dim in topic_dims:
            vals = [a["llm_scores"].get("topic", {}).get(dim)
                    for a in r_articles]
            vals = [v for v in vals if v is not None]
            topic_avgs[dim] = _safe_mean(vals)
        profile["topic"] = topic_avgs
        dashboard_data["llm_byline_profiles"].append(profile)

    # ------------------------------------------------------------------
    # 10. prominence_scores (keyword-based, still useful)
    # ------------------------------------------------------------------
    dashboard_data["prominence_scores"] = []
    for framing, data in results["cross_tabulation"]["prominence_scores"].items():
        dashboard_data["prominence_scores"].append({
            "framing": framing,
            **data,
        })

    # ------------------------------------------------------------------
    # 11. body_coverage
    # ------------------------------------------------------------------
    bodies_with = sum(1 for a in articles if (a.get("body") or "").strip())
    dashboard_data["body_coverage"] = {
        "total": len(articles),
        "with_body": bodies_with,
        "coverage": round(bodies_with / max(len(articles), 1), 4),
    }

    # ------------------------------------------------------------------
    # 12. articles (with llm_scores included)
    # ------------------------------------------------------------------
    dashboard_data["articles"] = []
    for article in articles:
        body = article.get("body") or ""
        row = {
            "headline": article.get("headline", ""),
            "pub_date": (article.get("pub_date") or "")[:10],
            "section": article.get("section", ""),
            "byline": article.get("byline", ""),
            "word_count": article.get("word_count", 0),
            "print_page": article.get("print_page", ""),
            "framing": article.get("framing", [("unclassified", 0)])[0][0],
            "placement_tier": article.get("placement_tier", ""),
            "focus_tier": article.get("focus_tier", ""),
            "url": article.get("url", ""),
            "has_body": bool(body.strip()),
            "body_words": len(body.split()) if body else 0,
        }
        if article.get("llm_scores") and not article["llm_scores"].get("error"):
            row["llm_scores"] = article["llm_scores"]
        # Consensus-specific fields (only present when 3-model ensemble was run)
        cons = article.get("llm_consensus") or {}
        if cons and not cons.get("error"):
            row["consensus_certainty"] = cons.get("consensus_certainty")
            row["models_used"] = cons.get("models_used") or []
            row["per_model_scores"] = cons.get("per_model_scores") or {}
        dashboard_data["articles"].append(row)

    # ------------------------------------------------------------------
    # 13. Legacy keyword-based keys (prefixed with legacy_)
    # ------------------------------------------------------------------
    # legacy_framing_timeseries
    monthly = results["framing_analysis"]["monthly_framing"]
    all_framings_set = set()
    for counts in monthly.values():
        all_framings_set.update(counts.keys())
    legacy_framing_ts = []
    for month in sorted(monthly.keys()):
        row = {"month": month}
        for framing in sorted(all_framings_set):
            row[framing] = monthly[month].get(framing, 0)
        legacy_framing_ts.append(row)
    dashboard_data["legacy_framing_timeseries"] = legacy_framing_ts

    # legacy_placement_by_framing
    rates = results["placement_analysis"]["front_page_rates_by_framing"]
    dashboard_data["legacy_placement_by_framing"] = [
        {"framing": framing, **data} for framing, data in rates.items()
    ]

    # legacy_byline_data
    legacy_byline = []
    for reporter, count in results["byline_analysis"]["top_reporters"][:15]:
        profile = results["byline_analysis"]["reporter_profiles"].get(reporter, {})
        legacy_byline.append({
            "reporter": reporter,
            "count": count,
            "framings": profile.get("framing_distribution", {}),
        })
    dashboard_data["legacy_byline_data"] = legacy_byline

    # legacy_section_data
    dashboard_data["legacy_section_data"] = [
        {"section": section, "count": count}
        for section, count in results["section_analysis"]["section_distribution"].items()
    ]

    # legacy_word_count_by_framing
    dashboard_data["legacy_word_count_by_framing"] = [
        {"framing": framing, **stats}
        for framing, stats in results["word_count_analysis"]["by_framing"].items()
        if stats.get("count", 0) > 0
    ]

    # ------------------------------------------------------------------
    # 14. Focus analysis (focus / substantial / passing) - kept as-is
    # ------------------------------------------------------------------
    focus = results.get("focus_analysis", {})
    if focus:
        dashboard_data["focus_distribution"] = [
            {"tier": tier, "count": count}
            for tier, count in focus.get("tier_counts", {}).items()
        ]
        framing_keys = set()
        for fmap in focus.get("framing_by_tier", {}).values():
            framing_keys.update(fmap.keys())
        dashboard_data["focus_by_framing"] = []
        for tier in ("focus", "substantial", "passing"):
            row = {"tier": tier}
            tier_map = focus.get("framing_by_tier", {}).get(tier, {})
            for fr in framing_keys:
                row[fr] = tier_map.get(fr, 0)
            dashboard_data["focus_by_framing"].append(row)
        dashboard_data["focus_word_counts"] = [
            {"tier": tier, **stats}
            for tier, stats in focus.get("word_count_by_tier", {}).items()
            if stats.get("count", 0) > 0
        ]
        dashboard_data["focus_timeseries"] = [
            {"month": month, **counts}
            for month, counts in focus.get("monthly_by_tier", {}).items()
        ]

    # ------------------------------------------------------------------
    # 15. era_comparison - kept as-is
    # ------------------------------------------------------------------
    era = results.get("era_comparison")
    if era:
        dashboard_data["era_comparison"] = era

    # ------------------------------------------------------------------
    # Write output
    # ------------------------------------------------------------------
    output_path = os.path.join(output_dir, "dashboard_data.json")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(dashboard_data, f, indent=2, default=str, ensure_ascii=False)
    print(f"Dashboard data exported to {output_path}")

    return dashboard_data


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="NYT Trans Coverage Analysis Pipeline"
    )
    subparsers = parser.add_subparsers(dest="command", help="Command to run")

    # Collect command
    collect_parser = subparsers.add_parser("collect", help="Collect article data from NYT API")
    collect_parser.add_argument("--api-key", help="NYT Developer API key (else read from API.txt or NYT_API_KEY env)")
    collect_parser.add_argument("--start", required=True, help="Start date (YYYY-MM-DD)")
    collect_parser.add_argument("--end", required=True, help="End date (YYYY-MM-DD)")
    collect_parser.add_argument("--output", default="collected_articles.json", help="Output file path")
    collect_parser.add_argument(
        "--method", choices=["search", "archive"], default="search",
        help="Collection method: 'search' (targeted queries) or 'archive' (monthly bulk)"
    )

    # Analyze command
    analyze_parser = subparsers.add_parser("analyze", help="Analyze collected data")
    analyze_parser.add_argument("--input", required=True, help="Path to collected_articles.json")
    analyze_parser.add_argument("--output-dir", default="analysis_output", help="Output directory")

    # Full pipeline
    full_parser = subparsers.add_parser("full", help="Collect and analyze in one pass")
    full_parser.add_argument("--api-key", help="NYT Developer API key (else read from API.txt or NYT_API_KEY env)")
    full_parser.add_argument("--start", required=True, help="Start date (YYYY-MM-DD)")
    full_parser.add_argument("--end", required=True, help="End date (YYYY-MM-DD)")
    full_parser.add_argument("--output-dir", default="analysis_output", help="Output directory")
    full_parser.add_argument(
        "--method", choices=["search", "archive"], default="search",
        help="Collection method"
    )

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return

    try:
        if args.command == "collect":
            api_key = load_api_key(args.api_key)
            client = NYTClient(api_key)
            collect_articles(client, args.start, args.end, args.output, args.method)

        elif args.command == "analyze":
            with open(args.input, encoding="utf-8") as f:
                data = json.load(f)
            articles = data.get("articles", data) if isinstance(data, dict) else data
            results = analyze_articles(articles, args.output_dir)
            export_dashboard_data(results, articles, args.output_dir)

        elif args.command == "full":
            api_key = load_api_key(args.api_key)
            os.makedirs(args.output_dir, exist_ok=True)
            collected_path = os.path.join(args.output_dir, "collected_articles.json")

            client = NYTClient(api_key)
            articles = collect_articles(
                client, args.start, args.end, collected_path, args.method
            )
            results = analyze_articles(articles, args.output_dir)
            export_dashboard_data(results, articles, args.output_dir)
    except RuntimeError as e:
        print(f"ERROR: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
