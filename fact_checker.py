"""
fact_checker.py

Given a claim and a list of search results, this module:
  1. Fetches and cleans the text of each source page (falls back to
     the search snippet if the page can't be fetched/parsed).
  2. Runs a local, free Natural Language Inference (NLI) model to
     decide whether each source's text supports, contradicts, or is
     neutral toward the claim.
  3. Aggregates those per-source judgments into one of four verdicts
     -- True, False, Misleading, Unverified -- plus a plain-English
     "evidence strength" label (no raw confidence numbers exposed).
  4. Hands the collected evidence to summarizer.generate_summary()
     to produce a short, natural-language explanation, grounded in
     what was actually retrieved.

The NLI model (cross-encoder/nli-deberta-v3-xsmall) runs entirely
locally via HuggingFace `transformers` -- no paid API, no API key.
It is downloaded once (a few hundred MB) the first time you run the app.
"""

import re
from datetime import date
from urllib.parse import urlparse

import requests
import torch
from bs4 import BeautifulSoup
from transformers import AutoTokenizer, AutoModelForSequenceClassification

from summarizer import generate_summary, DEFAULT_MODEL as DEFAULT_SUMMARY_MODEL
from timing import timed

NLI_MODEL_NAME = "cross-encoder/nli-deberta-v3-xsmall"
LABELS = ["contradiction", "entailment", "neutral"]
_SMALL_NUMBERS = {
    "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
    "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10,
    "eleven": 11, "twelve": 12, "thirteen": 13, "fourteen": 14,
    "fifteen": 15, "sixteen": 16, "seventeen": 17, "eighteen": 18,
    "nineteen": 19, "twenty": 20,
}

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36"
    )
}

_MODEL_CACHE = {}


def load_nli_model():
    """Load (and cache) the tokenizer + model. Safe to call repeatedly."""
    if "tokenizer" not in _MODEL_CACHE:
        tokenizer = AutoTokenizer.from_pretrained(NLI_MODEL_NAME)
        model = AutoModelForSequenceClassification.from_pretrained(NLI_MODEL_NAME)
        model.eval()
        _MODEL_CACHE["tokenizer"] = tokenizer
        _MODEL_CACHE["model"] = model
    return _MODEL_CACHE["tokenizer"], _MODEL_CACHE["model"]


def fetch_page_text(url: str, timeout: int = 6, max_chars: int = 4000, timing: dict = None) -> str:
    """Best-effort fetch + extraction of the main text of a web page."""
    try:
        with timed(timing, "page_fetch"):
            resp = requests.get(url, headers=HEADERS, timeout=timeout)
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, "lxml")
        for tag in soup(["script", "style", "nav", "footer", "header", "aside", "form"]):
            tag.decompose()
        # Government portfolio pages often store the actual names and roles
        # in tables rather than paragraph tags. Extract visible body text so
        # those official listings can be checked as evidence too.
        body = soup.body or soup
        text = " ".join(body.stripped_strings).strip()
        return text[:max_chars]
    except Exception:
        return ""


def _chunk_words(text: str, chunk_size: int = 180):
    words = text.split()
    if not words:
        return []
    return [" ".join(words[i:i + chunk_size]) for i in range(0, len(words), chunk_size)]


def _classify_pair(premise: str, hypothesis: str, tokenizer, model):
    with torch.no_grad():
        features = tokenizer(
            [premise], [hypothesis],
            padding=True, truncation=True, max_length=256, return_tensors="pt",
        )
        logits = model(**features).logits
        probs = torch.softmax(logits, dim=1)[0]
    return {LABELS[i]: float(probs[i]) for i in range(len(LABELS))}


def score_source_against_claim(claim: str, source_text: str, tokenizer, model):
    """Return the strongest entailment/contradiction signal found in source_text,
    along with the exact chunk of text that produced it (used later as a
    grounding excerpt for the generated summary)."""
    best = {"label": "neutral", "score": 0.0, "chunk": ""}
    for chunk in _chunk_words(source_text):
        scores = _classify_pair(chunk, claim, tokenizer, model)
        top_label = max(("entailment", "contradiction"), key=lambda lbl: scores[lbl])
        if scores[top_label] > best["score"]:
            best = {"label": top_label, "score": scores[top_label], "chunk": chunk}
    return best


def _map_verdict(support_count: int, refute_count: int) -> str:
    # A mix of support/refutation means the retrieved evidence conflicts; it
    # does not by itself prove the claim is partly true.
    if support_count == refute_count and support_count > 0:
        return "Unverified"
    if support_count > refute_count:
        return "True"
    if refute_count > support_count:
        return "False"
    return "Unverified"


def _current_office_support(claim: str, url: str, source_text: str, as_of_year: int = None):
    """Recognize direct, current-year confirmation from an official government host."""
    claim = (claim or "").strip().lower()
    match = re.match(
        r"^(?P<person>.+?)\s+(?:is|currently\s+is|serves\s+as|currently\s+serves\s+as)\s+"
        r"(?:the\s+)?(?P<office>.+?)[.!?]*$",
        claim,
    )
    if not match:
        return False

    host = (urlparse(url).hostname or "").lower()
    if not host.endswith((".gov", ".gov.in", ".nic.in")):
        return False
    year = as_of_year if as_of_year is not None else date.today().year
    text = re.sub(r"[^a-z0-9]+", " ", (source_text or "").lower())
    if str(year) not in text:
        return False

    person_tokens = re.findall(r"[a-z0-9]+", match.group("person"))
    office = re.sub(r"\s+", " ", match.group("office")).strip()
    office_core = office.split(" of ")[0]
    office_tokens = re.findall(r"[a-z0-9]+", office_core)
    return bool(
        person_tokens
        and office_tokens
        and all(token in text for token in person_tokens)
        and all(token in text for token in office_tokens)
    )


def _tenure_duration_conflict(claim: str, excerpt: str, as_of_year: int = None):
    """Catch explicit tenure claims contradicted by a dated start event."""
    claim_lower = (claim or "").lower()
    text = (excerpt or "").strip()
    year_match = re.search(
        r"\b(?P<count>\d{1,3}|one|two|three|four|five|six|seven|eight|nine|ten|"
        r"eleven|twelve|thirteen|fourteen|fifteen|sixteen|seventeen|eighteen|"
        r"nineteen|twenty)\s*(?:-\s*)?years?\b",
        claim_lower,
    )
    if not year_match:
        return None
    count_text = year_match.group("count")
    claimed_years = int(count_text) if count_text.isdigit() else _SMALL_NUMBERS[count_text]

    offices = (
        "prime minister", "president", "governor", "mayor", "chief executive",
        "chairman", "chairwoman", "chairperson", "ceo",
    )
    role = next((item for item in offices if item in claim_lower), None)
    if role is None and re.search(r"\bpm\b", claim_lower):
        role = "prime minister"
    if role is None and re.search(r"\b(?:rule|ruled|ruling|govern|governed|governing)\b", claim_lower):
        # "Ruled for N years" is a tenure claim even when it omits a formal
        # title. Compare it with a dated government leadership start found in
        # the evidence, but only when that excerpt identifies the same person.
        role = "government leadership"
    if not role:
        return None

    current_year = as_of_year if as_of_year is not None else date.today().year
    person_match = re.search(
        r"^(?P<person>.+?)\s+(?:(?:has|is)\s+(?:been\s+)?)?(?:ruling|ruled|governing|governed)\b",
        claim_lower,
    ) if role == "government leadership" else None
    person_tokens = re.findall(r"[a-z0-9]+", person_match.group("person")) if person_match else []
    if person_tokens and not all(token in re.sub(r"[^a-z0-9]+", " ", text.lower()) for token in person_tokens):
        return None

    for sentence in re.split(r"(?<=[.!?])\s+", text):
        sentence_lower = sentence.lower()
        role_present = role == "government leadership" and any(
            office in sentence_lower for office in ("head of government", "prime minister", "chief minister")
        )
        role_present = role_present or role in sentence_lower or (
            role == "prime minister" and re.search(r"\bpm\b", sentence_lower)
        )
        if not role_present:
            continue
        start = re.search(
            r"\b(?:since|from|in|on|appointed|sworn\s+in|took\s+office|became)\b"
            r"[\s\S]{0,180}?\b(?P<year>19\d{2}|20\d{2})\b",
            sentence_lower,
        )
        if not start:
            continue
        started_year = int(start.group("year"))
        elapsed_years = max(0, current_year - started_year)
        # The source gives a year, not a precise month/day.
        if claimed_years > elapsed_years + 1:
            return {
                "claimed_years": claimed_years,
                "elapsed_years": elapsed_years,
                "start_year": started_year,
                "excerpt": sentence.strip(),
            }
    return None


def is_tenure_duration_claim(claim: str) -> bool:
    """Identify claims routed through the fixed path's dated-tenure guard."""
    claim_lower = (claim or "").lower()
    has_duration = re.search(
        r"\b(?:\d{1,3}|one|two|three|four|five|six|seven|eight|nine|ten|"
        r"eleven|twelve|thirteen|fourteen|fifteen|sixteen|seventeen|eighteen|"
        r"nineteen|twenty)\s*(?:-\s*)?years?\b",
        claim_lower,
    )
    has_office = any(
        office in claim_lower for office in (
            "prime minister", "president", "governor", "mayor", "chief executive",
            "chairman", "chairwoman", "chairperson", "ceo",
        )
    ) or bool(re.search(r"\bpm\b", claim_lower)) or bool(
        re.search(r"\b(?:rule|ruled|ruling|govern|governed|governing)\b", claim_lower)
    )
    return bool(has_duration and has_office)


def _evidence_strength(support_count: int, refute_count: int) -> str:
    """A plain-English stand-in for a raw confidence percentage."""
    strongest = max(support_count, refute_count)
    if strongest >= 3:
        return "Strong evidence"
    if strongest >= 1:
        return "Some evidence"
    return "Limited evidence"


def verify_claim(
    claim: str,
    search_results,
    tokenizer,
    model,
    min_confidence: float = 0.55,
    summarizer_model: str = DEFAULT_SUMMARY_MODEL,
    timing: dict = None,
):
    """
    Check `claim` against a list of search results
    (each a dict with "title", "url", "snippet").

    Returns a dict:
        {
          "claim": str,
          "verdict": "True" | "False" | "Misleading" | "Unverified",
          "strength": str,           # plain-English evidence strength
          "summary": str,            # generated explanation
          "sources": [ {"title", "url"}, ... ]
        }
    """
    evidence = []
    for res in search_results:
        page_text = fetch_page_text(res["url"], timing=timing)
        text_to_check = page_text if page_text.strip() else res.get("snippet", "")
        if not text_to_check.strip():
            continue
        with timed(timing, "nli_scoring"):
            result = score_source_against_claim(claim, text_to_check, tokenizer, model)
        stance = result["label"] if result["score"] >= min_confidence else "neutral"
        duration_conflict = _tenure_duration_conflict(
            claim, "\n".join(part for part in (page_text, res.get("snippet", "")) if part)
        )
        if duration_conflict:
            # A dated start directly contradicts an impossible tenure,
            # even if the NLI model confuses the wording.
            stance = "contradiction"
            result["chunk"] = duration_conflict["excerpt"]
        current_office_support = _current_office_support(
            claim, res["url"],
            "\n".join(part for part in (page_text, res.get("snippet", "")) if part),
        )
        if current_office_support:
            stance = "entailment"
            if not result.get("chunk"):
                result["chunk"] = res.get("snippet", "")[:220]
        evidence.append({
            "url": res["url"],
            "title": res.get("title") or res["url"],
            "stance": stance,
            "excerpt": result.get("chunk", "")[:220],
            "current_office_support": current_office_support,
        })

    support_count = sum(1 for e in evidence if e["stance"] == "entailment")
    refute_count = sum(1 for e in evidence if e["stance"] == "contradiction")

    authoritative_current_support = any(e["current_office_support"] for e in evidence)
    if authoritative_current_support:
        # A current, direct government listing supersedes contradictory stale
        # search results for a present-tense office-holding claim.
        for item in evidence:
            if item["stance"] == "contradiction" and not item["current_office_support"]:
                item["stance"] = "neutral"
        support_count = sum(1 for e in evidence if e["stance"] == "entailment")
        refute_count = sum(1 for e in evidence if e["stance"] == "contradiction")
        verdict = "True"
    else:
        verdict = _map_verdict(support_count, refute_count) if evidence else "Unverified"
    strength = _evidence_strength(support_count, refute_count)
    with timed(timing, "summary_generation"):
        summary = generate_summary(claim, verdict, evidence, model=summarizer_model, timing=timing)

    sources = [{"title": e["title"], "url": e["url"]} for e in evidence]

    return {
        "claim": claim,
        "verdict": verdict,
        "strength": strength,
        "summary": summary,
        "sources": sources,
    }
