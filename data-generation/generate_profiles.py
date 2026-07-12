"""Per-query traceable profile generation.

For every query in ``prefertripplan.jsonl`` we:

  1. Disintegrate ``local_constraint`` + every preference into atomic
     *source records*.  Set-valued sub-preds (cuisine / house_rules /
     attraction.category) are expanded per element regardless of nesting
     depth — whether they appear in a top-level ``AtomicPreference`` or
     inside a non-atomic paradigm (Composite / Conditional /
     Lexicographic / Compensatory / Temporal / Scoped).  Each element
     becomes its own ``Source`` record and its own drift unit; same-value
     sources across paradigms share a ``(trait_table, trait_key)`` drift
     group so they drift in lockstep when picked.
  2. Route each source through ``profile_traits`` to a trait table key.
     Numeric atomic sub-preds resolve their quartile (Q1 / Q2 / Q3) by
     walking the parent bank entry's ``example_values.default`` along
     the source's path with ``profile_traits.quartile_at``.
  3. Pick an aligned variant deterministically via ``pick_variant`` on
     ``(query_id, source descriptor)``.
  4. Assign ``drift_mode`` per query (aligned / omission / inversion)
     stratified to 30 / 35 / 35 within each level (easy/medium/hard).
  5. Apply the drift mutation — omission drops a deterministically-chosen
     source; inversion swaps the source's aligned variant for its
     inversion variant in the same field.

Output: each augmented record gains
    ``profile``           — the trait-populated profile dict
    ``profile_drift_mode``— "aligned" | "omission" | "inversion"
    ``profile_trace``     — the full source provenance trail
"""
from __future__ import annotations

import hashlib
import json
import re
import shutil
import sys
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

import profile_traits as PT

IN_PATH  = ROOT / "prefertripplan.jsonl"
OUT_PATH = ROOT / "prefertripplan.jsonl"
BANK_PATH = ROOT / "preference_bank.json"
BACKUP   = ROOT / "prefertripplan.PRE_PROFILE.jsonl.bak"


# --------------------------------------------------------------------------- #
# Bank loader (tolerates the bank file's trailing commas)
# --------------------------------------------------------------------------- #

def _load_bank() -> dict:
    src = BANK_PATH.read_text()
    src = re.sub(r",(\s*[}\]])", r"\1", src)
    return json.loads(src)


def _build_bank_index(bank: dict) -> dict:
    """Index every bank entry by an addressable key so that, given a
    paradigm + bank_id (+ op for Temporal), we can fetch the full entry
    including ``example_values.default`` for quartile lookups."""
    idx: dict = {}
    for kind, section in bank.items():
        if kind == "TemporalPreference" and isinstance(section, dict):
            for op_name, entries in section.items():
                for e in entries:
                    if isinstance(e, dict) and "id" in e:
                        idx[(kind, op_name, e["id"])] = e
        elif isinstance(section, list):
            for e in section:
                if isinstance(e, dict) and "id" in e:
                    idx[(kind, e["id"])] = e
    return idx


# --------------------------------------------------------------------------- #
# Trip-context narrative-resolution vocabulary.
#
# Three constants drive how bank rationales are re-attributed in the
# generated trip text:
#
#   * ROLES_TO_REWRITE         — exact role tokens that, if present in a
#                                rationale, are candidates for substitution
#                                when the resolved trip_context differs
#                                from the role's natural context.
#   * ROLE_TO_NATURAL_CONTEXT  — maps each role to the trip-context
#                                category it naturally belongs to.
#   * BIO_SPECIFIC_MARKERS     — phrases that, when co-present with a
#                                substitutable role in a rationale,
#                                make plain substitution invent a fact;
#                                their presence triggers the template
#                                hedge fallback at NL render time.
#
# The marker list was grown from a one-pass review of all 210 bank
# rationales: items present here actually appear in the corpus.
# --------------------------------------------------------------------------- #

ROLES_TO_REWRITE = {
    # 2-person partner roles
    "my partner", "my spouse", "my husband", "my wife",
    "the couple", "couple",

    # family / child roles
    "the kids", "the children", "the child",
    "our kids", "our children", "young child",
    "the family", "our family",

    # academic / friend-group roles
    "the students", "we students", "students", "our class",
    "university students",

    # work / professional roles
    "the colleagues", "my colleagues",
}


ROLE_TO_NATURAL_CONTEXT = {
    "my partner":          "partner trip",
    "my spouse":           "partner trip",
    "my husband":          "partner trip",
    "my wife":             "partner trip",
    "the couple":          "partner trip",
    "couple":              "partner trip",

    "the kids":            "family trip",
    "the children":        "family trip",
    "the child":           "family trip",
    "our kids":            "family trip",
    "our children":        "family trip",
    "young child":         "family trip",
    "the family":          "family trip",
    "our family":          "family trip",

    "the students":        "student trip",
    "we students":         "student trip",
    "students":            "student trip",
    "university students": "student trip",
    "our class":           "student trip",

    "the colleagues":      "work trip",
    "my colleagues":       "work trip",
}


BIO_SPECIFIC_MARKERS = {
    # Health / medical / dietary conditions — present in the bank rationales
    "allergic", "allergy", "fish allergy",
    "asthma", "celiac", "diabetic",
    "intolerance", "intolerant", "lactose",
    "light-sleeper", "light sleeper",
    "recovery", "recovering", "recovering gambler", "recovering from surgery",
    "burnout", "recover from burnout",
    "difficulty sleeping", "fatigue-related",
    "history of fatigue-related",

    # Specific occupations / identities / life-stages — observed in rationales
    "academic", "academic traveler",
    "backpacker",
    "university", "university students",
    "pooled budget", "social tension", "cost-splitting",
    "blogger", "blogging", "food blogging",
    "travel blog", "blog with", "subscribers", "80k subscribers",
    "crowd-aware traveler",
    "desk worker", "desk workers", "sedentary desk workers",
    "food enthusiast",
    "hands-on learner",
    "health enthusiast",
    "music enthusiast", "music / theater fan",
    "newlywed",
    "photographer",
    "physical therapist",
    "pragmatic wellness traveler",
    "retired", "retired person",
    "light traveller", "light traveler",

    # Time-anchored personal events — observed in rationales
    "21st birthday",
    "anniversary", "anniversary couple",
    "bucket-list", "bucket-list goal",
    "corporate reimbursement",
    "first-time", "first-time visitor", "first-time tourist",
    "honeymoon", "luxury honeymoon", "babymoon",
    "limited vacation", "limited vacation time",
    "milestone", "milestone trip",
    "promotion", "promoted",
    "school-year", "school-year reward",

    # Family / household specifics — observed in rationales
    "ages 8 and 10",
    "children ages",
    "early riser", "early risers",
    "newborn", "infant", "toddler", "teenager",
    "young child", "family with young child",
    "younger members", "younger travelers", "younger traveler",
    "grandparent", "in-laws",
    "bedtime", "naptime",

    # Specific group / event composition — observed in rationales
    "host someone", "visitor friend",
    "returning to hometown", "hometown",
    "group with recovering gambler",

    # Specific geographic / cultural identifiers (rare but defensible)
    "bay geography",  # SF
}


def _detect_role(rationale_text: str) -> str | None:
    """Return the first ROLES_TO_REWRITE token found in the rationale
    (case-insensitive substring match), or None if no role appears.
    Sorts by descending length so longer tokens like "my partner" win
    over "couple" / "partner" alone."""
    text = rationale_text.lower()
    for role in sorted(ROLES_TO_REWRITE, key=lambda r: -len(r)):
        if role in text:
            return role
    return None


def _has_bio_marker(rationale_text: str) -> str | None:
    """Return the first BIO_SPECIFIC_MARKERS phrase found in the
    rationale (case-insensitive substring match), or None."""
    text = rationale_text.lower()
    for marker in sorted(BIO_SPECIFIC_MARKERS, key=lambda m: -len(m)):
        if marker in text:
            return marker
    return None


def should_use_template_fallback(rationale: str,
                                  resolved_trip_context: str) -> tuple[bool, dict]:
    """Decide whether to use the template-hedge fallback for this
    rationale when rendering the trip text.

    The hedge fires ONLY when all three conditions hold:
      (1) a substitutable role token appears in the rationale
      (2) a bio-specific marker also appears in the same rationale
      (3) the chosen trip_context category differs from the role's
          natural category

    Returns ``(use_fallback, evidence)`` where ``evidence`` records the
    detected role, marker, and natural context for the trace."""
    role = _detect_role(rationale)
    if role is None:
        return False, {"role": None, "marker": None,
                       "natural_context": None, "mismatch": False}
    natural_ctx = ROLE_TO_NATURAL_CONTEXT.get(role)
    mismatch = (natural_ctx is not None
                and natural_ctx != resolved_trip_context)
    if not mismatch:
        return False, {"role": role, "marker": None,
                       "natural_context": natural_ctx, "mismatch": False}
    marker = _has_bio_marker(rationale)
    if marker is None:
        return False, {"role": role, "marker": None,
                       "natural_context": natural_ctx, "mismatch": True}
    return True, {"role": role, "marker": marker,
                  "natural_context": natural_ctx, "mismatch": True}


# --------------------------------------------------------------------------- #
# Trip-context phrase pool — count-consistent phrasing keyed by
# people_number.  Each entry pairs a display phrase with the trip-context
# category (used to match against ROLE_TO_NATURAL_CONTEXT when deciding
# whether to substitute or hedge a rationale).
#
# Used only when no rationale in the query carries a group keyword; the
# rationale-derived category takes precedence when present.
# --------------------------------------------------------------------------- #

# NOTE: phrases in this pool must be neutral about TRANSPORTATION mode
# and WEEK GROUPING. Avoid words like "road trip", "drive", "flight", "rail
# journey", "bus tour" (which leak the transportation mode into the opener
# even when the local_constraint has no transportation restriction) and
# avoid "weekend", "weekday" (which may clash with the actual date_day
# schedule and pre-commit the trip to specific day types). Use mode-neutral,
# day-agnostic descriptors: "trip", "getaway", "vacation", "adventure",
# "retreat", "holiday", "reunion".
COUNT_CONSISTENT_POOL: dict = {
    1: [
        ("solo trip",          "solo trip"),
        ("personal getaway",   "solo trip"),
        ("solo adventure",     "solo trip"),
        ("individual retreat", "solo trip"),
        ("solo vacation",      "solo trip"),
    ],
    2: [
        ("trip with my partner",       "partner trip"),
        ("getaway with my spouse",     "partner trip"),
        ("couple's getaway",           "partner trip"),
        ("two-person trip",            "partner trip"),
        ("trip with a close friend",   "friend trip"),
        ("trip with a longtime friend","friend trip"),
    ],
    3: [
        ("trip with two close friends","friend trip"),
        ("trio adventure",             "friend trip"),
        ("three-person friend trip",   "friend trip"),
        ("couple-with-child trip",     "family trip"),
        ("three-person family trip",   "family trip"),
    ],
    4: [
        ("small family of four",       "family trip"),
        ("family-of-four vacation",    "family trip"),
        ("four-person friend trip",    "friend trip"),
        ("two-couple trip",            "friend trip"),
        ("four-person friend group",   "friend trip"),
    ],
    5: [
        ("family trip with grandparent",   "family trip"),
        ("five-person friend group trip",  "friend trip"),
        ("mixed family-and-friend trip",   "group trip"),
        ("extended family trip",           "family trip"),
    ],
    6: [
        ("multi-family vacation",      "group trip"),
        ("friend group trip",          "friend trip"),
        ("two-family vacation",        "family trip"),
        ("extended family vacation",   "family trip"),
    ],
    "7+": [
        ("extended family reunion",     "family trip"),
        ("large group trip",            "group trip"),
        ("multi-generation vacation",   "family trip"),
        ("office / colleague group trip","work trip"),
        ("milestone group reunion",     "group trip"),
    ],
}


def _trip_context_bucket(people_number: int):
    if people_number is None:
        return 1
    if people_number <= 6:
        return people_number
    return "7+"


def resolve_trip_context(rec: dict) -> dict:
    """Resolve a query's trip_context.

    Two-stage:
      (1) Scan every rationale on the query for group-keyword tokens in
          ROLE_TO_NATURAL_CONTEXT and collect matched categories.
          A rationale-derived category is only accepted if it is
          *consistent* with the query's people_number bucket -- i.e. it
          appears in COUNT_CONSISTENT_POOL[bucket]'s valid categories.
          This prevents e.g. a "couple" role token in the rationale of
          a people_number=1 query from selecting category=partner_trip
          when the actual trip is solo.
          If >=1 consistent match: pick a category deterministically via
          md5(qid) modulo the filtered candidate set. Pick a phrase from
          that bucket whose category matches (fallback to any bucket
          phrase if the pair {category, bucket-phrase} is empty).
      (2) If no rationale matched (or all matches were inconsistent
          with people_number): pick a (phrase, category) pair
          deterministically from COUNT_CONSISTENT_POOL[people_number].

    Returns a dict with phrase, category, candidates, source
    ("rationale" | "rationale_dropped_inconsistent" | "people_number_fallback")
    and the people_number bucket.
    """
    qid = rec.get("query_id")
    people = rec.get("people_number") or 1

    # (1) scan rationales for group keywords
    matches: list[str] = []
    for pref in rec.get("preferences", []) or []:
        rationale = (pref.get("rationale") or "").lower()
        if not rationale:
            continue
        for role, ctx in ROLE_TO_NATURAL_CONTEXT.items():
            if role in rationale and ctx not in matches:
                matches.append(ctx)

    bucket = _trip_context_bucket(people)
    pool = COUNT_CONSISTENT_POOL.get(bucket, COUNT_CONSISTENT_POOL["7+"])
    # Categories consistent with this people_number bucket.
    valid_categories = {c for (_phrase, c) in pool}

    # Filter rationale matches to only categories consistent with the
    # people count. This is the fix: a "couple" mention in a solo query
    # no longer flips category to partner_trip -- solo bucket categories
    # never include partner_trip, so the mention is dropped.
    consistent_matches = [m for m in matches if m in valid_categories]

    if consistent_matches:
        distinct = sorted(set(consistent_matches))
        h = hashlib.md5(f"trip_context_cat|{qid}".encode()).digest()
        cat = distinct[int.from_bytes(h[:4], "big") % len(distinct)]
        # Try to pick a phrase from the bucket that matches the category
        cat_matching = [p for p in pool if p[1] == cat]
        chosen_pool = cat_matching if cat_matching else pool
        ph = hashlib.md5(f"trip_context_phrase|{qid}".encode()).digest()
        phrase, phrase_cat = chosen_pool[int.from_bytes(ph[:4], "big") % len(chosen_pool)]
        return {
            "phrase":            phrase,
            "category":          cat,
            "source":            "rationale",
            "rationale_matches": distinct,
            "rationale_dropped_matches": sorted(set(matches) - set(consistent_matches)),
            "people_number":     people,
            "bucket":            bucket,
        }

    # (2) fallback: no consistent rationale match, use people_number bucket
    h = hashlib.md5(f"trip_context_fallback|{qid}".encode()).digest()
    phrase, phrase_cat = pool[int.from_bytes(h[:4], "big") % len(pool)]
    return {
        "phrase":            phrase,
        "category":          phrase_cat,
        # If we had matches but all dropped for inconsistency, tag that.
        "source":            ("rationale_dropped_inconsistent"
                              if matches else "people_number_fallback"),
        "rationale_matches": [],
        "rationale_dropped_matches": sorted(set(matches)) if matches else [],
        "people_number":     people,
        "bucket":            bucket,
    }


# --------------------------------------------------------------------------- #
# Role-substitution for rationale text rewriting.
# --------------------------------------------------------------------------- #

# Per-target-context substitution defaults.  When a rationale carries a
# role token whose natural context differs from the resolved trip
# context AND no bio-specific marker fires (so straight substitution is
# safe), the role gets replaced with the value from this map.
ROLE_SUBSTITUTION_BY_CTX = {
    # (role, target_context) -> replacement phrase  (None = keep verbatim)
    ("my partner",          "family trip"):  "my spouse",
    ("my partner",          "friend trip"):  "a friend",
    ("my partner",          "student trip"): "a friend",
    ("my partner",          "group trip"):   "one of us",
    ("my partner",          "work trip"):    "one of us",
    ("my partner",          "solo trip"):    "I",

    ("my spouse",           "family trip"):  "my spouse",
    ("my spouse",           "friend trip"):  "a friend",
    ("my spouse",           "student trip"): "a friend",
    ("my spouse",           "group trip"):   "one of us",
    ("my spouse",           "work trip"):    "one of us",
    ("my spouse",           "solo trip"):    "I",

    ("my husband",          "family trip"):  "my husband",
    ("my husband",          "friend trip"):  "a friend",
    ("my husband",          "group trip"):   "one of us",
    ("my husband",          "solo trip"):    "I",
    ("my wife",             "family trip"):  "my wife",
    ("my wife",             "friend trip"):  "a friend",
    ("my wife",             "group trip"):   "one of us",
    ("my wife",             "solo trip"):    "I",

    ("the couple",          "family trip"):  "we",
    ("the couple",          "friend trip"):  "we",
    ("the couple",          "student trip"): "we",
    ("the couple",          "group trip"):   "we",
    ("the couple",          "work trip"):    "we",
    ("the couple",          "solo trip"):    "I",
    ("couple",              "family trip"):  "we",
    ("couple",              "friend trip"):  "we",
    ("couple",              "student trip"): "we",
    ("couple",              "group trip"):   "we",
    ("couple",              "work trip"):    "we",
    ("couple",              "solo trip"):    "I",

    ("the family",          "partner trip"): "we",
    ("the family",          "friend trip"):  "the group",
    ("the family",          "student trip"): "the group",
    ("the family",          "group trip"):   "the group",
    ("the family",          "work trip"):    "the group",
    ("the family",          "solo trip"):    "I",
    ("our family",          "partner trip"): "we",
    ("our family",          "friend trip"):  "the group",
    ("our family",          "student trip"): "the group",
    ("our family",          "group trip"):   "the group",
    ("our family",          "work trip"):    "the group",
    ("our family",          "solo trip"):    "I",

    ("the students",        "partner trip"): "I",
    ("the students",        "family trip"):  "we",
    ("the students",        "friend trip"):  "we",
    ("the students",        "group trip"):   "we",
    ("we students",         "partner trip"): "I",
    ("we students",         "family trip"):  "we",
    ("we students",         "friend trip"):  "we",
    ("we students",         "group trip"):   "we",
    ("students",            "partner trip"): "I",
    ("students",            "family trip"):  "we",
    ("students",            "friend trip"):  "we",
    ("students",            "group trip"):   "we",
    ("university students", "partner trip"): "I",
    ("university students", "family trip"):  "we",
    ("university students", "friend trip"):  "we",
    ("university students", "group trip"):   "we",

    ("the colleagues",      "partner trip"): "I",
    ("the colleagues",      "family trip"):  "we",
    ("the colleagues",      "friend trip"):  "the group",
    ("the colleagues",      "student trip"): "the group",
    ("the colleagues",      "group trip"):   "we",
    ("my colleagues",       "partner trip"): "I",
    ("my colleagues",       "family trip"):  "we",
    ("my colleagues",       "friend trip"):  "the group",
    ("my colleagues",       "student trip"): "the group",
    ("my colleagues",       "group trip"):   "we",
}


def _substitute_roles(rationale: str, target_ctx: str) -> tuple[str, list[dict]]:
    """Apply role substitutions to a rationale where natural-context
    differs from the target trip context.  Skips bio-fragile cases
    (kid-bearing tokens with non-family target) by leaving them as-is —
    the should_use_template_fallback gate will have already routed those
    to hedge replacement upstream.

    Returns ``(rewritten_text, substitutions_log)``."""
    out = rationale
    log: list[dict] = []
    if not rationale:
        return out, log
    text_lower = out.lower()
    # Order by descending length so multi-word roles win
    for role in sorted(ROLES_TO_REWRITE, key=lambda r: -len(r)):
        if role not in text_lower:
            continue
        natural = ROLE_TO_NATURAL_CONTEXT.get(role)
        if natural is None or natural == target_ctx:
            continue
        repl = ROLE_SUBSTITUTION_BY_CTX.get((role, target_ctx))
        if repl is None:
            continue
        # Case-preserving replacement on the lowered match
        idx = text_lower.find(role)
        while idx != -1:
            out = out[:idx] + repl + out[idx + len(role):]
            text_lower = out.lower()
            log.append({"role": role, "replacement": repl,
                        "target_ctx": target_ctx})
            idx = text_lower.find(role, idx + len(repl))
    return out, log


# --------------------------------------------------------------------------- #
# NL profile + NL query renderers
# --------------------------------------------------------------------------- #

_FIELD_OPENERS = {
    "Hobbies":                      "Hobbies and interests: ",
    "Lifestyle":                    "Lifestyle: ",
    "Travel Style":                 "Travel style: ",
    "Preferred Destinations":       "Preferred destinations: ",
    "Food and Dining Preferences":  "Food and dining: ",
    "Dislikes":                     "Things I actively avoid: ",
}


def render_nl_profile(rec: dict) -> str:
    """Compose the single-user profile text from the profile dict
    already attached to ``rec``.  Reads variant texts only — never bank
    rationale.  First-person voice, simple template per Interests field."""
    p = rec.get("profile") or {}
    interests = p.get("Interests") or {}
    loc = (p.get("Demographics") or {}).get("Location") or rec.get("org") or "the area"

    parts = [f"I'm based in {loc}."]
    for field in ("Hobbies", "Lifestyle", "Travel Style",
                  "Preferred Destinations", "Food and Dining Preferences",
                  "Dislikes"):
        vs = interests.get(field) or []
        if not vs:
            continue
        opener = _FIELD_OPENERS[field]
        # Join variant sentences with single space; each variant text is
        # already a full sentence.
        body = " ".join(vs)
        parts.append(opener + body)

    pets = p.get("Pets")
    if pets:
        parts.append(f"Pets: {pets}")
    return "\n\n".join(parts)


def _concession_prefix(category: str, all_drifted: bool) -> str:
    """Phrasing that introduces a rationale whose underlying sub-preds
    have one or more drifted sources."""
    intensifier = "fully" if all_drifted else "partially"
    if category == "partner trip":
        return f"My partner and I worked out a compromise here ({intensifier} a give-and-take):"
    if category == "family trip":
        return f"As a family, we agreed ({intensifier} a compromise on my part):"
    if category == "friend trip":
        return f"Our group agreed on this ({intensifier} a give-and-take):"
    if category == "student trip":
        return f"Our group agreed on this for the trip ({intensifier} a compromise):"
    if category == "group trip":
        return f"The group settled on this ({intensifier} a give-and-take):"
    if category == "work trip":
        return f"The group settled on this for the trip ({intensifier} a compromise):"
    if category == "solo trip":
        return "I went along with this (it's not quite my usual preference):"
    return "We've agreed to this for the trip:"


_ATTR_NL_ALIAS = {
    ("Accommodation", "rating"): "review_rate",
    ("Accommodation", "cost"):   "cost",
    ("Restaurant",    "rating"): "aggregate_rating",
    ("Restaurant",    "cost"):   "cost",
    ("Restaurant",    "cuisines"): "cuisine",
}
_OP_SYMBOL = {"==": "=", "!=": "≠", ">=": "≥", "<=": "≤",
              ">": ">", "<": "<", "in": "∈", "not_in": "∉"}


def _fmt_attr(entity: str, attribute: str) -> str:
    return _ATTR_NL_ALIAS.get((entity, attribute), attribute)


def _fmt_value(val):
    if isinstance(val, list):
        return "{" + ", ".join(str(v) for v in val) + "}"
    return str(val)


def _fmt_op_val(op: str, val) -> str:
    sym = _OP_SYMBOL.get(op, op)
    return f"{sym} {_fmt_value(val)}"


def _fmt_scope(scope) -> str:
    if scope in (None, ""):
        return ""
    return f" [scope: {scope}]"


def _fmt_atomic(node: dict, with_scope: bool = True) -> str:
    """Format a resolved AtomicPreference-like dict."""
    ent  = node.get("entity_type") or node.get("entity")
    attr = _fmt_attr(ent, node.get("attribute"))
    op   = node.get("op")
    val  = node.get("value")
    text = f"{ent}.{attr} {_fmt_op_val(op, val)}"
    if with_scope:
        text += _fmt_scope(node.get("scope"))
    return text


def _fmt_composite(tmpl: dict) -> str:
    op = tmpl.get("op", "AND")
    parts = []
    for c in tmpl.get("children") or []:
        parts.append(_fmt_atomic(c, with_scope=True))
    joiner = f" {op} "
    return joiner.join(parts) if parts else ""


def _fmt_conditional(tmpl: dict) -> str:
    cond = tmpl.get("condition") or {}
    then_p = tmpl.get("then_pref") or {}
    return f"IF {_fmt_atomic(cond, with_scope=False)} THEN {_fmt_atomic(then_p, with_scope=False)}"


def _fmt_lexicographic(tmpl: dict) -> str:
    prefs = tmpl.get("preferences") or []
    return " ≻ ".join(_fmt_atomic(p, with_scope=True) for p in prefs)


def _fmt_compensatory(tmpl: dict) -> str:
    """Render Compensatory with explicit scope tags on each slot so the
    universal-primary/margin vs same-day-existential-secondary semantic
    reads directly from the predicate literal.

    Layout:
        PRIMARY [scope: all]: <ideal>  |  MARGIN [scope: all]: <fallback>
        |  SECONDARY [scope: any, same day]: <compensator>

    meaning "every primary-entity item across the plan lands at Primary
    (ideal) or in the Margin band (compensated by a same-day Secondary
    satisfier)."
    """
    p = tmpl.get("primary_ap")   or {}
    m = tmpl.get("margin_ap")    or {}
    s = tmpl.get("secondary_ap") or {}
    return (
        f"PRIMARY [scope: all]: {_fmt_atomic(p, with_scope=False)} | "
        f"MARGIN [scope: all]: {_fmt_atomic(m, with_scope=False)} | "
        f"SECONDARY [scope: any, same day]: {_fmt_atomic(s, with_scope=False)}"
    )


def _fmt_numeric(tmpl: dict) -> str:
    ent  = tmpl.get("entity_type")
    attr = _fmt_attr(ent, tmpl.get("attribute"))
    direction = tmpl.get("direction", "max")
    agg = tmpl.get("aggregation", "avg")
    th = tmpl.get("threshold") or []
    fn = "max" if direction == "max" else "min"
    core = f"{fn}({ent}.{attr}) via {agg}"
    if isinstance(th, list) and len(th) == 2:
        core += f" [{th[0]}, {th[1]}]"
    return core


def _fmt_temporal(tmpl: dict) -> str:
    op = tmpl.get("op", "sometime")
    subj = tmpl.get("subject_ap") or {}
    ref  = tmpl.get("reference_ap")
    subj_txt = _fmt_atomic(subj, with_scope=False) if subj else ""
    if ref:
        ref_txt = _fmt_atomic(ref, with_scope=False)
        body = f"{subj_txt}, {ref_txt}"
    else:
        body = subj_txt
    # time_start / time_end / k parameters
    extras = []
    for k in ("time_start", "time_end", "k"):
        if k in tmpl and tmpl[k] is not None:
            extras.append(f"{k}={tmpl[k]}")
    if extras:
        body += f", {', '.join(extras)}"
    scope = tmpl.get("scope")
    return f"{op}({body})" + (_fmt_scope(scope) if scope else "")


def _fmt_scoped_inner(inner: dict) -> str:
    cls = inner.get("class")
    itmpl = inner.get("template") or {}
    if cls == "AtomicPreference":
        return _fmt_atomic(itmpl, with_scope=True)
    if cls == "CompositePreference":
        return _fmt_composite(itmpl)
    if cls == "ConditionalPreference":
        return _fmt_conditional(itmpl)
    if cls == "LexicographicPreference":
        return _fmt_lexicographic(itmpl)
    if cls == "CompensatoryPreference":
        return _fmt_compensatory(itmpl)
    if cls == "NumericPreference":
        return _fmt_numeric(itmpl)
    if cls == "TemporalPreference":
        return _fmt_temporal(itmpl)
    return str(itmpl)


def _fmt_scoped(tmpl: dict) -> str:
    inner = tmpl.get("inner") or {}
    scope_filters = tmpl.get("scope_filters") or []
    # Compact "WHEN attr op val: inner"
    whens = []
    for sf in scope_filters:
        sf_t = sf.get("template") or sf
        whens.append(_fmt_atomic(sf_t, with_scope=False))
    when_clause = " AND ".join(whens) if whens else ""
    inner_txt = _fmt_scoped_inner(inner)
    return f"WHEN {when_clause}: {inner_txt}" if when_clause else inner_txt


def format_resolved_predicate(pref: dict) -> str:
    """Render the resolved preference template into a human-readable
    predicate string using the actual quantitative/categorical values
    that were resolved for THIS query. This is the ONLY source of truth
    for the stringified predicate; the bank's original `raw_source`
    string is no longer emitted on the augmented record because it
    could diverge from resolved values (constraint-filter intersection,
    quartile grounding, pair-adjust, budget escalation)."""
    paradigm = pref.get("paradigm")
    tmpl = pref.get("template") or {}
    if paradigm == "AtomicPreference":
        return _fmt_atomic(tmpl, with_scope=True)
    if paradigm == "CompositePreference":
        return _fmt_composite(tmpl)
    if paradigm == "ConditionalPreference":
        return _fmt_conditional(tmpl)
    if paradigm == "LexicographicPreference":
        return _fmt_lexicographic(tmpl)
    if paradigm == "CompensatoryPreference":
        return _fmt_compensatory(tmpl)
    if paradigm == "NumericPreference":
        return _fmt_numeric(tmpl)
    if paradigm == "TemporalPreference":
        return _fmt_temporal(tmpl)
    if paradigm == "ScopedPreference":
        return _fmt_scoped(tmpl)
    return ""


def _format_lc(lc: dict) -> str:
    """Format the local_constraint values as inline clauses to be appended
    to the trip header sentence.  Empty when no constraint values are set."""
    parts = []
    if lc.get("cuisine"):
        cus = lc["cuisine"] if isinstance(lc["cuisine"], list) else [lc["cuisine"]]
        parts.append(f"the trip needs to cover cuisines {', '.join(cus)}")
    if lc.get("house rule"):
        parts.append(f"the accommodation must allow {lc['house rule']}")
    if lc.get("room type"):
        parts.append(f"the room type must be {lc['room type']}")
    if lc.get("transportation"):
        parts.append(f"the transportation is {lc['transportation']}")
    return "; ".join(parts)


def render_nl_query(rec: dict, trip_context: dict) -> tuple[str, list[dict]]:
    """Compose the natural-language trip request text.

    The text is built from:
      * trip header (people / org / dest / days / budget),
      * local_constraint values,
      * one paragraph per preference, with the bank rationale used as
        the load-bearing sentence — substituted, hedged, or kept based
        on drift state and the trip_context's category.

    Returns ``(nl_text, per_preference_render_log)`` where the log
    records, per preference, what was decided (use_verbatim,
    role_substitution, template_fallback, or concession_prefix added).
    """
    p_count   = rec.get("people_number") or 1
    days      = rec.get("days")
    org       = rec.get("org")
    dest      = rec.get("dest")
    budget    = rec.get("budget")
    visit_n   = rec.get("visiting_city_number")
    date_list  = rec.get("date") or []
    start_date = date_list[0]  if date_list else None
    end_date   = date_list[-1] if date_list else None
    phrase     = trip_context.get("phrase")
    category   = trip_context.get("category")

    # Block 1: trip facts + locked-in constraints (single paragraph).
    article = "an" if phrase[:1].lower() in "aeiou" else "a"
    header_clauses = [f"I'm planning {article} {phrase} from {org} to {dest}"]
    if start_date and end_date and start_date != end_date:
        header_clauses.append(f"for {days} days from {start_date} to {end_date} (inclusive)")
    elif start_date:
        header_clauses.append(f"for {days} days starting {start_date}")
    else:
        header_clauses.append(f"for {days} days")
    if visit_n and visit_n > 1:
        header_clauses.append(f"across {visit_n} cities")
    if p_count and p_count != 1:
        header_clauses.append(f"with {p_count} travelers")
    if budget is not None:
        header_clauses.append(f"on a budget of about ${int(budget)}")
    header = ", ".join(header_clauses) + "."

    lc_desc = _format_lc(rec.get("local_constraint") or {})
    if lc_desc:
        # Append the LC clauses to the same paragraph so the original
        # TravelPlanner-given facts and the local_constraint values read
        # together as one trip-statement block.
        header = header[:-1] + f"; {lc_desc}."
    parts = [header]

    # Per-person cost disclaimer: attach a single sentence when a preference
    # literal references an entity.cost threshold so the number is read on the
    # right basis.  Accommodation.cost is per-person per-night (see the
    # per-person effective-cost transform in augment_preferences.build_query_db);
    # Restaurant.cost follows the TravelPlanner Average-Cost convention
    # (per-person per-meal).
    _cost_hits = {"Accommodation": False, "Restaurant": False}
    def _collect_cost_hits(node):
        if isinstance(node, list):
            for it in node:
                _collect_cost_hits(it)
            return
        if not isinstance(node, dict):
            return
        ent = node.get("entity_type") or node.get("entity")
        if node.get("attribute") == "cost" and ent in _cost_hits:
            _cost_hits[ent] = True
        # Recurse into all likely subtree fields (paradigm-specific: nested
        # template wrapper for Scoped, children for Composite, condition/
        # then_pref for Conditional, primary/margin/secondary for
        # Compensatory, preferences for Lexicographic, inner + scope_filters
        # for Scoped).
        # Temporal uses subject_ap / reference_ap / target_ap; Scoped uses
        # inner; Composite uses children; Compensatory uses primary/margin/
        # secondary_ap; Conditional uses condition/then_pref; Lexicographic
        # uses preferences.  Cover every wrapper key any paradigm nests
        # atomic predicates under.
        for k in ("template", "condition", "then_pref", "primary_ap",
                  "margin_ap", "secondary_ap", "inner", "inner_pref",
                  "subject_ap", "reference_ap", "target_ap"):
            if node.get(k):
                _collect_cost_hits(node.get(k))
        _collect_cost_hits(node.get("children") or [])
        _collect_cost_hits(node.get("preferences") or [])
        _collect_cost_hits(node.get("scope_filters") or [])
    for _pref in (rec.get("preferences") or []):
        _collect_cost_hits(_pref.get("template") or {})
    _cost_notes = []
    if _cost_hits["Accommodation"]:
        _cost_notes.append("accommodation cost is per person per night")
    if _cost_hits["Restaurant"]:
        _cost_notes.append("restaurant cost is per person per meal")
    if _cost_notes:
        parts.append("Note: " + "; ".join(_cost_notes) + ".")

    # Group sources by their owning preference for drift-state lookups.
    sources_by_pref: dict[tuple, list[dict]] = {}
    for s in (rec.get("profile_trace") or {}).get("sources") or []:
        if s.get("kind") in ("destination", "local_constraint"):
            continue
        key = (s.get("paradigm"), s.get("op_name"), s.get("bank_id"))
        sources_by_pref.setdefault(key, []).append(s)

    render_log: list[dict] = []
    for pref in rec.get("preferences", []) or []:
        paradigm = pref.get("paradigm")
        bid      = pref.get("bank_id")
        op_name  = (pref.get("template") or {}).get("op") if paradigm == "TemporalPreference" else None
        rationale = (pref.get("rationale") or "").strip()
        if not rationale:
            continue

        srcs = sources_by_pref.get((paradigm, op_name, bid), [])
        drifted = [s for s in srcs if s.get("drifted")]
        n_total = len(srcs)
        n_drifted = len(drifted)
        any_drifted = n_drifted > 0
        all_drifted = n_drifted > 0 and n_drifted == n_total

        entry = {
            "preference":     f"{paradigm}/{op_name}#{bid}" if op_name else f"{paradigm}#{bid}",
            "rationale":      rationale,
            "n_sources":      n_total,
            "n_drifted":      n_drifted,
            "trip_context":   category,
        }

        use_fb, fb_evidence = should_use_template_fallback(rationale, category)
        entry["fallback_evidence"] = fb_evidence

        # Render the preference from the RESOLVED template (post-quartile
        # grounding, pair-adjust, and budget-multiplier envelope) so the
        # NL text matches solution_information / reference_information /
        # structured template exactly. `pref["raw_source"]` in the bank
        # is only a seed anchor and its values may diverge from resolved.
        resolved_src = format_resolved_predicate(pref)
        entry["resolved_source"] = resolved_src

        if use_fb and any_drifted:
            # Hedge — concrete values are still surfaced so the trip
            # remains plannable; the rationale itself is suppressed.
            text = (f"Constraint we've agreed on: {resolved_src}. "
                    f"(This doesn't fully match my own preferences — "
                    f"it's a compromise built into the plan.)")
            entry["action"] = "template_fallback_hedge"
            entry["text"]   = text
        else:
            # Role substitution where natural-context mismatches
            substituted, sub_log = _substitute_roles(rationale, category)
            if any_drifted:
                prefix = _concession_prefix(category, all_drifted)
                text = f"{prefix} {resolved_src}. Reasoning: {substituted}"
                entry["action"] = "concession_with_substitution" if sub_log else "concession"
            else:
                aligned_prefix = "My reason" if category == "solo trip" else "Our reason"
                text = f"{aligned_prefix}: {resolved_src}. {substituted}"
                entry["action"] = "aligned_with_substitution" if sub_log else "aligned_verbatim"
            entry["substitutions"] = sub_log
            entry["text"]          = text

        parts.append(entry["text"])
        render_log.append(entry)

    return "\n\n".join(parts), render_log


# --------------------------------------------------------------------------- #
# Normalization tables — mapping raw data values to trait-table keys.
# Grounded ONLY in vocabulary that actually appears in the dataset.
# --------------------------------------------------------------------------- #

# ROOM_TYPE: data uses both "entire home/apt" / "private room" / "shared room" /
# "not shared room" (local_constraint) and "Entire home/apt" / "Private room" /
# "Shared room" (preferences canonical form).  Map both to trait-table keys.
ROOM_TYPE_NORMALIZE = {
    "entire home/apt":      "entire home/apt",
    "private room":     "private room",
    "shared room":      "shared room",
    "not shared room":  "not shared room",
    "Entire home/apt":  "entire home/apt",
    "Private room":     "private room",
    "Shared room":      "shared room",
    "Not Shared room":  "not shared room",
}

# TRANSPORTATION:
#   * local_constraint    : "no flight" | "no self-driving"  (avoidance)
#   * preferences positive: ==/in {"self-driving","Flight","taxi"} or list
#   * preferences avoidance: not_in/!= {"Flight","self-driving"}
# The profile-trait table is anchored on the avoidance keys ("no flight",
# "no self-driving").  Positive preferences are routed to the *inversion*
# of the opposite avoidance trait — e.g. wanting self-driving = the
# road-trip-loving inversion of the "no self-driving" avoidance trait.
# Taxi preferences have no clean trait anchor; they are dropped.
TRANSPORT_FROM_RAW_AVOIDANCE = {
    "Flight":           "no flight",
    "flight":           "no flight",
    "self-driving":     "no self-driving",
    "self driving":     "no self-driving",
    "no flight":        "no flight",
    "no self-driving":  "no self-driving",
}
TRANSPORT_FROM_RAW_POSITIVE = {
    # mode the user actively prefers -> the avoidance trait whose
    # INVERSION best describes that preference.
    "Flight":       "no flight",         # frequent flyer = inverse of "no flight"
    "flight":       "no flight",
    "self-driving": "no self-driving",   # road-tripper  = inverse of "no self-driving"
    "self driving": "no self-driving",
}


# --------------------------------------------------------------------------- #
# Destination -> list of Preferred-Destination archetypes (up to 5 per
# destination).  At source-emission time, ONE archetype is sampled
# deterministically per query via profile_traits.pick_variant on the
# candidate list, so per-query reproducibility is preserved while
# dataset-wide profile variety is widened.
#
# Archetype vocabulary: 16 labels grounded in Table 7 of the
# travelplanner-plus-profile-space PDF (excluding "European Cities" as
# the corpus is US-only; merging "Tech Expos" into "Tech Conferences"
# and "Countryside" into "Quiet Countryside").
# --------------------------------------------------------------------------- #

DEST_TO_PREFERRED_DESTINATIONS = {
    # Beach / coastal cities
    "Atlantic City":   ["Beach Resorts", "Coastal Areas", "Major Cities",
                         "Luxury Resorts", "Music Festivals"],
    "Charleston":      ["Historic Cities", "Historical Sites", "Coastal Areas",
                         "Beach Resorts"],
    "Fort Lauderdale": ["Beach Resorts", "Coastal Areas", "Luxury Resorts",
                         "Major Cities"],
    "Honolulu":        ["Beach Resorts", "Exotic Islands", "Coastal Areas",
                         "Luxury Resorts", "Major Cities"],
    "Jacksonville":    ["Beach Resorts", "Coastal Areas", "Major Cities"],
    "Kahului":         ["Beach Resorts", "Exotic Islands", "Coastal Areas",
                         "Luxury Resorts", "Remote Locations"],
    "Miami":           ["Beach Resorts", "Major Cities", "Coastal Areas",
                         "Luxury Resorts", "Music Festivals"],
    "Norfolk":         ["Coastal Areas", "Historic Cities", "Major Cities"],
    "Pensacola":       ["Beach Resorts", "Coastal Areas", "Historical Sites"],
    "Punta Gorda":     ["Beach Resorts", "Coastal Areas", "Quiet Countryside"],
    "Savannah":        ["Historic Cities", "Historical Sites", "Coastal Areas",
                         "Quiet Countryside"],
    "Tampa":           ["Beach Resorts", "Coastal Areas", "Major Cities",
                         "Family-friendly Resorts"],
    "Wilmington":      ["Coastal Areas", "Historic Cities", "Beach Resorts"],

    # Caribbean territories
    "Charlotte Amalie":["Caribbean Islands", "Beach Resorts", "Coastal Areas",
                         "Exotic Islands"],
    "Ponce":           ["Caribbean Islands", "Beach Resorts", "Coastal Areas",
                         "Historic Cities"],
    "San Juan":        ["Caribbean Islands", "Beach Resorts", "Historic Cities",
                         "Coastal Areas"],

    # Nature / mountain gateway cities
    "Billings":        ["National Parks", "Quiet Countryside", "Remote Locations"],
    "Boise":           ["National Parks", "Quiet Countryside", "Remote Locations"],
    "Bozeman":         ["National Parks", "Quiet Countryside", "Remote Locations"],
    "Cheyenne":        ["National Parks", "Quiet Countryside", "Remote Locations"],
    "Colorado Springs":["National Parks", "Quiet Countryside", "Remote Locations"],
    "Eugene":          ["National Parks", "Quiet Countryside", "Coastal Areas",
                         "Music Festivals"],
    "Grand Junction":  ["National Parks", "Remote Locations", "Quiet Countryside"],
    "Great Falls":     ["National Parks", "Remote Locations", "Quiet Countryside"],
    "Gunnison":        ["National Parks", "Remote Locations", "Quiet Countryside"],
    "Helena":          ["National Parks", "Quiet Countryside", "Historic Cities",
                         "Remote Locations"],
    "Laramie":         ["National Parks", "Quiet Countryside", "Remote Locations"],
    "Medford":         ["National Parks", "Quiet Countryside", "Remote Locations"],
    "Traverse City":   ["Quiet Countryside", "Coastal Areas", "Remote Locations"],

    # Major cities
    "Atlanta":         ["Major Cities", "Historic Cities", "Business Hubs",
                         "Music Festivals"],
    "Austin":          ["Major Cities", "Music Festivals", "Tech Conferences"],
    "Baltimore":       ["Major Cities", "Historic Cities", "Coastal Areas",
                         "Historical Sites"],
    "Boston":          ["Major Cities", "Historic Cities", "Historical Sites",
                         "Business Hubs", "Coastal Areas"],
    "Charlotte":       ["Major Cities", "Business Hubs", "Historic Cities"],
    "Chicago":         ["Major Cities", "Historic Cities", "Business Hubs",
                         "Music Festivals", "Design Capitals"],
    "Cincinnati":      ["Major Cities", "Historic Cities", "Business Hubs"],
    "Cleveland":       ["Major Cities", "Historic Cities", "Music Festivals"],
    "Columbus":        ["Major Cities", "Historic Cities", "Business Hubs"],
    "Dallas":          ["Major Cities", "Business Hubs", "Music Festivals"],
    "Denver":          ["Major Cities", "National Parks", "Business Hubs"],
    "Detroit":         ["Major Cities", "Historic Cities", "Music Festivals",
                         "Design Capitals"],
    "El Paso":         ["Major Cities", "Historic Cities"],
    "Houston":         ["Major Cities", "Business Hubs", "Music Festivals"],
    "Indianapolis":    ["Major Cities", "Music Festivals", "Business Hubs"],
    "Kansas City":     ["Major Cities", "Music Festivals", "Historic Cities"],
    "Las Vegas":       ["Major Cities", "Luxury Resorts", "Music Festivals"],
    "Los Angeles":     ["Major Cities", "Beach Resorts", "Coastal Areas",
                         "Music Festivals", "Design Capitals"],
    "Memphis":         ["Major Cities", "Music Festivals", "Historic Cities"],
    "Milwaukee":       ["Major Cities", "Music Festivals", "Coastal Areas"],
    "Minneapolis":     ["Major Cities", "Business Hubs", "Music Festivals"],
    "Nashville":       ["Major Cities", "Music Festivals", "Historic Cities"],
    "New Orleans":     ["Major Cities", "Music Festivals", "Historic Cities",
                         "Coastal Areas"],
    "New York":        ["Major Cities", "Business Hubs", "Historic Cities",
                         "Design Capitals", "Music Festivals"],
    "New York City":   ["Major Cities", "Business Hubs", "Historic Cities",
                         "Design Capitals", "Music Festivals"],
    "Newark":          ["Major Cities", "Business Hubs"],
    "Oakland":         ["Major Cities", "Coastal Areas", "Tech Conferences"],
    "Oklahoma City":   ["Major Cities", "Historic Cities", "Music Festivals"],
    "Orlando":         ["Family-friendly Resorts", "Major Cities", "Luxury Resorts"],
    "Philadelphia":    ["Major Cities", "Historic Cities", "Historical Sites",
                         "Business Hubs"],
    "Phoenix":         ["Major Cities", "National Parks", "Luxury Resorts"],
    "Portland":        ["Major Cities", "Coastal Areas", "Tech Conferences",
                         "Music Festivals", "Design Capitals"],
    "Reno":            ["Major Cities", "National Parks", "Luxury Resorts",
                         "Music Festivals"],
    "Sacramento":      ["Major Cities", "Business Hubs"],
    "Salt Lake City":  ["Major Cities", "National Parks", "Tech Conferences"],
    "San Diego":       ["Coastal Areas", "Major Cities", "Beach Resorts",
                         "Luxury Resorts"],
    "San Francisco":   ["Major Cities", "Tech Conferences", "Business Hubs",
                         "Coastal Areas", "Design Capitals"],
    "San Jose":        ["Major Cities", "Tech Conferences", "Business Hubs"],
    "Santa Ana":       ["Major Cities", "Coastal Areas",
                         "Family-friendly Resorts"],
    "Seattle":         ["Major Cities", "Coastal Areas", "Tech Conferences",
                         "Business Hubs", "Music Festivals"],
    "St. Louis":       ["Major Cities", "Music Festivals", "Historic Cities"],
    "Tallahassee":     ["Historic Cities", "Historical Sites", "Quiet Countryside"],
    "Tucson":          ["Major Cities", "National Parks", "Quiet Countryside"],
    "Tulsa":           ["Major Cities", "Music Festivals", "Historic Cities"],
    "Washington":      ["Major Cities", "Historic Cities", "Historical Sites",
                         "Business Hubs"],

    # Historic / cultural cities
    "Charlottesville": ["Historic Cities", "Historical Sites", "Quiet Countryside"],
    "Knoxville":       ["Historic Cities", "Quiet Countryside", "Music Festivals"],
    "Louisville":      ["Historic Cities", "Music Festivals", "Quiet Countryside"],
    "Montgomery":      ["Historic Cities", "Historical Sites"],
    "Trenton":         ["Historic Cities", "Historical Sites"],

    # Quiet / countryside / smaller
    "Akron":           ["Quiet Countryside", "Historic Cities"],
    "Appleton":        ["Quiet Countryside", "Remote Locations"],
    "Buffalo":         ["Quiet Countryside", "Historic Cities", "Coastal Areas"],
    "Des Moines":      ["Quiet Countryside", "Major Cities"],
    "Escanaba":        ["Quiet Countryside", "Remote Locations"],
    "Fayetteville":    ["Quiet Countryside", "Historic Cities"],
    "Fresno":          ["Quiet Countryside", "National Parks"],
    "Huntsville":      ["Quiet Countryside", "Tech Conferences"],
    "Johnstown":       ["Quiet Countryside", "Historical Sites"],
    "Ketchikan":       ["Coastal Areas", "Remote Locations", "Quiet Countryside"],
    "La Crosse":       ["Quiet Countryside", "Coastal Areas"],
    "Manchester":      ["Quiet Countryside", "Historic Cities"],
    "Minot":           ["Quiet Countryside", "Remote Locations"],
    "Moline":          ["Quiet Countryside"],
    "Mosinee":         ["Quiet Countryside", "Remote Locations"],
    "Ogdensburg":      ["Quiet Countryside", "Remote Locations"],
    "Plattsburgh":     ["Quiet Countryside", "Remote Locations"],
    "Roanoke":         ["Quiet Countryside", "Historic Cities"],
    "Rochester":       ["Quiet Countryside", "Historic Cities", "Tech Conferences"],
    "Syracuse":        ["Quiet Countryside", "Historic Cities"],

    # State-level destinations
    "Arizona":         ["National Parks", "Major Cities", "Quiet Countryside",
                         "Remote Locations"],
    "California":      ["Major Cities", "Coastal Areas", "Tech Conferences",
                         "Design Capitals", "Beach Resorts"],
    "Colorado":        ["National Parks", "Remote Locations", "Quiet Countryside",
                         "Luxury Resorts"],
    "Florida":         ["Beach Resorts", "Coastal Areas",
                         "Family-friendly Resorts", "Major Cities"],
    "Georgia":         ["Historic Cities", "Coastal Areas", "Quiet Countryside",
                         "Music Festivals"],
    "Hawaii":          ["Beach Resorts", "Exotic Islands", "Coastal Areas",
                         "Luxury Resorts", "Remote Locations"],
    "Idaho":           ["National Parks", "Remote Locations", "Quiet Countryside"],
    "Illinois":        ["Major Cities", "Quiet Countryside", "Music Festivals"],
    "Indiana":         ["Quiet Countryside", "Major Cities"],
    "Iowa":            ["Quiet Countryside"],
    "Kentucky":        ["Historic Cities", "Music Festivals", "Quiet Countryside"],
    "Louisiana":       ["Coastal Areas", "Music Festivals", "Historic Cities",
                         "Major Cities"],
    "Massachusetts":   ["Historic Cities", "Historical Sites", "Coastal Areas",
                         "Major Cities"],
    "Michigan":        ["Quiet Countryside", "Coastal Areas", "Major Cities",
                         "Music Festivals"],
    "Minnesota":       ["Quiet Countryside", "Major Cities", "Remote Locations"],
    "Missouri":        ["Quiet Countryside", "Major Cities", "Music Festivals"],
    "Montana":         ["National Parks", "Quiet Countryside", "Remote Locations"],
    "Nebraska":        ["Quiet Countryside", "Major Cities"],
    "North Carolina":  ["Coastal Areas", "Historic Cities", "Quiet Countryside",
                         "Beach Resorts"],
    "North Dakota":    ["Quiet Countryside", "Remote Locations"],
    "Ohio":            ["Major Cities", "Historic Cities", "Quiet Countryside"],
    "Oregon":          ["National Parks", "Coastal Areas", "Quiet Countryside",
                         "Music Festivals"],
    "Pennsylvania":    ["Historic Cities", "Historical Sites", "Quiet Countryside",
                         "Major Cities"],
    "South Carolina":  ["Coastal Areas", "Historic Cities", "Beach Resorts"],
    "Tennessee":       ["Historic Cities", "Music Festivals", "National Parks",
                         "Quiet Countryside"],
    "Texas":           ["Major Cities", "Music Festivals", "Historic Cities",
                         "Business Hubs"],
    "Utah":            ["National Parks", "Remote Locations", "Quiet Countryside"],
    "Virginia":        ["Historic Cities", "Historical Sites", "Coastal Areas",
                         "National Parks"],
    "Wisconsin":       ["Quiet Countryside", "Coastal Areas", "Music Festivals"],
}


# --------------------------------------------------------------------------- #
# Source-record dataclass
# --------------------------------------------------------------------------- #

@dataclass
class Source:
    sid:          int               # within-query sequence number
    kind:         str               # "local_constraint" | "destination" |
                                    # "atomic_top" | "atomic_nested" |
                                    # "numeric_top"
    paradigm:     str | None        # bank paradigm; None for lc / destination
    bank_id:      int | None
    op_name:      str | None        # only for TemporalPreference
    path:         tuple             # slot path within parent template
    raw_value:    Any               # the value(s) carried by this source
    field:        str               # profile field this contributes to
    trait_table:  str               # attribute name in profile_traits
    trait_inversion_table: str      # inversion attribute name
    trait_key:    Any               # key into the trait dict

    # filled in after picking
    variant_text: str | None = None
    variant_idx:  int = -1
    drifted:      bool = False
    drift_action: str | None = None  # "drop" | "invert"


# --------------------------------------------------------------------------- #
# Atomic-position walker — yields (path, atomic_dict) for every atomic
# predicate location in a paradigm template.
# --------------------------------------------------------------------------- #

def _atomic_fields(d: Any) -> dict | None:
    """Return the atomic-fields dict if ``d`` represents an
    AtomicPreference, else None.

    AtomicPreference appears in two structural shapes in this corpus:
      * **flat**     ``{"class": "AtomicPreference", "entity_type": ..., "attribute": ..., ...}``
      * **wrapped**  ``{"class": "AtomicPreference", "template": {"entity_type": ..., ...}}``

    Direct (top-level) paradigms (Composite / Conditional / Lex /
    Compensatory / Temporal) use the flat shape for their sub-preds.
    ScopedPreference's inner predicates use the wrapped shape — including
    when the inner is itself Composite / Temporal / etc., so the wrapped
    shape can be nested arbitrarily deep.  This helper accepts both."""
    if not isinstance(d, dict):
        return None
    cls = d.get("class")
    if cls is not None and cls != "AtomicPreference":
        return None
    if "attribute" in d:
        return d
    tmpl = d.get("template")
    if isinstance(tmpl, dict) and "attribute" in tmpl:
        return tmpl
    return None


def _inner_template(d: Any) -> dict:
    """Return the inner paradigm-template dict for a wrapped sub-pref
    of any non-Atomic class — ``{class, template:{...}}`` -> the inner
    template; flat dicts pass through unchanged."""
    if not isinstance(d, dict):
        return {}
    if "class" in d and isinstance(d.get("template"), dict):
        return d["template"]
    return d


def _walk_atomic_positions(paradigm: str, template: dict) -> list[tuple[tuple, dict]]:
    """Return a list of (slot_path, atomic_dict) tuples — every atomic
    predicate location inside ``template`` for the given paradigm.

    The slot_path matches what ``profile_traits.quartile_at`` expects on
    the parent's ``example_values.default``.  The atomic_dict is always
    the unwrapped fields-dict (entity_type / attribute / op / value /
    scope at the top level).
    """
    out: list[tuple[tuple, dict]] = []
    t = template or {}

    if paradigm == "AtomicPreference":
        a = _atomic_fields(t)
        if a is not None:
            out.append(((), a))
        return out

    if paradigm == "CompositePreference":
        for i, ch in enumerate(t.get("children", []) or []):
            a = _atomic_fields(ch)
            if a is not None:
                out.append((("children", i), a))
        return out

    if paradigm == "ConditionalPreference":
        cond_a = _atomic_fields(t.get("condition"))
        then_a = _atomic_fields(t.get("then_pref"))
        if cond_a is not None:
            out.append((("condition",), cond_a))
        if then_a is not None:
            out.append((("then_pref",), then_a))
        return out

    if paradigm == "LexicographicPreference":
        for i, p in enumerate(t.get("preferences", []) or []):
            a = _atomic_fields(p)
            if a is not None:
                out.append((("preferences", i), a))
        return out

    if paradigm == "CompensatoryPreference":
        for slot in ("primary_ap", "margin_ap", "secondary_ap"):
            a = _atomic_fields(t.get(slot))
            if a is not None:
                out.append(((slot,), a))
        return out

    if paradigm == "TemporalPreference":
        for slot in ("subject_ap", "reference_ap"):
            a = _atomic_fields(t.get(slot))
            if a is not None:
                out.append(((slot,), a))
        return out

    if paradigm == "ScopedPreference":
        # Scoped wraps an inner predicate at ``inner.template``.  The
        # inner may itself be Atomic, Composite, Conditional,
        # Lexicographic, Compensatory, or Temporal — recurse so nested
        # atomic positions surface with a path prefixed by ``"inner"``.
        # NumericPreference inners are handled directly in
        # ``disintegrate`` (they don't decompose into atomic positions).
        # scope_filters are Day-attribute predicates and do not feed
        # into profile traits.
        inner = t.get("inner") or {}
        inner_cls = inner.get("class")
        if inner_cls == "AtomicPreference":
            a = _atomic_fields(inner)
            if a is not None:
                out.append((("inner",), a))
        elif inner_cls and inner_cls != "NumericPreference":
            inner_t = _inner_template(inner)
            sub_positions = _walk_atomic_positions(inner_cls, inner_t)
            for sub_path, atomic in sub_positions:
                out.append((("inner",) + sub_path, atomic))
        return out

    return out


# --------------------------------------------------------------------------- #
# Per-atomic router — emits Source records for one atomic predicate
# --------------------------------------------------------------------------- #

def _emit_sources_for_atomic(
    *,
    atomic:        dict,
    paradigm:      str,
    bank_id:       int | None,
    op_name:       str | None,
    path:          tuple,
    bank_example:  dict,
    is_top_level:  bool,
    sid_box:       list,
) -> list[Source]:
    """Translate one atomic predicate into Source records.  Handles the
    per-element expansion rule for top-level atomic list-valued sources
    vs. single-signal nested sources."""
    ent  = atomic.get("entity_type")
    attr = atomic.get("attribute")
    op   = atomic.get("op")
    val  = atomic.get("value")

    out: list[Source] = []

    def _next_sid() -> int:
        n = sid_box[0]
        sid_box[0] = n + 1
        return n

    def _add(*, raw_value, field, trait_table, inv_table, trait_key,
             kind_override=None):
        out.append(Source(
            sid=_next_sid(),
            kind=kind_override or ("atomic_top" if is_top_level else "atomic_nested"),
            paradigm=paradigm,
            bank_id=bank_id,
            op_name=op_name,
            path=path,
            raw_value=raw_value,
            field=field,
            trait_table=trait_table,
            trait_inversion_table=inv_table,
            trait_key=trait_key,
        ))

    # Restaurant.cuisine ---------------------------------------------------
    # Set-valued sub-preds expand per element regardless of whether the
    # parent paradigm is top-level Atomic or a non-atomic wrapper.  This
    # makes drift consistency natural: each cuisine becomes its own
    # source / drift unit, and multiple paradigms touching the same
    # cuisine value share a (trait_table, trait_key) drift group.
    if (ent, attr) == ("Restaurant", "cuisine"):
        vs = val if isinstance(val, list) else ([val] if val else [])
        if op in ("in", "=="):
            table, inv, fld = "CUISINE_PALATE", "CUISINE_PALATE_INVERSIONS", "Food and Dining Preferences"
        elif op in ("not_in", "!="):
            table, inv, fld = "CUISINE_PALATE_INVERSIONS", "CUISINE_PALATE", "Dislikes"
        else:
            return out
        for v in vs:
            if v and v in PT.CUISINE_PALATE:
                _add(raw_value=v, field=fld, trait_table=table,
                     inv_table=inv, trait_key=v)
        return out

    # Accommodation.house_rules --------------------------------------------
    # local_constraint listing of a bare token (e.g. "smoking") encodes
    # tolerance — the user is OK with smoking-permitted accommodations.
    # In preferences the value arrives as "No smoking" and the semantic
    # flips depending on op:
    #   * ``in / ==``    => user wants no-smoking enforced (averse)
    #   * ``not_in / !=``=> user OK with smoking allowed (tolerant)
    # Both polarities ultimately use the SAME trait_key (the bare token);
    # the trait_table differs (HOUSE_RULE_TRAITS for tolerant,
    # HOUSE_RULE_TRAITS_INVERSIONS for averse).  Same value with
    # different polarities lands in different drift groups, which is
    # the desired behavior (they encode opposite profile signals).
    if (ent, attr) == ("Accommodation", "house_rules"):
        vs = val if isinstance(val, list) else ([val] if val else [])
        for raw_v in vs:
            if not isinstance(raw_v, str):
                continue
            v = raw_v.strip()
            has_no_prefix = v.lower().startswith("no ")
            bare = v[3:] if has_no_prefix else v
            if bare not in PT.HOUSE_RULE_TRAITS:
                continue
            # Determine semantic: tolerant vs averse
            if has_no_prefix:
                averse = op in ("in", "==")
            else:
                averse = op in ("not_in", "!=")
            if averse:
                table = "HOUSE_RULE_TRAITS_INVERSIONS"
                inv   = "HOUSE_RULE_TRAITS"
            else:
                table = "HOUSE_RULE_TRAITS"
                inv   = "HOUSE_RULE_TRAITS_INVERSIONS"
            _add(raw_value=raw_v, field="Lifestyle",
                 trait_table=table, inv_table=inv,
                 trait_key=bare)
        return out

    # Accommodation.room_type ----------------------------------------------
    # Both polarities of room_type variants read as Travel Style
    # statements ("Values an entire unit" / "Doesn't need a whole unit;
    # comfortable sharing").  Routed uniformly to Travel Style.
    if (ent, attr) == ("Accommodation", "room_type"):
        v = val[0] if isinstance(val, list) and val else val
        if not v:
            return out
        k = ROOM_TYPE_NORMALIZE.get(v) or ROOM_TYPE_NORMALIZE.get(v.lower() if isinstance(v, str) else v)
        if not k or k not in PT.ROOM_TYPE_TRAITS:
            return out
        if op in ("==", "in"):
            _add(raw_value=v, field="Travel Style",
                 trait_table="ROOM_TYPE_TRAITS",
                 inv_table="ROOM_TYPE_TRAITS_INVERSIONS",
                 trait_key=k)
        elif op in ("!=", "not_in"):
            _add(raw_value=v, field="Travel Style",
                 trait_table="ROOM_TYPE_TRAITS_INVERSIONS",
                 inv_table="ROOM_TYPE_TRAITS",
                 trait_key=k)
        return out

    # Transportation.mode --------------------------------------------------
    # Per-element expansion: each mode in a multi-mode list becomes its
    # own source.  Positive selection (==/in) routes through the inverse
    # of the opposite mode's avoidance trait; avoidance (not_in/!=)
    # routes through the matching avoidance trait directly.  Modes
    # without a clean trait anchor (e.g. "taxi") are dropped.
    if (ent, attr) == ("Transportation", "mode"):
        vs = val if isinstance(val, list) else ([val] if val else [])
        if op in ("==", "in"):
            table, inv = "TRANSPORT_TRAITS_INVERSIONS", "TRANSPORT_TRAITS"
            mapper = TRANSPORT_FROM_RAW_POSITIVE
        elif op in ("not_in", "!="):
            table, inv = "TRANSPORT_TRAITS", "TRANSPORT_TRAITS_INVERSIONS"
            mapper = TRANSPORT_FROM_RAW_AVOIDANCE
        else:
            return out
        for v in vs:
            k = mapper.get(v)
            if k:
                _add(raw_value=v, field="Lifestyle",
                     trait_table=table, inv_table=inv,
                     trait_key=k)
        return out

    # Attraction.category --------------------------------------------------
    # Same per-element expansion rule as cuisine.
    if (ent, attr) == ("Attraction", "category"):
        vs = val if isinstance(val, list) else ([val] if val else [])
        if op in ("in", "=="):
            table, inv, fld = "ATTRACTION_CATEGORY_TRAITS", "ATTRACTION_CATEGORY_TRAITS_INVERSIONS", "Hobbies"
        elif op in ("not_in", "!="):
            table, inv, fld = "ATTRACTION_CATEGORY_TRAITS_INVERSIONS", "ATTRACTION_CATEGORY_TRAITS", "Dislikes"
        else:
            return out
        for v in vs:
            if v and v in PT.ATTRACTION_CATEGORY_TRAITS:
                _add(raw_value=v, field=fld, trait_table=table,
                     inv_table=inv, trait_key=v)
        return out

    # Numeric atomic (cost / rating, op >= / <=) ---------------------------
    # All cost/rating numerics describe HOW the user travels (budget vs
    # premium, quality-demanding vs flexible) regardless of entity.
    # Routed uniformly to Travel Style.  Cuisine handles the orthogonal
    # food-preference axis.
    if attr in ("cost", "rating") and op in (">=", "<=") and ent:
        quartile = PT.quartile_at(bank_example, *path)
        trait_key = (ent, attr, op, quartile)
        if trait_key not in PT.ATOMIC_NUMERIC_TRAITS:
            return out
        _add(raw_value=val, field="Travel Style",
             trait_table="ATOMIC_NUMERIC_TRAITS",
             inv_table="ATOMIC_NUMERIC_TRAITS_INVERSIONS",
             trait_key=trait_key)
        return out

    return out


# --------------------------------------------------------------------------- #
# disintegrate(rec, bank_idx) -> [Source, ...]
# --------------------------------------------------------------------------- #

def disintegrate(rec: dict, bank_idx: dict) -> list[Source]:
    sources: list[Source] = []
    sid_box = [0]

    def _next_sid() -> int:
        n = sid_box[0]
        sid_box[0] = n + 1
        return n

    # 1. local_constraint
    lc = rec.get("local_constraint") or {}

    if lc.get("cuisine"):
        cuisines = lc["cuisine"] if isinstance(lc["cuisine"], list) else [lc["cuisine"]]
        for c in cuisines:
            if c in PT.CUISINE_PALATE:
                sources.append(Source(
                    sid=_next_sid(), kind="local_constraint",
                    paradigm=None, bank_id=None, op_name=None,
                    path=("cuisine",), raw_value=c,
                    field="Food and Dining Preferences",
                    trait_table="CUISINE_PALATE",
                    trait_inversion_table="CUISINE_PALATE_INVERSIONS",
                    trait_key=c))

    if lc.get("house rule"):
        k = PT.HOUSE_RULE_KEY(lc["house rule"])
        if k in PT.HOUSE_RULE_TRAITS:
            sources.append(Source(
                sid=_next_sid(), kind="local_constraint",
                paradigm=None, bank_id=None, op_name=None,
                path=("house rule",), raw_value=lc["house rule"],
                field="Lifestyle",
                trait_table="HOUSE_RULE_TRAITS",
                trait_inversion_table="HOUSE_RULE_TRAITS_INVERSIONS",
                trait_key=k))

    if lc.get("room type"):
        rt = lc["room type"]
        k = ROOM_TYPE_NORMALIZE.get(rt) or ROOM_TYPE_NORMALIZE.get(rt.lower() if isinstance(rt, str) else rt)
        if k and k in PT.ROOM_TYPE_TRAITS:
            sources.append(Source(
                sid=_next_sid(), kind="local_constraint",
                paradigm=None, bank_id=None, op_name=None,
                path=("room type",), raw_value=rt,
                field="Travel Style",
                trait_table="ROOM_TYPE_TRAITS",
                trait_inversion_table="ROOM_TYPE_TRAITS_INVERSIONS",
                trait_key=k))

    if lc.get("transportation"):
        t = lc["transportation"]
        k = TRANSPORT_FROM_RAW_AVOIDANCE.get(t) or TRANSPORT_FROM_RAW_AVOIDANCE.get(
            t.strip().lower() if isinstance(t, str) else t)
        if k in PT.TRANSPORT_TRAITS:
            sources.append(Source(
                sid=_next_sid(), kind="local_constraint",
                paradigm=None, bank_id=None, op_name=None,
                path=("transportation",), raw_value=t,
                field="Lifestyle",
                trait_table="TRANSPORT_TRAITS",
                trait_inversion_table="TRANSPORT_TRAITS_INVERSIONS",
                trait_key=k))

    # 2. preferences
    for p in rec.get("preferences", []) or []:
        paradigm = p.get("paradigm")
        bid      = p.get("bank_id")
        tmpl     = p.get("template") or {}

        # NumericPreference -> top-level direction trait
        # All NumericPreference variants describe travel-style stance
        # (budget-leaning vs premium-leaning portfolio shape), routed
        # uniformly to Travel Style.
        if paradigm == "NumericPreference":
            ent       = tmpl.get("entity_type")
            attr      = tmpl.get("attribute")
            direction = tmpl.get("direction")
            agg       = tmpl.get("aggregation")
            tkey      = (ent, attr, direction)
            if tkey in PT.NUMERIC_DIRECTION_TRAITS:
                sources.append(Source(
                    sid=_next_sid(), kind="numeric_top",
                    paradigm=paradigm, bank_id=bid, op_name=None,
                    path=(), raw_value={
                        "entity": ent, "attribute": attr,
                        "direction": direction, "aggregation": agg,
                        "threshold": tmpl.get("threshold"),
                    },
                    field="Travel Style",
                    trait_table="NUMERIC_DIRECTION_TRAITS",
                    trait_inversion_table="NUMERIC_DIRECTION_TRAITS_INVERSIONS",
                    trait_key=tkey))
            continue

        # ScopedPreference wrapping NumericPreference inner: emit a
        # numeric_top source for the inner and skip the atomic walk.
        if paradigm == "ScopedPreference":
            inner = tmpl.get("inner") or {}
            if inner.get("class") == "NumericPreference":
                inner_t = inner.get("template") or {}
                ent       = inner_t.get("entity_type")
                attr      = inner_t.get("attribute")
                direction = inner_t.get("direction")
                agg       = inner_t.get("aggregation")
                tkey      = (ent, attr, direction)
                if tkey in PT.NUMERIC_DIRECTION_TRAITS:
                    sources.append(Source(
                        sid=_next_sid(), kind="numeric_top",
                        paradigm=paradigm, bank_id=bid, op_name=None,
                        path=("inner",), raw_value={
                            "entity": ent, "attribute": attr,
                            "direction": direction, "aggregation": agg,
                            "threshold": inner_t.get("threshold"),
                        },
                        field="Travel Style",
                        trait_table="NUMERIC_DIRECTION_TRAITS",
                        trait_inversion_table="NUMERIC_DIRECTION_TRAITS_INVERSIONS",
                        trait_key=tkey))
                continue

        # All other paradigms: fetch parent bank example for quartile lookup
        if paradigm == "TemporalPreference":
            op_name = tmpl.get("op")
            bank_entry = bank_idx.get((paradigm, op_name, bid))
        else:
            op_name = None
            bank_entry = bank_idx.get((paradigm, bid))
        bank_example = (bank_entry or {}).get("example_values", {}).get("default", {})

        positions = _walk_atomic_positions(paradigm, tmpl)
        is_top_level = (paradigm == "AtomicPreference")

        for path, atomic in positions:
            new_sources = _emit_sources_for_atomic(
                atomic=atomic, paradigm=paradigm, bank_id=bid,
                op_name=op_name, path=path, bank_example=bank_example,
                is_top_level=is_top_level, sid_box=sid_box,
            )
            sources.extend(new_sources)

    # 3. destination -> Preferred Destinations anchor
    # The destination maps to up to 5 candidate archetypes; sample ONE
    # deterministically per query via pick_variant on the candidate list
    # so the choice is reproducible per qid but distributes across the
    # corpus.
    dest = rec.get("dest")
    qid  = rec.get("query_id")
    candidates_all = DEST_TO_PREFERRED_DESTINATIONS.get(dest, [])
    candidates = [c for c in candidates_all
                  if c in PT.PREFERRED_DESTINATION_PHRASING]
    if candidates:
        chosen, _ = PT.pick_variant(candidates, qid, "destination_archetype",
                                    dest, tuple(candidates))
        sources.append(Source(
            sid=_next_sid(), kind="destination",
            paradigm=None, bank_id=None, op_name=None,
            path=("dest",),
            raw_value={"dest":       dest,
                       "candidates": list(candidates),
                       "sampled":    chosen},
            field="Preferred Destinations",
            trait_table="PREFERRED_DESTINATION_PHRASING",
            trait_inversion_table="PREFERRED_DESTINATION_PHRASING_INVERSIONS",
            trait_key=chosen))

    return sources


# --------------------------------------------------------------------------- #
# Trait-variant assignment
# --------------------------------------------------------------------------- #

def _seed_components(query_id: int, src: Source) -> tuple:
    """Stable per-source descriptor used as the variant-picker seed."""
    return (
        query_id,
        src.kind,
        src.paradigm,
        src.bank_id,
        src.op_name,
        src.path,
        src.trait_table,
        repr(src.trait_key),
    )


def assign_variant(src: Source, query_id: int, use_inversion: bool = False) -> None:
    """Pick a variant deterministically; fill in variant_text / idx."""
    if use_inversion:
        table = getattr(PT, src.trait_inversion_table, {})
        salt  = "INVERTED"
    else:
        table = getattr(PT, src.trait_table, {})
        salt  = "ALIGNED"
    variants = table.get(src.trait_key, [])
    text, idx = PT.pick_variant(variants, *_seed_components(query_id, src), salt)
    src.variant_text = text or None
    src.variant_idx  = idx


# --------------------------------------------------------------------------- #
# Drift assignment + application
# --------------------------------------------------------------------------- #

DRIFT_RATIOS = {"aligned": 0.30, "omission": 0.35, "inversion": 0.35}

# Per-query drift count is sampled deterministically in this fraction range
# of the eligible drift-group count (rounded, floored at 1).  Wider trips
# (multi-user / hard tier) have more eligible groups and so naturally
# receive more drifts.
DRIFT_FRAC_MIN, DRIFT_FRAC_MAX = 0.25, 0.50


def assign_drift_modes(records: list[dict]) -> dict[int, str]:
    """Return a deterministic mapping query_id -> drift mode,
    stratified so each ``level`` (easy/medium/hard) has exactly
    30 / 35 / 35 split (modulo rounding)."""
    by_level: dict[str, list[int]] = defaultdict(list)
    for r in records:
        by_level[r.get("level", "easy")].append(r["query_id"])
    modes: dict[int, str] = {}
    for level, qids in by_level.items():
        # Deterministic rank by md5 of (level, query_id) so the choice is
        # stable but the bucketing is well-distributed within the tier.
        ranked = sorted(
            qids,
            key=lambda q: hashlib.md5(f"drift|{level}|{q}".encode()).hexdigest(),
        )
        n = len(ranked)
        n_aligned  = round(DRIFT_RATIOS["aligned"]  * n)
        n_omission = round(DRIFT_RATIOS["omission"] * n)
        for i, qid in enumerate(ranked):
            if i < n_aligned:
                modes[qid] = "aligned"
            elif i < n_aligned + n_omission:
                modes[qid] = "omission"
            else:
                modes[qid] = "inversion"
    return modes


def _drift_group_key(src: Source) -> tuple | None:
    """Canonical drift-group identifier.  Two sources belong to the same
    drift group iff they share both the trait table (polarity) and the
    trait key (value).  This makes drift consistent across paradigms:
    e.g. a cuisine "Italian" surfacing in both local_constraint and a
    Lex preference shares a single drift unit.  ``destination`` sources
    are excluded — they are part of the trip request, not preferences,
    and must never be dropped or inverted."""
    if src.kind == "destination":
        return None
    tk = src.trait_key
    if isinstance(tk, list):
        tk = tuple(tk)
    return (src.trait_table, tk)


def _drift_count_for(query_id: int, n_eligible_groups: int) -> int:
    """Sample a per-query drift count in ``[DRIFT_FRAC_MIN, DRIFT_FRAC_MAX)``
    of the eligible group count, deterministically by query_id.  Floors
    at 1 when there is at least one eligible group so omission /
    inversion modes always produce at least one drift action."""
    if n_eligible_groups <= 0:
        return 0
    h = hashlib.md5(f"drift-frac|{query_id}".encode()).digest()
    u = int.from_bytes(h[:4], "big") / 2**32                       # [0,1)
    frac = DRIFT_FRAC_MIN + u * (DRIFT_FRAC_MAX - DRIFT_FRAC_MIN)  # [0.25, 0.50)
    return max(1, round(frac * n_eligible_groups))


def _pick_drift_groups(sources: list[Source], query_id: int, mode: str
                       ) -> tuple[set[tuple], int, int]:
    """Pick a deterministic set of drift-group keys to drift this query.

    Returns ``(picked_set, drift_count, n_eligible_groups)``.  Eligible
    groups exclude destination sources.  ``picked_set`` is empty for
    ``aligned`` mode or when no eligible groups exist."""
    if mode == "aligned" or not sources:
        return set(), 0, 0
    groups: dict[tuple, list[Source]] = defaultdict(list)
    for s in sources:
        gk = _drift_group_key(s)
        if gk is None:
            continue
        groups[gk].append(s)
    n_eligible = len(groups)
    if n_eligible == 0:
        return set(), 0, 0
    k = _drift_count_for(query_id, n_eligible)
    ranked = sorted(
        groups.keys(),
        key=lambda g: hashlib.md5(
            f"drift-rank|{mode}|{query_id}|{g!r}".encode()
        ).hexdigest(),
    )
    return set(ranked[:k]), k, n_eligible


def apply_drift(sources: list[Source], query_id: int, mode: str) -> dict:
    """Mutate ``sources`` in-place to apply the drift.  All sources
    whose drift-group key is in the picked set get marked with the
    drift action — so cuisines appearing in both local_constraint and
    a Lex preference drift together.  Returns metadata about the picked
    groups and per-source actions."""
    picked, k, n_eligible = _pick_drift_groups(sources, query_id, mode)
    actions: list[dict] = []
    for s in sources:
        gk = _drift_group_key(s)
        if gk is None or gk not in picked:
            continue
        s.drifted = True
        if mode == "omission":
            s.drift_action = "drop"
        elif mode == "inversion":
            s.drift_action = "invert"
            assign_variant(s, query_id, use_inversion=True)
        actions.append({
            "sid":          s.sid,
            "group_key":    [s.trait_table,
                             list(s.trait_key) if isinstance(s.trait_key, tuple)
                             else s.trait_key],
            "action":       s.drift_action,
            "field":        s.field,
        })
    return {
        "mode":              mode,
        "drift_count":       k,
        "eligible_groups":   n_eligible,
        "picked_groups":     [
            [tab, list(key) if isinstance(key, tuple) else key]
            for tab, key in picked
        ],
        "actions":           actions,
    }


# --------------------------------------------------------------------------- #
# Profile assembly
# --------------------------------------------------------------------------- #

PROFILE_INTERESTS_FIELDS = (
    "Hobbies", "Lifestyle", "Travel Style",
    "Preferred Destinations", "Food and Dining Preferences",
    "Dislikes",
)

# Canonical profile-field placement, keyed by the trait table actually
# used to pick the variant.  Drift inversion swaps the source from its
# aligned table to its inversion table; the field follows the variant
# (a positive-toned variant lands in the positive field, a Dislike-toned
# variant lands in Dislikes).
#
# Placement rationale:
#   * Cuisine                     — Food and Dining Preferences |  Dislikes
#       (positive cuisines describe the user's palate; avoidances are
#        real food dislikes)
#   * House rule                  — Lifestyle | Lifestyle
#       (both polarities describe a personal lifestyle facet: the user
#        smokes / has pets / values privacy)
#   * Room type                   — Travel Style | Travel Style
#       (inversion variants like "doesn't need a whole unit; comfortable
#        sharing" are travel-style preferences, not real dislikes)
#   * Transport mode              — Lifestyle | Lifestyle
#       (eco-conscious / road-tripper traits)
#   * Attraction category         — Hobbies | Dislikes
#       (positive categories are hobbies; avoidance variants read as
#        Dislikes)
#   * Numeric atomic & top-level  — Travel Style | Travel Style
#       (budget / luxury / quality cues describe HOW the user travels;
#        cuisine handles the WHAT side)
#   * Preferred destinations      — Preferred Destinations
_TABLE_FIELD_OVERRIDES = {
    "CUISINE_PALATE":                            "Food and Dining Preferences",
    "CUISINE_PALATE_INVERSIONS":                 "Dislikes",
    "HOUSE_RULE_TRAITS":                         "Lifestyle",
    "HOUSE_RULE_TRAITS_INVERSIONS":              "Lifestyle",
    "ROOM_TYPE_TRAITS":                          "Travel Style",
    "ROOM_TYPE_TRAITS_INVERSIONS":               "Travel Style",
    "TRANSPORT_TRAITS":                          "Lifestyle",
    "TRANSPORT_TRAITS_INVERSIONS":               "Lifestyle",
    "ATTRACTION_CATEGORY_TRAITS":                "Hobbies",
    "ATTRACTION_CATEGORY_TRAITS_INVERSIONS":     "Dislikes",
    "ATOMIC_NUMERIC_TRAITS":                     "Travel Style",
    "ATOMIC_NUMERIC_TRAITS_INVERSIONS":          "Travel Style",
    "NUMERIC_DIRECTION_TRAITS":                  "Travel Style",
    "NUMERIC_DIRECTION_TRAITS_INVERSIONS":       "Travel Style",
    "PREFERRED_DESTINATION_PHRASING":            "Preferred Destinations",
    "PREFERRED_DESTINATION_PHRASING_INVERSIONS": "Preferred Destinations",
}


def _effective_field(src: Source) -> str:
    """Return the profile field where the source's currently-chosen
    variant should land — based on which trait table the variant came
    from.  Aligned variants come from ``src.trait_table``; inverted
    variants (under drift) come from ``src.trait_inversion_table``."""
    used_table = (src.trait_inversion_table
                  if src.drift_action == "invert"
                  else src.trait_table)
    return _TABLE_FIELD_OVERRIDES.get(used_table, src.field)


def _pets_from_sources(sources: list[Source]) -> str | None:
    """Pets line: derived exclusively from a `pets` house-rule signal in
    the query's local_constraint.  Behaviour:
      - Aligned pet-tolerant signal   -> use the aligned variant text
        drawn from HOUSE_RULE_TRAITS["pets"] via that source's
        ``variant_text``.
      - Inverted (averse) signal      -> use the inversion variant text
        from HOUSE_RULE_TRAITS_INVERSIONS["pets"].
      - Dropped (omission drift)      -> None (line is omitted).
      - No pets signal in the query   -> None (line is omitted).
    We never fabricate a "No pets." claim when the query is silent about
    pets; silence in the query becomes silence in the profile.
    """
    aligned_text: str | None = None
    inverted_text: str | None = None
    for s in sources:
        if s.trait_key != "pets" or s.drift_action == "drop":
            continue
        text = s.variant_text
        if not text:
            continue
        if s.drift_action == "invert":
            inverted_text = text
        else:
            aligned_text = text
    return inverted_text or aligned_text  # inversion wins if both somehow set


def build_profile(rec: dict, sources: list[Source], drift_mode: str) -> dict:
    profile = {
        "Demographics": {
            "Age Range":    "TBD",
            "Gender":       "TBD",
            "Income Level": "TBD",
            "Location":     rec.get("org", "TBD"),
            "Education":    "TBD",
        },
        "Occupation & Industry": {
            "Job Title":     "TBD",
            "Industry Type": "TBD",
        },
        "Interests": {
            "Hobbies":                      [],
            "Lifestyle":                    [],
            "Travel Style":                 [],
            "Preferred Destinations":       [],
            "Food and Dining Preferences":  [],
            "Dislikes":                     [],
        },
    }
    pets_line = _pets_from_sources(sources)
    if pets_line is not None:
        profile["Pets"] = pets_line
    for s in sources:
        if s.drift_action == "drop" or not s.variant_text:
            continue
        field = _effective_field(s)
        if field in profile["Interests"]:
            profile["Interests"][field].append(s.variant_text)
    # De-duplicate each field, preserving first-seen order
    for f in profile["Interests"]:
        seen = set()
        uniq = []
        for t in profile["Interests"][f]:
            if t not in seen:
                seen.add(t)
                uniq.append(t)
        profile["Interests"][f] = uniq
    return profile


def _source_to_trace_dict(s: Source) -> dict:
    return {
        "sid":          s.sid,
        "kind":         s.kind,
        "paradigm":     s.paradigm,
        "bank_id":      s.bank_id,
        "op_name":      s.op_name,
        "path":         list(s.path),
        "raw_value":    s.raw_value,
        "field":        s.field,
        "trait_table":  s.trait_table,
        "trait_key":    list(s.trait_key) if isinstance(s.trait_key, tuple) else s.trait_key,
        "variant_idx":  s.variant_idx,
        "variant_text": s.variant_text,
        "drifted":      s.drifted,
        "drift_action": s.drift_action,
    }


# --------------------------------------------------------------------------- #
# Driver
# --------------------------------------------------------------------------- #

def main():
    bank = _load_bank()
    bank_idx = _build_bank_index(bank)

    # Load all augmented records
    records: list[dict] = []
    with IN_PATH.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    print(f"Loaded {len(records)} augmented records.")

    # Assign drift modes (stratified per level)
    drift_modes = assign_drift_modes(records)

    # Build per-query profiles
    out_records: list[dict] = []
    src_counter   = Counter()       # diagnostic: sources per source-kind
    field_counter = Counter()       # diagnostic: signals per field
    drift_dist    = defaultdict(Counter)
    empty_profile_qids: list[int] = []

    for rec in records:
        qid = rec["query_id"]
        level = rec.get("level", "easy")
        sources = disintegrate(rec, bank_idx)

        # Pick aligned variants for all sources first
        for s in sources:
            assign_variant(s, qid, use_inversion=False)

        # Apply drift
        mode = drift_modes[qid]
        drift_meta = apply_drift(sources, qid, mode)

        # Assemble profile
        profile = build_profile(rec, sources, mode)
        if all(len(v) == 0 for k, v in profile["Interests"].items()):
            empty_profile_qids.append(qid)

        # Telemetry
        drift_dist[level][mode] += 1
        for s in sources:
            src_counter[s.kind] += 1
            field_counter[s.field] += 1

        rec_out = dict(rec)
        rec_out["profile"] = profile
        rec_out["profile_drift_mode"] = mode
        rec_out["profile_trace"] = {
            "drift_meta": drift_meta,
            "sources":    [_source_to_trace_dict(s) for s in sources],
        }

        # ----- NL generation ---------------------------------------------
        trip_context = resolve_trip_context(rec_out)
        templated_profile  = render_nl_profile(rec_out)
        templated_query, nl_render_log = render_nl_query(rec_out, trip_context)
        rec_out["trip_context"]          = trip_context
        rec_out["templated_nl_profile"]  = templated_profile
        rec_out["templated_nl_query"]    = templated_query
        rec_out["nl_render_log"]         = nl_render_log

        # Drop legacy field names carried over from prior runs of this script
        for legacy in ("nl_profile", "nl_query"):
            rec_out.pop(legacy, None)

        out_records.append(rec_out)

    # Diagnostics
    print("\n=== Source kinds ===")
    for k, n in src_counter.most_common():
        print(f"  {k:20s} {n}")
    print("\n=== Profile field signal counts ===")
    for f, n in field_counter.most_common():
        print(f"  {f:30s} {n}")
    print("\n=== Drift mode distribution by level ===")
    for level in sorted(drift_dist):
        row = drift_dist[level]
        tot = sum(row.values())
        print(f"  {level:6s} (n={tot:4d})  "
              + "  ".join(f"{m}={row[m]} ({row[m]/tot:5.1%})"
                          for m in ("aligned", "omission", "inversion")))
    print(f"\nQueries with empty profile (no Interests signals): {len(empty_profile_qids)}")
    if empty_profile_qids[:5]:
        print(f"  example qids: {empty_profile_qids[:10]}")

    # Backup + write
    if not BACKUP.exists():
        shutil.copy2(IN_PATH, BACKUP)
        print(f"\nBackup written: {BACKUP.name}")
    with OUT_PATH.open("w") as f:
        for rec in out_records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    print(f"Wrote {len(out_records)} records with profile to {OUT_PATH.name}")


if __name__ == "__main__":
    main()
