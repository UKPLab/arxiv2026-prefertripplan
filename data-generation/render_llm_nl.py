"""LLM-based natural-language renderers for profile + query.

Adds two new fields per augmented record:
    llm_nl_profile  — fluent first-person profile introduction
    llm_nl_query    — fluent trip-request message, drift-aware framing

Supports three backends (pick via ``--backend`` or env ``LLM_NL_BACKEND``):
    anthropic   — Anthropic API (default).  Needs ANTHROPIC_API_KEY.
    openai      — OpenAI API.  Needs OPENAI_API_KEY (and optionally
                  OPENAI_BASE_URL to point at an OpenAI-compatible
                  endpoint).
    vllm-offline — In-process vLLM ``LLM`` class.  Loads weights and
                  batches all prompts in a single ``llm.chat()`` call.

A sidecar cache file (``llm_nl_cache.jsonl``) holds results keyed by
``(qid, kind)``.  Runs are resumable — restart picks up where it left
off.

Usage:
  python3 render_llm_nl.py                                    # anthropic default
  python3 render_llm_nl.py --backend openai --model gpt-4o-mini
  python3 render_llm_nl.py --backend vllm-offline --model meta-llama/Llama-3.1-8B-Instruct
  python3 render_llm_nl.py --sample 5                         # smoke test
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).resolve().parent

# Input is the augmenter + generate_profiles.py output.  Output and cache
# default to MODEL-SCOPED filenames (see ``_model_output_paths`` below) so
# two concurrent runs with different backends never clobber each other's
# caches or output files.  Override explicitly with ``--out`` / ``--cache``
# to write to a shared file if you know what you're doing.
IN_PATH             = ROOT / "prefertripplan.jsonl"
OUT_PATH_TEMPLATE   = "prefertripplan.{slug}.jsonl"
CACHE_PATH_TEMPLATE = "llm_nl_cache.{slug}.jsonl"

# Legacy (pre-model-scoping) filenames.  Kept only so a caller who passes
# a literal --out / --cache pointing at these files can still write there;
# not used as defaults anywhere.
LEGACY_OUT_PATH   = ROOT / "prefertripplan.jsonl"
LEGACY_CACHE_PATH = ROOT / "llm_nl_cache.jsonl"


def _model_slug(name: str) -> str:
    """Turn a model id into a filename-safe slug.

    Examples:
      claude-haiku-4-5-20251001         -> claude-haiku-4-5-20251001
      meta-llama/Llama-3.1-8B-Instruct  -> meta-llama_Llama-3.1-8B-Instruct
      gpt-5.4-mini                      -> gpt-5.4-mini
    Every character not in ``[A-Za-z0-9._-]`` becomes ``_``.
    """
    import re as _re
    return _re.sub(r"[^A-Za-z0-9._-]+", "_", name).strip("_")


def _model_output_paths(model: str) -> tuple[Path, Path]:
    """Return ``(out_path, cache_path)`` for a model, both in ROOT and
    both suffixed with the model slug."""
    slug = _model_slug(model)
    return (ROOT / OUT_PATH_TEMPLATE.format(slug=slug),
            ROOT / CACHE_PATH_TEMPLATE.format(slug=slug))


# --------------------------------------------------------------------------- #
# Progress reporter                                                           #
# --------------------------------------------------------------------------- #
# Small, dependency-optional per-completion progress + ETA line, used by the
# parallel backends (Anthropic, OpenAI). Prefers `tqdm` when available; falls
# back to a self-contained \r-updated text line so runs stay legible in
# terminals without tqdm installed. vLLM's own tqdm bar is left untouched.
class _ProgressReporter:
    def __init__(self, total: int, label: str = "generating"):
        self.total  = total
        self.done   = 0
        self.label  = label
        self.start  = time.time()
        self._last_line_len = 0
        self._tqdm  = None
        try:
            from tqdm import tqdm as _tqdm  # optional dep
            self._tqdm = _tqdm(total=total, desc=label, unit="req",
                               dynamic_ncols=True, mininterval=0.3,
                               leave=True)
        except Exception:
            # No tqdm -> print an initial banner so the user sees a signal
            # right away, then rely on step() for live updates.
            self._fallback_write(
                f"[progress] {label}: 0/{total} (starting...)"
            )

    def _fmt_dt(self, s: float) -> str:
        s = int(max(s, 0))
        h, rem = divmod(s, 3600)
        m, sec = divmod(rem, 60)
        if h: return f"{h}h{m:02d}m{sec:02d}s"
        if m: return f"{m}m{sec:02d}s"
        return f"{sec}s"

    def _fallback_write(self, s: str) -> None:
        # In-place update via carriage return; pad to erase any earlier
        # longer line.
        pad = " " * max(0, self._last_line_len - len(s))
        sys.stderr.write("\r" + s + pad)
        sys.stderr.flush()
        self._last_line_len = len(s)

    def step(self, n: int = 1) -> None:
        self.done += n
        if self._tqdm is not None:
            self._tqdm.update(n)
            return
        elapsed = time.time() - self.start
        rate    = self.done / elapsed if elapsed > 0 else 0.0
        remain  = (self.total - self.done) / rate if rate > 0 else 0.0
        pct     = 100.0 * self.done / self.total if self.total else 100.0
        self._fallback_write(
            f"[progress] {self.label}: {self.done}/{self.total} "
            f"({pct:5.1f}%)  elapsed={self._fmt_dt(elapsed)}  "
            f"eta={self._fmt_dt(remain)}  rate={rate:.2f} req/s"
        )

    def close(self) -> None:
        if self._tqdm is not None:
            self._tqdm.close()
            self._tqdm = None
        else:
            # End the \r line with a newline so subsequent prints go on
            # their own row.
            if self._last_line_len:
                sys.stderr.write("\n")
                sys.stderr.flush()
                self._last_line_len = 0

DEFAULT_MODELS = {
    "anthropic":     "claude-haiku-4-5-20251001",
    "openai":        "gpt-5.4-mini",
    "vllm-offline":  "meta-llama/Llama-3.1-8B-Instruct",
}
TEMPERATURE = 0.7


# --------------------------------------------------------------------------- #
# Prompts (shared across backends)
# --------------------------------------------------------------------------- #

PROFILE_PROMPT = """You are rewriting a structured profile into a fluent, natural, first-person introduction. The profile describes a SINGLE traveler.

FAITHFULNESS:
- Use ONLY the content listed below. Do NOT invent demographics, occupations, family structure, or any preference not listed.
- Do not add hobbies, foods, opinions, or dislikes that aren't in the input.
- Do not name companions.

STYLE:
- One flowing paragraph (roughly 4–8 sentences). No bullet lists, no headers, no field labels.
- The opening must INTEGRATE the location into a full, natural-sounding sentence that introduces the person. The location should be part of a real clause, not a period-terminated fragment standing alone.

  Rule for opener content: the location is the ONLY new information the opener may introduce that isn't in the Interests fields. Anything ELSE the opener says about the person — hobbies, lifestyle stance, travel style, food, dislikes, occupational identity, duration of residence, effect of home city on their travel style — MUST come from a real item in the Interests input. Do NOT invent filler traits to pad the opener into a "richer-sounding" first sentence. The reader is introduced to the person by the input, not by the model's guesses.

    GOOD openers (each uses ONLY the location + an actual Interests item, or the location alone as an anchor):
      - "I'm based in <Location>, and <a Hobbies / Lifestyle / Travel-Style item from the input, rephrased naturally>..."
      - "I'm based in <Location>. <continues with an actual Interests item>..."
      - "Home for me is <Location>, and <actual Interests item>..."
      - "Coming from <Location>, I'm the kind of traveler who <actual Interests item>..."
      - "<Location>-based, and <actual Interests item>..."

    BAD openers (telegraphic labels — read as a header, not a natural sentence):
      - "Home's <Location>." / "<Location>'s home."   ← truncated, reads as a stamp
      - "<Location>. I'm ..."
      - "Home base is <Location>. I love ..."
      - "Location: <Location>..."

    BAD openers (fabricated filler — the opener invents traits that aren't in the input):
      - "I'm based in Boston, and I've never been able to resist a decent bookshop..."   ← if the input says nothing about bookshops
      - "Living in Sacramento means I spend most weekends chasing farmers' markets..."   ← same
      - "Coming from Cleveland, I'm the kind of traveler who books early and researches obsessively..."   ← same
      - "I've lived in Baltimore for years, and outdoor recreation is core to how I travel..."   ← "for years" is a fabricated duration; the input never states residence length
      - "Home for me is Boston — an amateur historian who never gets tired of a well-curated museum..."   ← "amateur historian" is a fabricated occupational/hobbyist identity; the input just says they enjoy contemplating artifacts, not that they identify as any kind of historian
      - "Living in <Location> has shaped how I travel..."   ← implies the location caused their travel style; the input never states a causal link
      - Any adjective, hobby, occupation, duration, or personality trait in the opener that isn't traceable to a specific line of the Interests input.

  The location is the ANCHOR at the start, not the entire first sentence. But everything ELSE in the first sentence must come from the actual input.

- Weave every Interests item that has content into the paragraph. Skip fields marked "(none specified)" silently — do not mention their absence.
- Vary sentence rhythm; do not open every clause with "I".
- The variant sentences in the input are already complete claims; combine them into fluent prose without inventing new facts around them (this applies to the WHOLE paragraph, not just the opener).

PETS RULE:
- The pets_line below is the ONLY signal about pets. If pets_line is exactly "(no signal)" — omit any mention of pets from the paragraph entirely. Do NOT write "no pets", "I don't have pets", "no animals", or any similar disclaimer.
- If pets_line contains any other text (e.g. describing a pet-owner traveler or a pet-averse traveler), weave that stance into the paragraph naturally. If the pets_line implies pet ownership, invent a plausible pet name+type (e.g. "my beagle Riley", "my cat Luna") — but do this ONLY when the pets_line is present.

PRICE BASIS (accommodation / restaurant traits):
- Any Interests item that mentions accommodation cost tiers ("per-person nightly rate", "budget-tier per-person nightly", "premium per-person nightly", "median per-person nightly", etc.) is on a PER-PERSON PER-NIGHT basis. Preserve any explicit "per person per night" / "per-person nightly" wording that already appears in the trait — do NOT paraphrase it away, and do NOT rewrite it as generic "cost" without the basis.
- Any Interests item that mentions restaurant cost / meal spend is on a PER-PERSON PER-MEAL basis (TravelPlanner Average-Cost convention).
- Do NOT introduce dollar figures the input doesn't already contain — but when a trait already carries a per-person basis in words, that basis must survive into the fluent paragraph.

INPUT:
Location: {location}
Hobbies and interests: {hobbies}
Lifestyle: {lifestyle}
Travel style: {travel_style}
Preferred destinations: {pref_dests}
Food and dining preferences: {food}
Things I actively avoid: {dislikes}
Pets line: {pets}

Write the profile paragraph now. Output only the prose — no preamble, no closing."""


QUERY_PROMPT = """You are rewriting a structured trip request into a fluent, natural traveler's message. The traveler is one person describing the trip; the trip_context phrase tells you how that person frames the trip (solo / partner / family / friend group / etc.).

FAITHFULNESS:
- Use ONLY the content listed below. Do NOT invent companion biographies, occupations, ages, or preferences not listed.
- You MAY refer to companions implied by the trip-context phrase ("my partner", "the family", "our group") but do not give them specific traits beyond what the structured input says.

COST BASIS (accommodation / restaurant thresholds — READ BEFORE VERBALIZING):
- `Accommodation.cost` thresholds ($150, ≥ 200, ≤ 100, etc.) are on a PER-PERSON PER-NIGHT basis. Verbalize them that way — e.g. `Accommodation.cost ≤ 100 [all]` → "every stay comes in at $100 per person per night or less" (not "$100 a night" for the room, and not "$100 per person" without the per-night qualifier).
- `Restaurant.cost` thresholds are on a PER-PERSON PER-MEAL basis (TravelPlanner Average-Cost convention). Verbalize accordingly — e.g. `Restaurant.cost ≥ 45` → "restaurants running at $45 per person per meal or higher" or "restaurants averaging $45 or more per person per meal".
- KEEP THE BASIS ATTACHED TO THE NUMBER — DO NOT ADD A SEPARATE CLARIFIER SENTENCE. The "per person per night" / "per person per meal" qualifier belongs immediately next to the dollar figure ("$220 per person per night", "$45 per person per meal"). NEVER add a standalone housekeeping sentence like "just to be clear, the accommodation cost is per person per night", "note that the restaurant figure is per person per meal", "the prices I quote are on a per-traveler basis", "for the stay, the cost I mention is per person per night", or any other meta-remark that spells out the basis separately from the number itself. If every cost mention already carries the qualifier inline, the reader knows the basis — a separate clarifier is redundant and unnatural.
- The band inside `min(Accommodation.cost) via <agg> [lo, hi]` / `max(Restaurant.cost) via <agg> [lo, hi]` is still bookkeeping and must NOT be surfaced — the per-person basis rule applies only to concrete threshold predicates.
- TRIP BUDGET IS NOT PER-PERSON: the `Budget: ~$<N>` figure in the trip facts is the TOTAL budget for the WHOLE trip — spanning every traveler, every day, every category (accommodation, dining, transportation, attractions). NEVER attach "per person", "per night", "per meal", "per day", or any similar qualifier to this figure. Verbalize it plainly: "on a budget of about $<N>", "the trip budget is around $<N>", "we're aiming to stay under $<N> in total". Do NOT split it across travelers ("$<N>/2 = ..."), across nights, or across cost categories. The per-person cost basis rule above applies ONLY to `Accommodation.cost` and `Restaurant.cost` preference thresholds — it does NOT apply to the overall trip budget.

CONCRETE VALUES MUST BE PRESERVED (WITH ONE EXCEPTION):
- Each preference comes with a structured predicate. Threshold-style numeric values ($150, ≥ 4.5, 3.0), value sets ({Nightlife, Museums}), operator directions (≥, ≤, ∈, ∉), scopes ([all] every item vs [any] at least one item), day indices, cuisines, house-rule flags — all MUST appear in your prose.
- What you rewrite is the SURFACE NOTATION, not the values. Values survive; only notation naturalizes.
- FAITHFULNESS — DO NOT INVENT QUANTIFIERS: Never add "at least one X and one Y" style quantifiers on a set unless the predicate explicitly requires them (i.e. an explicit temporal operator or an explicit count constraint). The set-membership operator `∈` alone is NOT a count constraint.
- The reader must be able to recover threshold predicates from your prose. "≥ 3.0" is "at least 3.0" or "3.0 or higher", not "decent".
- FAITHFUL COVERAGE OF COMPOUND PREDICATES: When a predicate has multiple named parts, EVERY part must appear in the verbalization. It is a common failure mode to drop the middle piece — do not do this.
    - Compensatory `PRIMARY: A | MARGIN: B | SECONDARY: C` — all THREE parts (primary threshold, margin threshold, secondary compensating attribute) must appear. Primary and Margin are trip-wide universal on the primary entity; every primary-entity item lands either at Primary (ideal) or in the Margin band (acceptable ONLY when compensated by Secondary same-day). Secondary is same-day existential — one qualifying secondary-entity item per day suffices to compensate. Do NOT drop the MARGIN, and do NOT universalize the Secondary into "every Y" — Secondary is same-day any-scope, not trip-wide.
    - Conditional `WHEN cond: pref` (or `IF cond THEN pref`) — both the condition and the consequent must appear.
    - Lexicographic `A ≻ B` — both the first-ranked A and the secondary B must appear, and their strict-priority ranking (A ranked above B) must be clear. B is NOT a fallback; see the Lexicographic SEMANTIC-SCOPING rule for phrasing.
    - Composite `A AND B` / `A OR B` — both operands and the connective must appear.
    - Temporal `op(A, ...)` or `op(A, B, ...)` — the operator's semantics AND every argument (the target predicate, any time arguments, any secondary predicate) must appear.
    - Scoped predicates — the scope (per_day / per_city / global / all / any / week_group) must be reflected in the verbalization (universal vs. existential; per-day vs. across-trip).
- EXCEPTION — OPTIMIZATION AGGREGATIONS: predicates of the form "min(<attr>) via <agg> [lo, hi]" or "max(<attr>) via <agg> [lo, hi]" (e.g. `min(Accommodation.cost) via avg [50, 310]`, `max(Restaurant.aggregate_rating) via avg [3.0, 5.0]`) express PURE OPTIMIZATION — the goal is to minimize / maximize the aggregate. The [lo, hi] band is bookkeeping metadata (search envelope), NOT a preference threshold. OMIT the band values entirely; the optimization action itself IS the preference.
  - `min(Accommodation.cost) via avg [50, 310]` → "ideally the average per-person nightly accommodation cost across the trip is kept as low as possible"
  - `max(Restaurant.aggregate_rating) via avg [3.0, 5.0]` → "ideally the average restaurant rating across the trip trends as high as possible"
  - Do NOT write "the $50 to $310 band", "trend toward the low end of the [50, 310] range", or any surface mention of the bookkeeping band.
  - This exception applies ONLY to `min(...)` / `max(...)` optimization predicates. Threshold predicates like `Accommodation.cost ≤ 100` still keep their number (verbalized as "$100 per person per night or less").

SET-MEMBERSHIP SEMANTICS — OR-KIND vs AND-KIND:
Set-based predicates like `Entity.attr ∈ {V1, V2, V3}` do NOT all mean the same thing. Their reading depends on which attribute they apply to:

(a) OR-KIND sets — the default. Attributes: cuisine, attraction category, room type (and any other categorical attribute unless otherwise stated). The set enumerates ALLOWED OPTIONS; a valid entity value need only be ONE of them (any single member satisfies membership). The connective is a LOOSE "or" / "like" enumeration, NOT a count constraint.
    - `Attraction.category ∈ {Nightlife, Museums}` → "interested in attractions like nightlife spots and museums" (or "attractions such as nightlife spots or museums", "somewhere along the lines of nightlife or museum spots"). The "and"/"or" is loose enumeration; do NOT read this as "one nightlife AND one museum both required".
    - `Restaurant.cuisine ∈ {Chinese, Mexican} [scope: all]` → "every meal is Chinese or Mexican" (each meal is one of the two — not both). Scope [all] applies universally to each meal; the set inside is still OR-kind.
    - `Restaurant.cuisine ∈ {Chinese, Mexican} [scope: any]` → "somewhere on the trip we'd like a Chinese or Mexican meal".
    - NEVER write "at least one nightlife AND at least one museum" for `∈ {Nightlife, Museums}` — that adds a count/enumeration quantifier not in the predicate.
    - The scope tag ([all] / [any]) is the ONLY source of quantification; the set operator ∈ alone does not quantify.

(b) AND-KIND sets — the exception. Attribute: house_rules. Elements of the set are treated as conjunctive: all listed rules apply together.
    - `Accommodation.house_rules ∈ {smoking, parties}` → "the accommodation allows smoking and allows parties" (both, together).
    - `Accommodation.house_rules ∉ {No smoking, No parties, No visitors}` → "the accommodation isn't restricted by no-smoking, no-parties, or no-visitors rules" (none of the three restrictions apply). This is the AND-kind reading of ∉: excludes ALL listed rules.
- SCOPE and set semantics COMPOSE:
    - `∈ {V1, V2} [scope: all]` — each item is one of {V1, V2} (OR-within-set, universal across items)
    - `∈ {V1, V2} [scope: any]` — at least one item is one of {V1, V2} (OR-within-set, existential)
    - `[scope: any]` is the ONLY case where "at least one" language is appropriate. Never introduce "at least one" for [all]-scope or unscoped predicates.

NATURAL-LANGUAGE RENDERING OF PREDICATE NOTATION (cheatsheet — never leave structural notation like `Entity.attr ∈ {X}` or `hold_after(...)` or `min(...) via ...` in the prose; translate every construct):
- `Entity.attr ∈ {V1, V2}` (OR-KIND) → *"<attr> like <V1> and <V2>"* / *"somewhere along the lines of <V1> or <V2>"*. NO invented count ("at least one X and one Y"); the scope tag is the only quantifier.
- `Entity.attr ∈ {V1, V2}` (AND-KIND, house_rules only) → *"the accommodation allows <V1> and <V2>"* (both listed rules together).
- `Entity.attr ∉ {V1, V2}` (OR-KIND) → *"not looking at places that are <V1> or <V2>"*.
- `Entity.attr ∉ {V1, V2, V3}` (AND-KIND, house_rules only) → *"the accommodation isn't restricted by <V1>, <V2>, or <V3>"* (excludes all three).
- `Entity.attr ≥ N` / `≤ N` → *"rated N or above"*, *"at least $N per person per night"* (accommodation cost), *"under $N per person per meal"* (restaurant cost), *"no more than N"*.
- Scope `[all]` → *"every …"*, *"each …"*, *"throughout the trip"* (universal — never insert "at least one").
- Scope `[any]` → *"at least one …"*, *"somewhere on the trip"* (existential — this is the ONLY case where "at least one" is licensed).
- Unscoped predicate → treat as universal by default; do NOT introduce "at least one".
- `min(<attr>) via <agg> [lo, hi]` → *"ideally the <agg> <attr> across the trip is kept as low as possible"* — OMIT the [lo, hi] band.
- `max(<attr>) via <agg> [lo, hi]` → *"ideally the <agg> <attr> across the trip trends as high as possible"* — OMIT the [lo, hi] band.

SEMANTIC SCOPING — WHERE EACH PREFERENCE KIND APPLIES (weave into the prose subtly, never as a rule listing; a reader should tell from HOW the preference locates itself in the trip, not by category labels):

Atomic — one condition on one entity type across the whole trip.
- `[any]`: at least one such entity satisfies. *"somewhere on the trip"*, *"at least one"*, *"one or more"*.
- `[all]`: every such entity satisfies. *"every X"*, *"each X"*, *"throughout the trip"*.

Composite (AND / OR) — two independent conditions across the whole trip; NO same-day or same-slot coupling.
- AND-composite → both operands should hold per their own scopes; OR-composite → at least one.
- Verbalize the connective with LOWERCASE natural English: *"both … and …"* / *"either … or …"*. Never carry the uppercase AND/OR from the predicate notation into the output — that's structural notation, not natural language.
- Do not imply co-occurrence.

Conditional (IF cond THEN pref) — the consequent kicks in when the condition happens SOMEWHERE in the trip, and the consequent itself can then happen ANYWHERE in the trip — NOT tied to the same day as the condition.
- *"if the trip involves X, then somewhere on the trip we'd also like Y"*, *"assuming X, we'd prefer Y"*.
- Do NOT phrase as *"on days when X, then Y that day"* — that reads as Scoped, not Conditional.

Lexicographic (A ≻ B) — STRICT PRIORITY ORDER. A is foremost; B is secondary. NOT a fallback.
- Satisfying A but not B is strictly preferred over satisfying B but not A. The best case is both A and B satisfied together when that's structurally possible; B is never a substitute for A.
- Convey the strict ordering with RANKING language, not just "priority". Signals that carry strict-lex intent: *"first-ranked"*, *"our first-order priority"*, *"decisively ahead of"*, *"comfortably above"*, *"outranks"*, *"at the top of our list, and clearly above the second"*, *"the secondary is next in the ranking but distinctly behind"*. Bare *"top priority is A, and B is nice"* reads as a soft preference over two items — strengthen it with ranking words.
- Phrase as a straight ranking: *"our first-order priority is <A>; ranked below that, <B>"*, *"<A> comes first, comfortably above <B>"*, *"at the top of our list is <A>; <B> ranks below"*, *"first-ranked <A>; <B> is next but distinctly behind"*. Substitute the actual concrete predicates for <A> and <B> in your output.
- DO NOT introduce mixed-satisfaction comparisons that interpolate A and B (e.g. *"even A on some items over B on all items"*, *"A on this side plus B on that side"*, *"A partially plus B fully"*). Convey the ordering with the ranking alone — never with a mix.
- DO NOT use fallback language (*"B as a backup"*, *"B if A doesn't work"*, *"fall back to B"*).
- DO NOT use additive language when A and B share the same entity+attribute with different values (*"on top of that B"* reads wrong when only one of A, B can hold on a given item — use pure ranking there instead).
- DO NOT write the literal placeholder tokens "A", "B", "primary", "secondary", "first tier", "second tier", or any structural label in the output. Every reference to the ranked items must name the actual predicate values in natural English (thresholds, categories, cuisines, modes, etc.).
- LEXICOGRAPHIC `[any]`-scope OVERRIDE — DO NOT use *"at least one"* for `[any]`-scope sub-preferences inside a lexicographic ranking. Strict-priority ordering makes the higher-ranked slot the STRONGEST cause for satisfaction, and its intent is MAXIMIZING (more is better) — *"at least one"* smuggles in a minimizing lower-bound reading (*"one is enough, no more needed"*) that clashes with that intent. Render `[any]`-scope Lex slots with **maximizing-neutral open-ended** wording instead: *"one or more …"*, *"some …"*, *"a … [singular indefinite]"*. This override applies **only inside Lexicographic**; all other paradigms can continue to use *"at least one"* for `[any]` per the cheatsheet.
- Example: `Accommodation.review_rate ≥ 4 [all] ≻ Restaurant.aggregate_rating ≥ 3.5 [any]` → *"Consistently comfortable accommodations matter more to us than an occasional highly-rated meal — our first-order priority is every stay rated 4 or higher. Ranked below, we'd also like one or more restaurants rated 3.5 or higher."*
- Example: `Transportation.mode = Flight [all] ≻ Transportation.mode ∈ {taxi} [all]` → *"On transportation, air travel cuts transit time and fatigue — our decisive first choice is every inter-city leg as a flight. Taxi is next in the ranking but distinctly behind."*

Numeric optimization (`min` / `max` aggregate) — trip's aggregate trends one direction. OMIT the [lo, hi] band.
- *"ideally the <agg> <attr> across the trip is kept as low as possible / as high as possible"*. No numbers from the band.

Scoped — inner preference applies only within a filter-carved sub-scope (a set of days, a city, or a filtered slice of entities).
- Use the filter to name the sub-scope naturally, then place the inner preference INSIDE it.
- *"on stay days, we'd like at least one museum"*, *"in every city we visit, we'd like at least one nature spot"*, *"when we're eating dinner, we'd lean toward Italian"*.
- Distinguish from Conditional: Scoped confines the inner preference's evaluation to the sub-scope; Conditional lets the consequent happen anywhere in the plan.

Compensatory (PRIMARY | MARGIN | SECONDARY) — three parts; ALL three must survive in the prose.
- Primary: the trip-wide ideal target on the primary entity — every primary-entity item across the trip should hit Primary.
- Margin: the tolerance band on the SAME primary entity (the attribute may match primary's or may be different — but the margin still applies to the same entity, never a different one). An item that hits Margin but not Primary is acceptable only when it is COMPENSATED by Secondary.
- Secondary: the compensator that redeems a drop into the Margin band — evaluated as an "at least one" existence on the SAME DAY as the Margin-only item (never trip-wide "somewhere", never a different day).
- Primary and Margin are UNIVERSAL over primary-entity items — every primary-entity item across the trip lands either at Primary or in the Margin band (with Secondary compensating the drop). A single Margin-uncompensated or below-Margin item breaks the preference.
- SAME entity type as PRIMARY → same-entity compensation. *"we'd like every X to hit the primary ideally; a drop into the margin band is acceptable only when THAT same X also has the secondary property"*.
- DIFFERENT entity type from PRIMARY → same-DAY compensation. *"we'd like every X to hit the primary ideally; a drop into the margin band is acceptable only when the day ALSO has a Y satisfying the secondary"*. The compensator lives on the same day — never a different day, never a trip-wide "somewhere".
- Language cues for Primary/Margin: *"every X"*, *"each X"*, *"throughout the trip"*, *"across the trip"* — all appropriate (they're trip-wide universal on primary entity). For Secondary: *"same-day"*, *"the day already has"*, *"when the day includes"* — existential and same-day only.
- Choose generic wording that fits the primary entity naturally — a Restaurant.primary reads as "every meal" or "every restaurant", an Attraction.primary reads as "every attraction stop", an Accommodation.primary reads as "every stay" (accommodation is one per city, so this reduces gracefully). Never universalize the SECONDARY: keep it same-day existential.

Temporal — an operator applied over a scope group. TWO orthogonal scopes must both be visible in the prose without conflating them:
  (a) Operator scope: `global` / `per_day` / `per_city` / `week_group` / `travel_phase` — governs how the trip is partitioned before the operator applies. If unstated, treat as `global`. `travel_phase` values: *"stay days"* (settled-in) or *"travelling days"* (arrival / departure).
  (b) Inner-predicate `[any]` / `[all]` scope — governs how many entities satisfy the subject / reference INSIDE the temporal. Both survive independently in the prose.

Temporal step — day-level by default (positions and ordering compare day indices; prose says *"days"*). When `day_as_step=False` is explicitly on (also implicit for `per_day` operator scope), positions compare within-day slot indices — prose then drops the *"day"* framing and uses slot-named framing TIED TO THE ENTITY TYPE (*"the next dining slot"*, *"an earlier attraction stop"*, *"within the next 2 restaurant slots"*, *"the very next meal"*) — never a bare *"slot"* divorced from the entity. `always` and `sometime` are inherently day-based and have no slot-step variant. Never mix day and slot framing within one predicate.

Operator phrasings (day-step by default; slot-step only when day_as_step is explicitly off):
- `always(P)`: every day in the scope has at least one entity satisfying P. Day-based only.
  *"ideally every day, at least one X …"* / *"on each stay day, at least one X …"*.
- `sometime(P)`: at least one occurrence somewhere in the scope. Day-based only.
  *"somewhere on the trip, at least one X …"* / *"on some day in each city, at least one X …"*.
- `atmost_once(P)`: at most one occurrence across the scope.
  day-step: *"across the trip, we'd rather have X on at most one day"* / *"at most one X per city"*.
  slot-step: *"one X-slot is plenty across the trip — we'd rather not repeat it"* (name the entity: *"one museum visit is plenty"*, *"one $50+ per-person meal is plenty"*).
- `within(P, k)`: P should appear within the first k positions of the scope.
  day-step: *"within the first k days of the trip, ideally at least one X …"* / *"within each city's first k days …"*.
  slot-step: *"within the first k <entity>-slots …"* — e.g. *"within the first 2 dining slots"*, *"within the first 3 attraction stops"*.
- `sometime_before(A, B)`: A precedes B in the scope.
  day-step: *"we'd like A to happen on some day before B in the trip"* / *"in each city, A on an earlier day than B"*.
  slot-step: *"we'd like A at an earlier <entity>-slot than B in the trip"* — e.g. *"museums earlier on the attraction schedule than shopping"*.
- `sometime_after(A, B)`: A follows B in the scope.
  day-step: *"we'd like A to come on some day after B in the trip"* / *"in each city, A on a later day than B"*.
  slot-step: *"we'd like A at a later <entity>-slot than B in the trip"* — e.g. *"the fancier dinner later on the dining schedule than the casual lunch"*.
- `always_within(A, B, k)`: whenever A appears, B follows within the next k positions in the same scope group.
  day-step: *"whenever A lands on the schedule, we'd like B to follow within the next k days"*.
  slot-step: *"whenever A lands on the schedule, we'd like B to follow within the next k <entity>-slots"* — e.g. *"the very next meal"* (k=1) or *"within the next 2 attraction stops"* — always naming the entity type.
- `hold_during(P, d_start, d_end)`: P holds on every day in [d_start, d_end]. Day-indexed by construction.
  *"during days d_start through d_end, ideally every X …"*.
- `hold_after(P, d_start)`: P holds on every day from d_start onward. Day-indexed by construction.
  *"from day d_start onward, ideally every X …"*.

Scope wording cues (match to operator scope):
- `global` (or unstated) → *"across the trip"*, *"anywhere on the trip"*, *"on any day of the trip"* (day-step) / *"at any point on the schedule"* (slot-step, where applicable).
- `per_day` → *"on each day"*, *"within each day"* (inherently slot-step for ordered operators).
- `per_city` → *"in each city we visit"*, *"per city"*.
- `week_group` → *"on weekends"*, *"on weekdays"*.
- `travel_phase` → *"on stay days"* (settled-in) or *"on travelling days"* (arrival / departure).

WORKED VERBALIZATIONS — ONE EXAMPLE PER PARADIGM (study the pattern: every predicate value survives; rationales are short, generic, free of clashing numbers; every named part of a compound predicate appears in the prose; scope and step semantics come through in the phrasing, not by label):

(1) ATOMIC — `Accommodation.review_rate ≥ 4 [scope: all]`
    rationale: "Milestone trip; sworn off substandard accommodations after one bad experience."
    verbalization: "After a bad past stay we've sworn off substandard accommodations, so we'd ideally like every stay on this trip to be rated 4 or higher."

(2) COMPOSITE — AND — `Restaurant.cuisine ∈ {Street Food, Mexican, Chinese} [all] AND Restaurant.aggregate_rating ≥ 2.5 [all]`
    rationale: "Wants real ordinary food but won't risk a genuinely bad meal."
    verbalization: "For real, ordinary food that still doesn't risk a genuinely bad meal, we'd ideally like every restaurant to serve Street Food, Mexican, or Chinese and be rated 2.5 or higher."
    (Note: use lowercase "and" / "or" as natural connectives in the verbalization; the uppercase AND / OR in the input predicate is structural notation and MUST NOT be carried into the output.)

(3) COMPOSITE — OR — `Accommodation.cost ≤ 100 [all] OR Accommodation.review_rate ≥ 4 [all]`
    rationale: "The mid-range is worst value — either cheap or excellent."
    verbalization: "Given the mid-range is the worst value for us, we'd ideally like every stay to satisfy one of two things: either it comes in at $100 per person per night or less, or it's rated 4 or higher."

(4) NUMERIC — OPTIMIZATION — `max(Accommodation.review_rate) via min [1, 5]`
    rationale: "Prefer places with the least downside."
    verbalization: "I'm after places with the least downside, so I'd ideally like the trip's minimum accommodation review score to be as high as possible."
    (The [1, 5] band is bookkeeping — never surfaced.)

(5) CONDITIONAL — `IF Restaurant.cost ≥ 50 [any] THEN Restaurant.cost ≤ 15 [any]`
    rationale: "Balance a premium dining experience with a cheap one."
    verbalization: "If the trip includes any restaurant that runs at $50 or more per person per meal, then somewhere on the trip we'd also like at least one restaurant that comes in at $15 or less per person per meal — balancing a premium meal with a modest one so the food budget stays sane."
    (The two sides are independent existence checks across the whole plan — the "cheap" side does NOT have to fall on the same day as the "premium" side.)

(6) LEXICOGRAPHIC — `Accommodation.review_rate ≥ 4 [all] ≻ Restaurant.aggregate_rating ≥ 3.5 [any]`
    rationale: "Consistently comfortable accommodations matter more than an occasional highly-rated meal."
    verbalization: "Consistently comfortable accommodations matter more to us than an occasional highly-rated meal — our first-order priority is every stay rated 4 or higher. Ranked below, we'd also like at least one restaurant rated 3.5 or higher."
    (Strict-priority ranking, expressed with rank words — never as a fallback, never with additive/mixed phrasing.)

(6b) LEXICOGRAPHIC (same entity + same attribute; ranking of alternatives) — `Transportation.mode = Flight [all] ≻ Transportation.mode ∈ {taxi} [all]`
    rationale: "Air travel cuts transit time and fatigue."
    verbalization: "On transportation, air travel cuts transit time and fatigue — our decisive first choice is every inter-city leg as a flight. Taxi is next in the ranking but distinctly behind."
    (Same entity+attribute with different values → phrase as pure ranking of alternatives. No additive "on top of that" phrasing. No mixed-satisfaction comparisons like "some legs flight and some legs taxi" — the source data doesn't model mixed modes across legs.)

(7) COMPENSATORY — same-entity secondary — `PRIMARY: Attraction.rating ≥ 4.5 | MARGIN: Attraction.rating ≥ 4.0 | SECONDARY: Attraction.category ∈ {Sights & Landmarks, Nature & Parks}`
    rationale (generic): "This trip is angled toward photography, where category significance matters more than a small rating shortfall — a slightly lower-rated iconic landmark can produce better photos than a highly-rated escape room."
    verbalization: "This trip is angled toward photography, and category significance tends to matter more than a small rating shortfall — a slightly lower-rated iconic landmark can produce better photos than a highly-rated escape room. So we'd like every attraction rated 4.5 or higher ideally; we'd accept a drop as far as 4.0 only when the same attraction is a Sights & Landmarks or Nature & Parks spot, where the category makes up for the shortfall."
    (All three thresholds and the compensating category set survive; the rationale's illustrative 4.1/4.8 numbers are generalized; universal "every" wording for Primary/Margin is correct — Compensatory is trip-wide on the primary entity.)

(8) COMPENSATORY — cross-entity secondary (same-day compensation) — `PRIMARY: Restaurant.aggregate_rating ≥ 4.5 | MARGIN: Restaurant.aggregate_rating ≥ 4.0 | SECONDARY: Attraction.category ∈ {Museums, Sights & Landmarks}`
    rationale (generic): "a slightly weaker meal is redeemed when the day itself already features a strong cultural anchor"
    verbalization: "We'd ideally like every restaurant rated 4.5 or higher; we'd accept a drop as far as 4.0 only on days that also have a Museums or Sights & Landmarks attraction on the schedule — a cultural anchor on the same day redeems a slightly weaker meal."
    (Universal "every restaurant" for Primary/Margin — trip-wide on the primary entity. Cross-entity compensation is same-day existential only — never phrase as "somewhere on the trip" for the Secondary.)

(9) SCOPED — `WHEN Restaurant.meal_type = dinner: Restaurant.cost ≥ 25`
    rationale: "Dinner is the occasion meal — worth going up on."
    verbalization: "Dinner is the occasion meal of the day for us, so specifically at dinner, we'd like restaurants running at $25 or more per person per meal — lunch and breakfast can stay modest."
    (The scope filter carves out only dinner slots; the inner threshold applies INSIDE that scope, not to breakfast / lunch.)

(10) TEMPORAL — `always(Attraction.rating ≥ 4.5) [scope: global]`
    rationale: "We don't want to waste a single day on mediocre experiences."
    verbalization: "We don't want to waste a single day on mediocre experiences — ideally every day of the trip has at least one attraction rated 4.5 or higher."

(11) TEMPORAL — `sometime(Attraction.category ∈ {Museums}) [scope: per_city]`
    rationale (generic): "wanting at least one culturally significant museum visit per city"
    verbalization: "Wanting at least one culturally significant museum visit per city, ideally on some day in each city we visit, at least one attraction on the schedule is a museum."

(12) TEMPORAL — `atmost_once(Restaurant.cuisine ∈ {French}) [scope: global]`
    rationale: "French food is too rich to eat multiple days in a row."
    verbalization: "French food is too rich to eat multiple days in a row, so across the trip we'd rather have French on at most one day."

(13) TEMPORAL — `hold_after(Restaurant.cost ≥ 45, time_start=2) [scope: global]`
    rationale: "A celebratory close to the trip with higher-spend dining at the tail end."
    verbalization: "As a celebratory close to the trip, from day 2 onward we'd ideally like every restaurant running at $45 per person per meal or higher — saving the higher-spend dining for the trip's back half."

(14) TEMPORAL — `hold_during(Attraction.category ∈ {Sights & Landmarks}, time_start=1, time_end=2) [scope: per_city]`
    rationale: "Preferred landmarks are best when energy is fresh."
    verbalization: "Preferred landmarks are best when energy is fresh, so during days 1 through 2 in each city, we'd ideally like every attraction we schedule to be a Sights & Landmarks spot."

(15) TEMPORAL — `sometime_before(Attraction.category = Museums, Attraction.category = Shopping) [scope: global]`
    rationale: "Cultural context makes purchases feel meaningful."
    verbalization: "Cultural context makes purchases feel meaningful, so we'd like a museum visit on some day before any shopping day in the trip."
    (Day-step: `global` scope, no day_as_step override → the ordering is day-by-day.)

(16) TEMPORAL — `sometime_after(Attraction.category ∈ {Museums, Sights & Landmarks}, Attraction.category ∈ {Shopping}) [scope: per_city]`
    rationale (generic): "immerse first, buy later — souvenirs feel meaningful after the cultural context"
    verbalization: "For souvenirs to feel meaningful, in each city we'd ideally like a Museums or Sights & Landmarks stop on some day after a Shopping stop within that city."

(17) TEMPORAL — `always_within(Restaurant.aggregate_rating ≤ 3.5, Restaurant.aggregate_rating ≥ 3.5, time_end=1) [scope: per_day]`
    rationale: "Quality bounce-back — a below-threshold meal must be followed by a recovery."
    verbalization: "As a quality bounce-back rhythm within each day, whenever a restaurant rated 3.5 or lower lands on the schedule, we'd like the very next dining slot that same day to be rated 3.5 or higher."
    (`per_day` operator scope forces slot-step: prose uses "the very next dining slot", not "the next day".)

(18) TEMPORAL — `within(Attraction.category ∈ {Museums, Sights & Landmarks, Classes & Workshops}, time_end=3) [scope: global]`
    rationale (generic): "cultural and educational experiences land better early on"
    verbalization: "Cultural experiences land better early on, so within the first 3 days of the trip, we'd ideally like at least one attraction to be a Museums, Sights & Landmarks, or Classes & Workshops spot."

(19) TEMPORAL — `sometime(Attraction.category ∈ {Concerts & Shows}) [scope: week_group]`
    rationale: "Music/theatre fan travelled specifically for a show."
    verbalization: "We're specifically here for the live-show scene, so ideally on some day of the weekends of the trip, at least one attraction is a Concerts & Shows event."
    (`week_group` scope — the sub-scope is weekend-days vs weekday-days; the *"some day"* wording matches sometime's day-step.)

(20) TEMPORAL — `always(Attraction.rating ≥ 4.5) [scope: travel_phase]` with the phase = *stay*
    rationale: "The stay days are our chance to actually spend time somewhere — we don't want them wasted."
    verbalization: "The stay days are our chance to actually spend time somewhere, so on each stay day of the trip, we'd ideally like at least one attraction rated 4.5 or higher — arrival and departure days can be looser."

TONE — TWO REGISTERS, EACH IN ITS OWN SENTENCES:

(1) LOCKED-IN TRIP FACTS are firm requirements. Phrase each one ONCE as a natural declarative need, then move on. Do NOT reassert, emphasize, tail-clause, or explain-again:
    AVOID reassertion tails: "— that's essential", "— that's a requirement", "— both are requirements", "— non-negotiable", "— we can't compromise on this", "— we need this", "— that's a must", ", so <restated same thing>". If the same constraint is said twice in one sentence or paragraph, cut the second copy.
    AVOID too-aggressive rulebook wording: "non-negotiable", "absolutely", "no exceptions", "strictly", "won't compromise on", "hard rule".
    AVOID too-weak background-fact wording that reads as passing description: "the accommodation allows smoking", "the room type is an entire room", "we're going with X". These are undistinguishable from preferences.

    Two constraint kinds — the wording strength differs:

    (a) AND single-value hard constraints — house rule (e.g. "smoking"), room type (e.g. "entire home/apt"), transportation (e.g. "no self-driving").
        These are strict: the single stated value must hold across the whole trip. Use firm but calm need-wording, once per constraint, no tail-clause.
        Good: "The accommodation needs to allow smoking."
        Good: "We need an entire-room booking for the whole stay."
        Good: "We're not self-driving on this trip."
        Bad (reasserts): "The accommodation needs to allow smoking — that's essential."
        Bad (reasserts): "We're not self-driving on this trip, so any transportation between cities has to be non-self-driving."
        Bad (too weak): "The accommodation allows smoking."

    (b) Set-based hard constraint — cuisine (a list of cuisines).
        The list defines the cuisines the trip is oriented around; a valid plan draws from these options. It is NOT an AND requirement that every listed cuisine must appear. Do NOT use "must cover all", "all four cuisines have to show up", "each of these has to appear", "the trip has to cover X, Y, and Z" (which reads as AND-strict).
        Good: "The trip revolves around Italian, French, Mexican, and Chinese cuisine."
        Good: "For food, we're keeping to Italian, French, Mexican, and Chinese."
        Good: "We're eating our way through Italian, French, Mexican, and Chinese cuisine on this trip."
        Bad (AND-strict): "The trip has to cover Italian, French, Mexican, and Chinese cuisine."
        Bad (AND-strict + reassertion): "The trip has to cover Italian, French, Mexican, and Chinese cuisine, so all four cuisines need to show up across the meals."

(2) PREFERENCES are SOFT and ASPIRATIONAL, even when scope is [all]:
    USE: "I'd ideally like", "we'd prefer", "hoping for", "leaning toward", "where possible", "ideally", "if we can", "as a target", "would be nice to", "with an eye toward", "the plan is to"
    AVOID: "must", "need to", "have to", "required", "essential", "non-negotiable", "every X has to", "X is required", "hard rule"
    For [all]-scope preferences, express the universal target as a wish — "ideally every meal is rated 4.5 or better" (not "every meal must be rated 4.5+"). The universal "every / each / throughout" survives; only the demand language softens.

SENTENCE-LEVEL SEPARATION (STRICT):
Every sentence in the query is EITHER a constraint sentence OR a preference sentence — never both. Do NOT combine a constraint clause and a preference clause in the same sentence, even with "and" / semicolons / dashes. If a constraint and a preference are on the same topic (e.g. accommodation), place the constraint sentence first, then the preference sentence right after it — adjacent, but structurally separate.
- WRONG: "The accommodation has to allow smoking, and I'd ideally like every stay under $150 per person per night."   ← constraint + preference in one sentence
- RIGHT: "The accommodation has to allow smoking. On top of that, I'd ideally like every stay to come in under $150 per person per night."
- WRONG: "We need Mexican and Chinese cuisine covered, but I'd hope every restaurant is rated 3.0 or better."   ← same
- RIGHT: "We need Mexican and Chinese cuisine covered across the meals. That said, I'd hope every restaurant we sit down to is rated 3.0 or better."
The trip-anchoring opener (framing, origin, destination, dates, duration, travelers, budget) is neither a constraint nor a preference — it can stand as its own opening sentence(s).

STRUCTURE — ONE OR TWO NATURAL PARAGRAPHS, TOPICALLY INTERLEAVED:
Do NOT front-load ALL constraint sentences and then dump ALL preference sentences (that recreates the two-block layout). Instead, interleave BY TOPIC — one topic at a time:
- Group by topic (accommodation, food/cuisine, attractions, transportation, budget, day-by-day rhythm, etc.). Within each topic, put the constraint sentence(s) first and the preference sentence(s) immediately after, so a topic-block might look like "[accommodation constraint sentence]. [accommodation preference sentence]." then move on to the next topic.
- Open by anchoring the trip (framing + origin + destination + dates + duration + travelers + budget). Both start and end dates must appear explicitly ("from <start> to <end>", "<start> through <end>", etc.), along with the number of days.
- Omit any locked-in item marked "(none)".
- Total output: 1–2 flowing paragraphs. Vary sentence openings; don't number preferences.

CONNECTING TWO PREFERENCES ON THE SAME TOPIC — REFINEMENT, NOT RESTATEMENT:
When two DIFFERENT structural preferences (e.g. a Composite and a Scoped, or an Atomic and a Compensatory) touch the SAME topic (both about attractions, both about restaurants, etc.), keep them in SEPARATE sentences — never merge into one clause — but the second sentence must READ AS A REFINEMENT of the first, not a parallel restatement. Two structurally-independent preferences on the same topic are usually related by narrowing (subset), scope (only on certain days / meal slots), tightening (existential highlight on top of a universal floor), or ranking (priority ordering). Use a CONNECTIVE that shows the relationship:
- Subset narrowing:  *"Within that, on stay days we'd specifically like … "*  /  *"Among those, we'd narrow to … "*  /  *"More specifically, on … we'd focus on … "*.
- Scope carve-out:  *"On stay days in particular, we'd like … "*  /  *"Specifically at dinner, … "*  /  *"In the second city, … "*.
- Existential highlight over universal:  *"On top of that, at least one … "*  /  *"And within those, we'd love one to really stand out at … "*.
- Priority ranking:  *"Above all else, … "*  /  *"That's the top priority; below it, … "*.
NEVER produce two consecutive sentences that just repeat "we'd ideally like every X" or "we'd like at least one X" with a different value set — that reads as repetitive parallel structure. If the two preferences cover overlapping categories on the same entity, the second sentence should call out what makes it DIFFERENT (a stricter subset, a scope carve-out, an existential highlight) rather than opening the same way with different values.

  BAD (parallel restatement):
    "We'd ideally like every attraction to be a museum, nature park, or landmark stop.
     On stay days we'd also like museums or nature parks."
  GOOD (refinement, explicit relationship):
    "We'd ideally like every attraction to be a museum, nature park, or landmark stop, rated 4.5 or higher.
     Within those, on stay days specifically, we'd focus on museums or nature parks — longer-visit attractions make the most sense when we actually have time to enjoy them."

  BAD:
    "We'd like every restaurant to serve Italian, Mexican, or Chinese.
     For dinner, we'd like Italian or Mexican."
  GOOD:
    "We'd like every restaurant to serve Italian, Mexican, or Chinese.
     Specifically at dinner, we'd narrow that to Italian or Mexican — a bit more of an occasion feel."

CONCISENESS — SAME INFORMATION, TIGHTER PHRASING:
The query should be coherent but not verbose. Every preference must still carry BOTH its concrete predicate (values, thresholds, operators, sets, scopes) AND its rationale, but state them concisely rather than expanded across multiple padded clauses. Filler transitions ("On top of that,", "Alongside that,", "That said,", "For food,") are fine when they help the flow — a real traveler uses them. What to cut is REDUNDANCY, not information:
- Don't split a predicate and its rationale into two long consecutive sentences that repeat each other. Merge them: "for real, ordinary food, ideally every meal we sit down to is Chinese or Mexican and rated 3.0 or higher" (rationale + predicate in one tight sentence).
- Don't restate the same threshold in different words ("$100 per person per night or less; keeping overnight spending tight" — the intent is already implied by the threshold once). One tight phrase is enough.
- Don't expand a predicate into an operational explanation ("if a meal comes in at 3.5 or lower, the next dining slot that day should be at 3.5 or higher — a recovery within one slot rather than letting two mediocre meals stack up" — the trailing "a recovery within one slot ..." just re-explains the predicate).
- Keep rationale content when it adds motivation beyond the predicate ("wants real ordinary food", "as a celebratory finale", "given the mid-range is worst value") — but keep it short: a clause, not a paragraph.
- A rationale that clashes with the trip context still gets generalized (per the RATIONALES section), not dropped.
Rule of thumb: each preference is ONE tight sentence containing predicate + brief rationale clause, unless the predicate itself is complex enough (lex, compensatory, conditional) to justify two.

DRIFT — NEVER EXPLICIT:
The structured input labels each preference with a drift_state (aligned / concession / self_compromise / fallback_hedge). This is an internal diagnostic; do NOT reveal it or hint at compromise/trade-off/give-and-take/"agreed on"/"settled on"/"against my usual inclination"/etc. Every preference reads in the same soft aspirational register. The profile↔preference gap is exactly what downstream systems will be tested on — don't spot it for them.

PRONOUN AGREEMENT WITH TRIP CONTEXT (STRICT):
Even though we drop drift-indicating wording, pronoun choice MUST match the trip context. Do NOT default to "I" on a multi-person trip or "we" on a solo trip — this creates an unintentional mismatch signal.
- If people_number == 1 (solo trip): use "I" / "me" / "my" throughout. Never say "we" or "us".
- If people_number > 1 (partner / family / friend / group trip): use "we" / "us" / "our" for constraints, preferences, and rationale clauses. Never slip into "I".
- Apply this pronoun rule to rationales too. On a 2-person trip a rationale like "keeping overnight spending frugal is the mode I'm in" should read "the mode we're in".

OPENING SENTENCE — TWO ALLOWED FORMS:
The opener (the trip-anchoring first sentence) is exempt from the strict "we"-only rule on multi-person trips, and may take EITHER of two forms — pick freely:

(a) FIRST-PERSON — the speaker introduces the plan as themselves. Uses "I" regardless of group size (the speaker is the one writing the request, even for a multi-person trip). Examples:
   - "I'm planning a solo adventure from Washington to Tampa, 3 days from ..."
   - "I'm planning a two-person trip from Houston to Colorado, 5 days from ..."
   - "I'm putting together a trip with my partner from Cleveland to Florida, 5 days from ..."

(b) AGNOSTIC / ADDRESSED-TO-ASSISTANT — a neutral request framed at the assistant, with no pronoun for the traveler(s). Uses no "I" or "we". Examples:
   - "Help create a plan for a 3-day solo trip from Washington to Tampa, ..."
   - "Kindly assist in creating a plan for a 5-day two-person trip from Houston to Colorado, ..."
   - "Could you help with creating a plan for a 5-day trip from Cleveland to Florida with my partner, ..."
   - "Provide a plan for a 3-day trip from Harrisburg to Detroit with a longtime friend, ..."

OPENER MUST BE TRAVEL MODE- AND DAY-AGNOSTIC. The opener must NOT presuppose a transportation MODE (that's for locked-in constraints or preferences later) or a WEEK-GROUP day-type (that's determined by the actual dates). Strip such words from the `trip_context_phrase` if it carries them.
   - AVOID travel mode hints: *"road trip"*, *"driving trip"*, *"flying getaway"*, *"flight vacation"*, *"rail trip"*, *"train journey"*, *"bus tour"*.
   - AVOID day-type hints: *"weekend"*, *"weekday"*, *"weekend getaway"*.
   - USE neutral descriptors from and faithful to the trip_context_phrase: *"trip"*, *"getaway"*, *"vacation"*, *"adventure"*, *"holiday"*, *"retreat"*, *"reunion"*, *"two-person trip"*, *"couple's getaway"*, *"family trip"*.
   - Example: *"two-person road trip from Houston to Colorado, 5 days from ..."* → *"two-person trip from Houston to Colorado, 5 days from ..."*. *"weekend with a close friend"* → *"trip with a close friend"*.

After the opener, immediately switch to the strict pronoun rule for the trip context (solo → "I"; multi-person → "we"). Do not mix within the same sentence: e.g. on a 2-person trip, "I'd ideally like every stay under $100 per person per night" is WRONG — write "we'd ideally like every stay under $100 per person per night". Choose ONE opener form per query; do not blend them.

RATIONALES — REPURPOSE, DO NOT DROP:
Each preference may include a rationale (the motivation the augmenter chose for it). Rationales sometimes mention specifics that don't match the trip context (e.g. a rationale about a "newlywed anniversary" attached to a solo trip, or an "80K-subscriber travel blogger" identity on a family trip). You have two jobs:
- If the rationale sits naturally alongside this trip context, weave in the motivating idea as a brief clause ("this is a photography-focused trip so I'd like well-rated spots").
- If the rationale mentions identity/relationship/audience specifics that clash with the trip context, GENERALIZE its underlying motivation instead of dropping it entirely — extract the abstract "why" (photography quality, unhurried immersion, budget discipline, culinary curiosity, sensory recovery, etc.) and phrase it in trip-neutral terms. Never invent new identity claims or reference the mismatch. Never say "I'm a travel blogger" or "for our anniversary" — instead say "this trip is angled toward photography" or "wanting to make the meals feel like an occasion".
- Never mark a rationale as generalized, softened, dropped, or adjusted. Never label its drift_state.

RATIONALE — GENERALIZE ANY CONCRETE VALUES INSIDE THE RATIONALE TEXT (NUMERIC AND CATEGORICAL) THAT DON'T MATCH THE RESOLVED PREDICATE:
Rationales are bank-authored motivation prose kept verbatim, so they can still contain illustrative values — numeric OR categorical — that were true of the bank's original template but have since been filtered/adjusted out of the resolved predicate. These are *motivating illustrations*, NOT enforced values — the enforced values live in the resolved predicate and are already surfaced from there. If you copy an illustrative value from the rationale into the NL, it can CLASH with the resolved predicate (e.g. mentioning a mode / cuisine / category that was constraint-filtered out) and confuse the reader. So:

- REPLACE any concrete NUMERIC value inside the rationale text with a generic descriptor before weaving it in:
    - "a 4.1-rated iconic landmark" → "a slightly lower-rated iconic landmark"
    - "a 4.8-rated escape room"     → "a highly-rated escape room"
    - "$75 dinner"                    → "a pricier dinner"
    - "80K-subscriber blogger"      → generalize / drop the identity part per the rule above.

- REPLACE any concrete CATEGORICAL value in the rationale that isn't in the resolved predicate's value set (mode, cuisine, category, house-rule flag, room type, etc.):
    - If the resolved predicate is `Transportation.mode ∈ {taxi}` but the rationale mentions "taxi or self-driving", generalize to "long road journeys" or just "taxi" — never re-introduce "self-driving" (which the constraint filter removed).
    - If the resolved predicate is `Restaurant.cuisine ∈ {French}` but the rationale mentions "French or Italian dinner", generalize to "a European dinner" or just "French" — do not re-introduce "Italian".
    - If the rationale mentions a category that isn't in the resolved predicate's value set, either drop that specific value or paraphrase to a class-level descriptor (e.g. "spa treatment" → "a low-intensity recovery activity" when the resolved predicate no longer has "Spas & Wellness").
  Reference the resolved predicate — NOT the rationale — as the source of truth for which specific values are in play. The rationale supplies motivation, not enforcement.

- KEEP concrete values ONLY when they are present in the resolved predicate. Those already survive in the predicate verbalization and must be present exactly there.
- The rationale is a MOTIVATION clause, not a second source of values.

RATIONALE — PRESERVE THE RESOLVED PREDICATE'S SCOPE AGAINST MISMATCHED RATIONALE WORDING:
Rationales frequently carry universal wording ("every X", "each X", "everywhere", "always", "only visit …", "consistently") or existential wording ("at least one X", "one meaningful X", "some X", "occasionally", "somewhere"). If the rationale's implied quantifier does NOT match the resolved predicate's `[scope: any]` / `[scope: all]` tag, THE PROSE MUST FOLLOW THE RESOLVED SCOPE — the resolved predicate's scope is paramount and must never be altered. The rationale supplies the "why" (motivation, character of the preference), never the "how many" (scope). Absorb the motivating idea and strip the mismatched quantification.

- Resolved `[scope: any]` + rationale wording suggests *every / each / everywhere / all / only visit / consistently* → the prose STILL renders as *"at least one …"*, *"somewhere on the trip"*, *"some …"*. Do NOT smuggle "everywhere", "every", "each", "only" from the rationale into the prose for that predicate.
- Resolved `[scope: all]` + rationale wording suggests *at least one / occasionally / one X / some* → the prose STILL renders as *"every …"*, *"each …"*, *"throughout the trip"*. Do NOT smuggle "at least one" / "some" from the rationale into the prose for that predicate.
- Unscoped predicate → treat as universal per the rendering cheatsheet regardless of what the rationale implies.

Worked example (the failure mode this rule fixes; the lexicographic prose follows the RANKING-language rules already stated for Lexicographic — NOT fallback / conditional phrasing):
    Resolved: `Attraction.category ∈ {Museums, Sights & Landmarks} [scope: any] ≻ Attraction.rating ≥ 4.5 [scope: any]`
    Rationale: "Atmosphere and social energy matter more than whether every attraction is top-rated; would rather have memorable experiences than only visit highly reviewed places."
    WRONG (rationale wording leaked onto a `[any]` predicate): "we'd prefer attractions like Museums or Sights & Landmarks over chasing 4.5-or-higher ratings **everywhere**" — "everywhere" imports the rationale's universal framing on top of an existential predicate.
    ALSO WRONG (rewritten as fallback, not lexicographic): "if a Museum or Sights & Landmark isn't on offer, at least one 4.5-or-higher attraction is the acceptable step-down" — this reads as substitution, but lex means both preferences remain in play and rank; the secondary is not a substitute for the primary.
    RIGHT (resolved `[any]` scope preserved, lexicographic ranking language used, and the Lexicographic `[any]` override applied — no *"at least one"*): "our first-order priority is having one or more attractions be Museums or Sights & Landmarks; ranked comfortably below that but still on our list, one or more attractions rated 4.5 or higher would land well too. Atmosphere and interactive experiences of museums / landmarks carry more weight for us than pure rating chases."
    Notes on the right rendering: (i) both `[any]` predicates read existentially with a maximizing-neutral quantifier (*"one or more …"*, per the Lexicographic `[any]` override) — NOT *"at least one"*, which would smuggle in a minimizing lower-bound reading incompatible with strict-priority satisfaction; (ii) neither is quantified as universal in the predicate clause; (iii) ranking language ("first-order priority", "ranked comfortably below") does the lex work; (iv) the rationale's motivation appears only in a trailing clause phrased in trip-neutral terms, and its "every stop" / "only visit" wording is generalized ("pure rating chases") so it never contradicts the two `[any]` predicates.

TRIP FACTS (must all appear in the prose, both dates explicitly):
- Framing phrase: {trip_context_phrase}
- From: {org}
- To: {dest}
- Duration: {days} days
- Start date (inclusive): {start_date}
- End date (inclusive): {end_date}
- Cities to visit: {visit_n}
- Travelers: {people_number}
- Budget: ~${budget}   (TOTAL trip budget across every traveler, every night, every category — never per-person, per-night, or per-meal)

LOCKED-IN TRIP FACTS (must all appear as decisions-already-made; omit any "(none)" line):
{constraints}

PREFERENCES TO COVER (every predicate value below must survive into the prose in natural English; rationales repurposed per the rule above):
{preferences}

Write the trip request now as ONE or TWO flowing paragraphs. Output only the message — no preamble, no closing remarks."""


# --------------------------------------------------------------------------- #
# Backend abstraction
# --------------------------------------------------------------------------- #

class Backend:
    """Abstract base. ``generate_batch`` returns one text per prompt.

    ``on_result`` is an optional per-completion callback: as soon as a
    single prompt finishes, backends invoke it with ``(index, text)``
    so the caller can persist the result to disk immediately.  This
    makes Ctrl-C interruptions non-destructive -- everything completed
    up to that point has already been written."""
    name: str = "abstract"
    def generate_batch(self, prompts: list[str], *, max_tokens: int,
                       on_result=None) -> list[str]:
        raise NotImplementedError


class AnthropicBackend(Backend):
    name = "anthropic"
    def __init__(self, model: str, workers: int = 8):
        import anthropic
        # Disable SDK internal retries so `_one` handles 429s visibly.
        # Cap the per-request timeout so stuck sockets can't stall a
        # worker indefinitely.
        self.client     = anthropic.Anthropic(max_retries=0, timeout=120.0)
        self.model      = model
        self.workers    = workers

    def _one(self, prompt: str, max_tokens: int) -> str:
        """Single Anthropic call with explicit rate-limit handling.

        See OpenAIBackend._one for the rationale -- Anthropic's SDK also
        retries 429s silently under the hood.  Surface the wait to the
        user so a stalled progress bar has an explanation.
        """
        import re as _re, time as _time, sys as _sys, random as _random
        from anthropic import RateLimitError, APITimeoutError, APIConnectionError

        max_attempts = 8
        for attempt in range(max_attempts):
            try:
                msg = self.client.messages.create(
                    model       = self.model,
                    max_tokens  = max_tokens,
                    temperature = TEMPERATURE,
                    messages    = [{"role": "user", "content": prompt}],
                )
                out = []
                for b in msg.content:
                    if getattr(b, "type", None) == "text":
                        out.append(b.text)
                return "".join(out).strip()
            except RateLimitError as e:
                # Anthropic surfaces a retry-after header when available.
                retry_after = None
                resp = getattr(e, "response", None)
                if resp is not None and hasattr(resp, "headers"):
                    try:
                        retry_after = float(resp.headers.get("retry-after") or 0)
                    except (TypeError, ValueError):
                        retry_after = None
                wait = retry_after if retry_after else min(60.0, 2 ** attempt)
                wait = max(wait, 0.5) + _random.uniform(0.0, 0.5)
                print(f"\n[rate-limit] anthropic 429   sleeping {wait:.2f}s "
                      f"(attempt {attempt+1}/{max_attempts})",
                      file=_sys.stderr, flush=True)
                _time.sleep(wait)
                continue
            except (APITimeoutError, APIConnectionError) as e:
                wait = min(30.0, 2 ** attempt)
                print(f"\n[api-transient] {type(e).__name__}: "
                      f"retrying in {wait:.1f}s (attempt {attempt+1}/{max_attempts})",
                      file=_sys.stderr, flush=True)
                _time.sleep(wait)
                continue
        print(f"\n[error] Anthropic call failed after {max_attempts} attempts; skipping.",
              file=_sys.stderr, flush=True)
        return ""

    def generate_batch(self, prompts, *, max_tokens, on_result=None):
        results = [None] * len(prompts)
        reporter = _ProgressReporter(len(prompts),
                                     label=f"anthropic ({self.workers}w)")
        try:
            with ThreadPoolExecutor(max_workers=self.workers) as ex:
                futs = {ex.submit(self._one, p, max_tokens): i
                        for i, p in enumerate(prompts)}
                for f in as_completed(futs):
                    idx = futs[f]
                    try:
                        text = f.result()
                    except Exception as e:
                        text = ""
                    results[idx] = text
                    if on_result is not None and text:
                        try:
                            on_result(idx, text)
                        except Exception:
                            pass
                    reporter.step()
        finally:
            reporter.close()
        return results


class OpenAIBackend(Backend):
    name = "openai"
    def __init__(self, model: str, workers: int = 8,
                 base_url: str | None = None, api_key: str | None = None):
        from openai import OpenAI
        kwargs = {}
        if base_url is not None:
            kwargs["base_url"] = base_url
        if api_key is not None:
            kwargs["api_key"] = api_key
        # Disable the SDK's internal silent retry — we surface 429s
        # explicitly in `_one` so the progress bar can log the wait.
        # Also cap per-request timeout so a stuck connection can't
        # freeze a worker forever.
        kwargs["max_retries"] = 0
        kwargs["timeout"] = 120.0
        self.client  = OpenAI(**kwargs)
        self.model   = model
        self.workers = workers

    def _one(self, prompt: str, max_tokens: int) -> str:
        """Single OpenAI call with EXPLICIT rate-limit handling.

        The default OpenAI SDK retries 429s internally with exponential
        backoff -- silently, from the caller's POV.  On a heavily-loaded
        run that looks like the process 'stalled' for tens of seconds
        with no output.  We short-circuit that by catching RateLimitError
        directly, parsing the server's suggested wait, logging it to
        stderr so the user sees why the progress bar isn't advancing,
        and retrying.
        """
        import re as _re, time as _time, sys as _sys
        from openai import RateLimitError, APITimeoutError, APIConnectionError

        max_attempts = 8
        for attempt in range(max_attempts):
            try:
                resp = self.client.chat.completions.create(
                    model       = self.model,
                    messages    = [{"role": "user", "content": prompt}],
                    temperature = TEMPERATURE,
                    max_completion_tokens  = max_tokens,
                )
                return (resp.choices[0].message.content or "").strip()
            except RateLimitError as e:
                msg = str(getattr(e, "message", e))
                # Try to extract "Please try again in Xms" or "Xs".
                m = _re.search(r"try again in\s+([\d.]+)\s*(ms|s|m)", msg)
                if m:
                    v = float(m.group(1)); unit = m.group(2)
                    wait = v/1000.0 if unit == "ms" else (v*60.0 if unit == "m" else v)
                else:
                    wait = min(60.0, 2 ** attempt)
                # Small jitter + a floor so many workers don't wake up
                # exactly together and re-trigger the same limit.
                import random as _random
                wait = max(wait, 0.5) + _random.uniform(0.0, 0.5)
                # Trim the 4× wordier SDK message down to the diagnostic
                # essentials: which limit, how much used, wait duration.
                short = msg
                lim = _re.search(r"Limit\s+(\d+)", short)
                usd = _re.search(r"Used\s+(\d+)", short)
                req = _re.search(r"Requested\s+(\d+)", short)
                brief = f"limit={lim.group(1) if lim else '?'} used={usd.group(1) if usd else '?'} req={req.group(1) if req else '?'}" if (lim or usd or req) else short[:80]
                print(f"\n[rate-limit] {brief}   sleeping {wait:.2f}s "
                      f"(attempt {attempt+1}/{max_attempts})",
                      file=_sys.stderr, flush=True)
                _time.sleep(wait)
                continue
            except (APITimeoutError, APIConnectionError) as e:
                wait = min(30.0, 2 ** attempt)
                print(f"\n[api-transient] {type(e).__name__}: "
                      f"retrying in {wait:.1f}s (attempt {attempt+1}/{max_attempts})",
                      file=_sys.stderr, flush=True)
                _time.sleep(wait)
                continue
        # All retries exhausted
        print(f"\n[error] OpenAI call failed after {max_attempts} attempts; skipping.",
              file=_sys.stderr, flush=True)
        return ""

    def generate_batch(self, prompts, *, max_tokens, on_result=None):
        results = [None] * len(prompts)
        reporter = _ProgressReporter(len(prompts),
                                     label=f"openai ({self.workers}w)")
        try:
            with ThreadPoolExecutor(max_workers=self.workers) as ex:
                futs = {ex.submit(self._one, p, max_tokens): i
                        for i, p in enumerate(prompts)}
                for f in as_completed(futs):
                    idx = futs[f]
                    try:
                        text = f.result()
                    except Exception as e:
                        text = ""
                    results[idx] = text
                    if on_result is not None and text:
                        try:
                            on_result(idx, text)
                        except Exception:
                            pass
                    reporter.step()
        finally:
            reporter.close()
        return results


class VLLMOfflineBackend(Backend):
    """In-process vLLM batched inference.  Loads model weights once,
    then issues all prompts in a single ``LLM.chat()`` call so vLLM's
    continuous-batching scheduler can pack everything efficiently."""
    name = "vllm-offline"
    def __init__(self, model: str, *,
                 gpu_memory_utilization: float = 0.9,
                 max_model_len: int | None = None,
                 dtype: str = "auto",
                 tensor_parallel_size: int = 1,
                 trust_remote_code: bool = True,
                 **extra):
        from vllm import LLM
        self.LLM_class = LLM
        self.model = model
        self.llm = LLM(
            model                  = model,
            gpu_memory_utilization = gpu_memory_utilization,
            max_model_len          = max_model_len,
            dtype                  = dtype,
            tensor_parallel_size   = tensor_parallel_size,
            trust_remote_code      = trust_remote_code,
            **extra,
        )

    def generate_batch(self, prompts, *, max_tokens, on_result=None):
        from vllm import SamplingParams
        sp = SamplingParams(
            temperature = TEMPERATURE,
            max_tokens  = max_tokens,
        )
        # Use chat API so prompts get the model's chat template applied.
        conversations = [[{"role": "user", "content": p}] for p in prompts]
        outputs = self.llm.chat(conversations, sampling_params=sp,
                                use_tqdm=True)
        # vLLM returns RequestOutput in the same order as prompts.
        texts = [o.outputs[0].text.strip() for o in outputs]
        # vLLM batches internally, so we can only fire on_result after
        # the whole batch completes.  Still, this saves everything to
        # cache on run end (before any downstream merge step).
        if on_result is not None:
            for i, t in enumerate(texts):
                if t:
                    try:
                        on_result(i, t)
                    except Exception:
                        pass
        return texts


def make_backend(name: str, model: str, *,
                 workers: int = 8,
                 openai_base_url: str | None = None,
                 openai_api_key: str | None = None,
                 vllm_gpu_memory: float = 0.9,
                 vllm_max_model_len: int | None = None,
                 vllm_dtype: str = "auto",
                 vllm_tensor_parallel: int = 1) -> Backend:
    if name == "anthropic":
        return AnthropicBackend(model=model, workers=workers)
    if name == "openai":
        return OpenAIBackend(
            model    = model,
            workers  = workers,
            base_url = openai_base_url,
            api_key  = openai_api_key,
        )
    if name == "vllm-offline":
        return VLLMOfflineBackend(
            model                  = model,
            gpu_memory_utilization = vllm_gpu_memory,
            max_model_len          = vllm_max_model_len,
            dtype                  = vllm_dtype,
            tensor_parallel_size   = vllm_tensor_parallel,
        )
    raise ValueError(f"Unknown backend: {name!r}")


# --------------------------------------------------------------------------- #
# Input formatting
# --------------------------------------------------------------------------- #

def _join_or_none(items: list[str]) -> str:
    return " ".join(items) if items else "(none specified)"


def _profile_input(rec: dict) -> dict:
    p          = rec.get("profile") or {}
    interests  = p.get("Interests") or {}
    loc        = (p.get("Demographics") or {}).get("Location") or rec.get("org") or "the area"
    return {
        "location":     loc,
        "hobbies":      _join_or_none(interests.get("Hobbies") or []),
        "lifestyle":    _join_or_none(interests.get("Lifestyle") or []),
        "travel_style": _join_or_none(interests.get("Travel Style") or []),
        "pref_dests":   _join_or_none(interests.get("Preferred Destinations") or []),
        "food":         _join_or_none(interests.get("Food and Dining Preferences") or []),
        "dislikes":     _join_or_none(interests.get("Dislikes") or []),
        "pets":         (p.get("Pets") or "(no signal)"),
    }


def _drift_state_for_entry(entry: dict, trip_category: str) -> str:
    action = entry.get("action", "")
    if action == "template_fallback_hedge":
        return "fallback_hedge"
    if action.startswith("aligned"):
        return "aligned"
    if trip_category == "solo trip":
        return "self_compromise"
    return "concession"


def _query_input(rec: dict) -> dict:
    tc = rec.get("trip_context") or {}
    lc = rec.get("local_constraint") or {}

    date_list  = rec.get("date") or []
    start_date = date_list[0]  if date_list else "(unspecified)"
    end_date   = date_list[-1] if date_list else "(unspecified)"

    lc_lines = []
    if lc.get("cuisine"):
        cus = lc["cuisine"] if isinstance(lc["cuisine"], list) else [lc["cuisine"]]
        lc_lines.append(f"- cuisine: {', '.join(cus)}  (the trip must cover these cuisines)")
    if lc.get("house rule"):
        lc_lines.append(f"- house rule: {lc['house rule']}  (the accommodation must allow / be open to this)")
    if lc.get("room type"):
        lc_lines.append(f"- room type: {lc['room type']}  (the room type is fixed)")
    if lc.get("transportation"):
        lc_lines.append(f"- transportation: {lc['transportation']}  (the transportation choice is fixed)")
    constraints = "\n".join(lc_lines) if lc_lines else "(none)"

    pref_lines = []
    for i, entry in enumerate(rec.get("nl_render_log") or [], start=1):
        rationale = entry.get("rationale", "").strip()
        # Use the resolved predicate string only. `raw_source` (the
        # bank's original template string) is no longer emitted on the
        # augmented record, since it could diverge from post-resolution
        # values.
        raw_src = (entry.get("resolved_source") or "").strip()
        if not rationale and not raw_src:
            continue
        ds = _drift_state_for_entry(entry, tc.get("category", ""))
        # Drift state is INTERNAL — used only to decide how to handle the
        # rationale, never surfaced in the prose. Rationales are repurposed
        # (generalized to a trip-neutral motivation) rather than dropped.
        pref_lines.append(
            f"{i}. predicate (render into natural English, keeping every concrete value): {raw_src}"
        )
        if rationale:
            if ds == "fallback_hedge":
                pref_lines.append(
                    "   rationale (REPURPOSE — this specific rationale mentions "
                    "identity/relationship/audience specifics that clash with the "
                    "trip context. Extract only the ABSTRACT motivation and phrase "
                    "it in trip-neutral terms. Never quote it verbatim; never "
                    "invent new identity claims): "
                    f"{rationale}"
                )
            else:
                pref_lines.append(
                    "   rationale (weave in as a brief motivating clause if it "
                    "reads naturally alongside the trip; otherwise generalize its "
                    "underlying motivation into trip-neutral wording — do NOT drop "
                    "it entirely, and do NOT sound like a concession or trade-off): "
                    f"{rationale}"
                )
    preferences = "\n".join(pref_lines) if pref_lines else "(no preferences to cover)"

    return {
        "trip_context_phrase": tc.get("phrase", "trip"),
        "org":                 rec.get("org", "?"),
        "dest":                rec.get("dest", "?"),
        "days":                rec.get("days", "?"),
        "visit_n":             rec.get("visiting_city_number", 1),
        "people_number":       rec.get("people_number", 1),
        "start_date":          start_date,
        "end_date":            end_date,
        "budget":              int(rec.get("budget", 0)),
        "constraints":         constraints,
        "preferences":         preferences,
    }


def _render_prompt(tmpl: str, **kwargs) -> str:
    """Substitute only the named `{key}` placeholders; leave every other
    `{...}` intact. The prompts contain many literal curly braces from
    structured predicate notation (e.g. `{Nightlife, Museums}`,
    `{V1, V2}`), which would collide with `str.format()`. Doing a plain
    per-key replace avoids that entirely."""
    out = tmpl
    for k, v in kwargs.items():
        out = out.replace("{" + k + "}", str(v))
    return out


def build_profile_prompt(rec: dict) -> str:
    return _render_prompt(PROFILE_PROMPT, **_profile_input(rec))


def build_query_prompt(rec: dict) -> str:
    return _render_prompt(QUERY_PROMPT, **_query_input(rec))


# --------------------------------------------------------------------------- #
# Caching
# --------------------------------------------------------------------------- #

def _load_cache(cache_path: Path) -> dict:
    cache: dict[tuple[int, str], str] = {}
    if not cache_path.exists():
        return cache
    with cache_path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                e = json.loads(line)
                cache[(e["qid"], e["kind"])] = e["text"]
            except Exception:
                continue
    return cache


def _append_cache(qid: int, kind: str, text: str, cache_path: Path) -> None:
    # Open in append mode with line-buffering; flush + fsync so a Ctrl-C
    # or a stalled request that gets killed by the user leaves every
    # already-completed prompt safely on disk.
    with cache_path.open("a") as f:
        f.write(json.dumps({"qid": qid, "kind": kind, "text": text}) + "\n")
        f.flush()
        try:
            import os as _os
            _os.fsync(f.fileno())
        except OSError:
            pass


# --------------------------------------------------------------------------- #
# Orchestration
# --------------------------------------------------------------------------- #

def generate_all(records: list[dict], backend: Backend, cache_path: Path, *,
                 verbose: bool = True) -> None:
    """For each record, produce (profile, query) and append to
    ``cache_path``.  ``cache_path`` is model-scoped by default (see
    ``_model_output_paths``) so parallel runs with different backends
    stay isolated.

    All pending prompts of both kinds are accumulated and dispatched in a
    single ``backend.generate_batch`` call.  This lets the vLLM offline
    backend pack everything into one engine pass; the API backends use
    their internal thread-pool concurrency."""
    cache = _load_cache(cache_path)
    if verbose:
        n_p = sum(1 for k in cache if k[1] == "profile")
        n_q = sum(1 for k in cache if k[1] == "query")
        print(f"[cache] {cache_path.name}: profile={n_p}, query={n_q}")

    # Build pending list
    pending: list[tuple[int, str, str, int]] = []
    # (qid, kind, prompt, max_tokens)
    for rec in records:
        qid = rec["query_id"]
        if (qid, "profile") not in cache:
            pending.append((qid, "profile", build_profile_prompt(rec), 2048))
        if (qid, "query") not in cache:
            pending.append((qid, "query",   build_query_prompt(rec),   2048))

    if verbose:
        print(f"[run] backend={backend.name}  pending={len(pending)}  "
              f"already_cached={len(records) * 2 - len(pending)}")
    if not pending:
        return

    prompts    = [p for (_, _, p, _) in pending]
    max_tokens = max(mt for (_, _, _, mt) in pending)

    # Serialize cache writes across worker threads so appends can't
    # interleave partial JSON lines on disk.
    import threading
    _cache_lock = threading.Lock()

    def _persist(index: int, text: str) -> None:
        """Called from the backend as soon as prompt ``index`` finishes.
        Appends the result to the cache file immediately -- so any
        Ctrl-C partway through leaves every completed prompt on disk."""
        qid, kind, _, _ = pending[index]
        with _cache_lock:
            _append_cache(qid, kind, text, cache_path)

    t0 = time.time()
    if verbose:
        print(f"[run] dispatching {len(prompts)} prompts...")
    try:
        texts = backend.generate_batch(prompts, max_tokens=max_tokens,
                                        on_result=_persist)
    except KeyboardInterrupt:
        # backends may not fully drain futures on Ctrl-C; the cache
        # holds whatever _persist saw before the interrupt.
        print(f"\n[run] interrupted; cache holds all completions so far",
              file=sys.stderr)
        raise
    dt = time.time() - t0
    if verbose:
        print(f"[run] generated {len(texts)} texts in {dt:.1f}s "
              f"({len(texts)/max(dt, 1e-6):.2f} prompts/sec)")


def merge_into_jsonl(records: list[dict], in_path: Path, out_path: Path,
                     cache_path: Path) -> None:
    """Read the JSONL again (to preserve any records not in our slice),
    annotate from ``cache_path``, and write to ``out_path``.  ``out_path``
    is model-scoped by default so parallel runs stay isolated."""
    cache = _load_cache(cache_path)
    full = []
    with in_path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            full.append(json.loads(line))
    n_p = n_q = 0
    for rec in full:
        qid = rec["query_id"]
        if (qid, "profile") in cache:
            rec["llm_nl_profile"] = cache[(qid, "profile")]; n_p += 1
        if (qid, "query") in cache:
            rec["llm_nl_query"]   = cache[(qid, "query")];   n_q += 1
    with out_path.open("w") as f:
        for rec in full:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    print(f"[out] merged {n_p} llm_nl_profile, {n_q} llm_nl_query into "
          f"{len(full)} records → {out_path}")


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #

def main():
    ap = argparse.ArgumentParser(
        description=__doc__.splitlines()[0],
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--backend", choices=["anthropic", "openai", "vllm-offline"],
                    default=os.environ.get("LLM_NL_BACKEND", "anthropic"))
    ap.add_argument("--model", default=None,
                    help="Model name. Defaults: anthropic→claude-haiku-4-5-20251001, "
                         "openai→gpt-5.4-nano, vllm-offline→meta-llama/Llama-3.1-8B-Instruct")
    ap.add_argument("--workers", type=int, default=4,
                    help="Concurrent in-flight requests for API backends.")
    ap.add_argument("--sample", type=int, default=0,
                    help="If >0, only generate for this many records (smoke test).")
    ap.add_argument("--sample-qids", type=str, default=None,
                    help="Comma-separated list of query_ids to restrict to, "
                         "e.g. '2,461,764'. Overrides --sample when set. Also "
                         "accepts a path to a text file with one qid per line "
                         "or a comma-separated qid list.")
    ap.add_argument("--in",  dest="in_path",  default=str(IN_PATH))
    # --out / --cache default to MODEL-SCOPED paths (see _model_output_paths),
    # resolved after --model / --backend are parsed.  Leave unset to let each
    # backend/model write to its own file and cache; parallel runs with
    # different models stay isolated automatically.  Pass an explicit path
    # (e.g. --out prefertripplan.jsonl) to opt into shared output.
    ap.add_argument("--out",   dest="out_path",   default=None,
                    help="Output JSONL. Defaults to prefertripplan.<model_slug>.jsonl.")
    ap.add_argument("--cache", dest="cache_path", default=None,
                    help="Cache JSONL (resumable). Defaults to llm_nl_cache.<model_slug>.jsonl.")
    # OpenAI-specific
    ap.add_argument("--openai-base-url", default=os.environ.get("OPENAI_BASE_URL"),
                    help="OpenAI-compatible endpoint URL (e.g. http://localhost:8000/v1).")
    # vLLM-offline-specific
    ap.add_argument("--vllm-gpu-memory", type=float, default=0.9,
                    help="vLLM gpu_memory_utilization (0.0-1.0).")
    ap.add_argument("--vllm-max-model-len", type=int, default=None,
                    help="vLLM max_model_len.  Cap if your GPU can't fit the model's full context.")
    ap.add_argument("--vllm-dtype", default="auto",
                    choices=["auto", "float16", "bfloat16", "float32"],
                    help="vLLM dtype.")
    ap.add_argument("--vllm-tensor-parallel", type=int, default=1,
                    help="vLLM tensor_parallel_size.  Set to number of GPUs.")
    args = ap.parse_args()

    model = args.model or DEFAULT_MODELS[args.backend]
    # Resolve model-scoped defaults for --out and --cache when they weren't
    # explicitly provided.  This keeps concurrent multi-model runs isolated
    # by default while still allowing shared paths on explicit override.
    default_out, default_cache = _model_output_paths(model)
    out_path   = Path(args.out_path)   if args.out_path   else default_out
    cache_path = Path(args.cache_path) if args.cache_path else default_cache
    print(f"[cfg] backend={args.backend}  model={model}  workers={args.workers}")
    print(f"[cfg] out={out_path.name}  cache={cache_path.name}")

    records: list[dict] = []
    with open(args.in_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    if args.sample_qids:
        raw = args.sample_qids
        # If it's a readable file, load qids from it (one per line or CSV).
        p = Path(raw)
        if p.is_file():
            raw = p.read_text()
        wanted = {int(t.strip()) for t in raw.replace("\n", ",").split(",")
                  if t.strip()}
        records = [r for r in records if r.get("query_id") in wanted]
        missing = sorted(wanted - {r["query_id"] for r in records})
        print(f"[in]  --sample-qids: {len(records)} matched, "
              f"{len(missing)} not found: {missing[:10]}"
              + ("..." if len(missing) > 10 else ""))
    elif args.sample > 0:
        records = records[:args.sample]
    print(f"[in]  {len(records)} records loaded from {args.in_path}")

    backend = make_backend(
        args.backend,
        model,
        workers              = args.workers,
        openai_base_url      = args.openai_base_url,
        vllm_gpu_memory      = args.vllm_gpu_memory,
        vllm_max_model_len   = args.vllm_max_model_len,
        vllm_dtype           = args.vllm_dtype,
        vllm_tensor_parallel = args.vllm_tensor_parallel,
    )

    generate_all(records, backend, cache_path)

    merge_into_jsonl(records, Path(args.in_path), out_path, cache_path)


if __name__ == "__main__":
    main()
