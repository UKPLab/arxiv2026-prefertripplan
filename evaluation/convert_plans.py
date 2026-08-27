#!/usr/bin/env python3
"""
convert_plans.py — ``plans_*.jsonl`` → structured plan JSONL.

Reads ``plan-generation/generate_plans.py`` output (where
``llm_travel_plan`` is free-form Markdown-ish text emitted by an LLM,
sometimes wrapped in a ``{"travel_plan": "..."}`` JSON envelope) and
emits one structured record per row::

    {"id":        <int>,             # 1-indexed HF dataset id (passed through)
     "source_id": <int|null>,        # test row's pointer into test_large; null
                                     # on test_large (passed through)
     "query":     <str>,
     "plan":      [{"days": <int>,   # 1-indexed by default (matches "Day N:")
                    "current_city": ..., "transportation": ...,
                    "breakfast": ..., "attraction": ..., "lunch": ...,
                    "dinner": ..., "accommodation": ...}, ...]}

The plan text is read from the ``llm_travel_plan`` field only.  Any
``llm_reasoning`` is ignored — reasoning is not part of the plan.  The
``travel_plan`` key of the input JSON envelope (when present) is
unwrapped internally and NEVER appears in the converted output; the
output starts directly from the ``plan`` list of days.

Fields absent from a day block are filled with ``"-"``.  Plans that
couldn't be parsed (``llm_travel_plan`` is ``None``, an empty string,
or contains no ``Day N:`` marker) emit ``"plan": []`` — downstream
``eval.py`` reads that as an undelivered plan via its
``if tested_plan['plan']:`` check.

Indexing.  ``id`` / ``source_id`` are passed through verbatim from the
input — they are already 1-indexed in the post-migration
``plan-generation`` schema, and matching them 1:1 against the HF dataset
``id`` is the pairing invariant every downstream step relies on.  No
row-id shifting happens here.  The per-day ``days`` field defaults to
1-based (matching the ``"Day 1:"`` numbering in the LLM output); pass
``--day-base 0`` if you need a 0-based emit.

Legacy files with ``idx`` (0-indexed, pre-migration) instead of ``id``
are rejected with a clear error pointing to the one-off migration
script (``plan-generation/migrate_legacy_cache.py``).  Migrate the
plans file once, then re-run this converter.

Robustness features (based on surveying 450 real plans across 2 backends):
  * Unwraps ``{"travel_plan": "..."}`` envelopes (with un-escaping of
    round-tripped ``\\n`` and ``\\r\\n``).
  * Recovers PARTIAL plans from truncated envelopes (model was cut off
    at the max-token limit and never emitted a closing ``"}``): the
    prefix is regex-stripped and the surviving days are un-escaped and
    parsed.  Better a partial plan than an empty one — downstream
    evaluators can decide whether to score it.
  * Converts inline ``<br>`` breaks into real newlines.
  * Inserts a newline when ``Day N:`` and ``Current City:`` share a line.
  * Splits day blocks even without a blank-line separator.
  * Case-insensitive header matching; tolerates ``Attractions:`` /
    ``Accommodations:`` / ``Transport:`` / ``Lodging:`` variants.
  * De-duplicates repeated headers within a day (first wins).
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

# Canonical field order per day — matches example_format.jsonl exactly.
_DAY_KEYS: tuple[str, ...] = (
    "current_city", "transportation", "breakfast",
    "attraction", "lunch", "dinner", "accommodation",
)

# Header text (lowercased, stripped) → canonical key.  Includes the plural /
# alternate spellings actually observed in the plan corpus.
_HEADER_ALIASES: dict[str, str] = {
    "current city":   "current_city",
    "transportation": "transportation",
    "transport":      "transportation",
    "breakfast":      "breakfast",
    "attraction":     "attraction",
    "attractions":    "attraction",
    "lunch":          "lunch",
    "dinner":         "dinner",
    "accommodation":  "accommodation",
    "accommodations": "accommodation",
    "lodging":        "accommodation",
}

_DAY_HEADER_RE = re.compile(r"^\s*Day\s+(\d+)\s*[:.\-]?\s*(.*)$", re.IGNORECASE)
_KV_LINE_RE    = re.compile(r"^([A-Za-z][A-Za-z _/-]{0,40}?)\s*:\s*(.*)$")
_ENVELOPE_KEY  = "travel_plan"
_ENVELOPE_HEAD_RE = re.compile(
    r'^\s*\{\s*"' + re.escape(_ENVELOPE_KEY) + r'"\s*:\s*"',
    re.DOTALL,
)


# --------------------------------------------------------------------------- #
# Text normalisation                                                           #
# --------------------------------------------------------------------------- #
def _unwrap_envelope(text: str) -> str:
    """Strip a ``{"travel_plan": "..."}`` wrapper if present; else pass through.

    Falls back to a regex-based extract when ``json.loads`` fails, which
    happens when the model was truncated at the max-token limit and never
    emitted a closing ``"}``.  Recovers the (partial) plan text so days
    that DID make it through the token budget aren't lost.

    The ``travel_plan`` key is an INPUT artifact only — the caller
    receives the inner plan text and the key never surfaces in the
    converted output."""
    s = text.strip()
    if not s.startswith("{"):
        return text
    try:
        obj = json.loads(s)
        if isinstance(obj, dict) and isinstance(obj.get(_ENVELOPE_KEY), str):
            return obj[_ENVELOPE_KEY]
    except json.JSONDecodeError:
        pass
    # Truncated envelope — extract everything after the opening quote of
    # the travel_plan value and un-escape JSON string escapes ourselves.
    m = _ENVELOPE_HEAD_RE.match(s)
    if not m:
        return text
    inner = s[m.end():]
    # Drop trailing closing quote/brace if the envelope IS well-formed but
    # some other field prevented json.loads from succeeding.
    inner = re.sub(r'"\s*\}?\s*$', "", inner)
    return _json_unescape(inner)


def _json_unescape(s: str) -> str:
    """Left-to-right single-pass un-escape of JSON string escapes.

    Doing this with chained ``str.replace`` calls is unsafe: replacing
    ``\\n`` first turns ``\\\\n`` (an escaped backslash followed by 'n')
    into ``\\<newline>`` instead of the correct ``\\n`` literal — because
    ``\\\\ -> \\`` runs after and misses the already-consumed backslash.
    A single-pass walk avoids that class of bug entirely."""
    out: list[str] = []
    i, n = 0, len(s)
    while i < n:
        c = s[i]
        if c == "\\" and i + 1 < n:
            nxt = s[i + 1]
            out.append({
                "n": "\n", "r": "\n", "t": "\t",
                '"': '"',  "\\": "\\", "/": "/",
            }.get(nxt, c + nxt))
            i += 2
        else:
            out.append(c)
            i += 1
    return "".join(out)


def _normalize_text(text: str | None) -> str:
    """Turn every line-break dialect the LLM might have used into real ``\\n``."""
    if not text:
        return ""
    text = _unwrap_envelope(text)
    text = re.sub(r"<\s*br\s*/?\s*>", "\n", text, flags=re.IGNORECASE)
    text = text.replace("\\r\\n", "\n").replace("\\n", "\n")     # JSON-escaped
    text = text.replace("\r\n", "\n").replace("\r", "\n")        # CRLF / CR
    # "Day N: Current City: ..." on one line -> split them
    text = re.sub(r"(Day\s+\d+\s*:)\s+(?=Current City\s*:)",
                  r"\1\n", text, flags=re.IGNORECASE)
    return text


# --------------------------------------------------------------------------- #
# Structural parsing                                                           #
# --------------------------------------------------------------------------- #
def _split_days(text: str) -> list[tuple[int, str]]:
    """Return ``[(day_number, body_text), ...]`` in appearance order."""
    days: list[tuple[int, list[str]]] = []
    current: list[str] | None = None
    for line in text.split("\n"):
        m = _DAY_HEADER_RE.match(line)
        if m:
            day_num = int(m.group(1))
            trailing = m.group(2).strip()
            current = []
            days.append((day_num, current))
            if trailing:
                current.append(trailing)
            continue
        if current is not None:
            current.append(line)
    return [(n, "\n".join(body).strip()) for n, body in days]


def _parse_day_body(body: str) -> dict[str, str]:
    """Parse ``Header: value`` lines into a dict keyed by canonical field
    names, filling absent fields with ``"-"``."""
    out: dict[str, str] = {}
    pending_key: str | None = None
    buf: list[str] = []

    def flush() -> None:
        nonlocal pending_key, buf
        if pending_key is None:
            return
        val = " ".join(x.strip() for x in buf if x.strip()).strip()
        out.setdefault(pending_key, val or "-")   # first occurrence wins
        pending_key = None
        buf = []

    for raw in body.split("\n"):
        line = raw.strip()
        if not line:
            continue
        m = _KV_LINE_RE.match(line)
        canonical: str | None = None
        if m:
            head_lc = m.group(1).strip().lower()
            canonical = _HEADER_ALIASES.get(head_lc)
        if canonical:
            flush()
            pending_key = canonical
            rest = m.group(2).strip()
            if rest:
                buf.append(rest)
        elif pending_key is not None:
            # Continuation of prior field (rare — some accommodation strings
            # in the source DB have embedded newlines).
            buf.append(line)
    flush()

    for k in _DAY_KEYS:
        out.setdefault(k, "-")
    return {k: out[k] for k in _DAY_KEYS}


# --------------------------------------------------------------------------- #
# Public API                                                                   #
# --------------------------------------------------------------------------- #
def convert_plan_text(plan_text: str | None, *, day_base: int) -> list[dict]:
    """Convert one raw ``llm_travel_plan`` string into the structured day list.

    ``day_base`` is the desired base for the emitted ``days`` field: pass
    ``1`` to preserve the ``"Day 1:"`` numbering as-is, ``0`` to shift it
    down by one.  Returns ``[]`` when the text has no parseable day block."""
    text = _normalize_text(plan_text)
    if not text:
        return []
    days = _split_days(text)
    if not days:
        return []
    shift = day_base - 1                                          # source is 1-based
    plan: list[dict] = []
    for day_num, body in days:
        row = {"days": day_num + shift}
        row.update(_parse_day_body(body))
        plan.append(row)
    return plan


def convert_file(in_path: Path, out_path: Path, *, day_base: int) -> None:
    """Stream-convert ``in_path`` to ``out_path``.

    ``id`` and ``source_id`` are passed through verbatim from each input
    row -- they are already 1-indexed in the post-migration schema and
    align 1:1 with the HF dataset.  ``day_base`` controls only the
    per-day ``days`` field (default ``1``, matching the ``"Day N:"``
    numbering in the LLM output).

    Legacy files carrying the pre-migration ``idx`` key (and no ``id``)
    are rejected with a clear pointer to
    ``plan-generation/migrate_legacy_cache.py``.
    """
    n_in = n_with_plan = n_empty = 0
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with in_path.open() as fin, out_path.open("w") as fout:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            n_in += 1
            row = json.loads(line)
            if "id" not in row:
                if "idx" in row:
                    raise SystemExit(
                        f"[convert] {in_path} carries the legacy 'idx' "
                        f"key but no 'id'.  Run\n"
                        f"    python3 plan-generation/migrate_legacy_cache.py "
                        f"{in_path}\n"
                        f"once to convert it to the id / source_id schema, "
                        f"then re-run this converter."
                    )
                raise SystemExit(
                    f"[convert] {in_path} row {n_in} has no 'id' field; "
                    f"cannot convert.")
            plan = convert_plan_text(row.get("llm_travel_plan"), day_base=day_base)
            raw_sid = row.get("source_id")
            sid = int(raw_sid) if raw_sid is not None else None
            rec = {
                "id":        int(row["id"]),
                "source_id": sid,
                "query":     row.get("query", ""),
                "plan":      plan,
            }
            fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
            if plan:
                n_with_plan += 1
            else:
                n_empty += 1
    print(f"[convert] {in_path.name}: {n_in} records "
          f"({n_with_plan} with plans, {n_empty} empty) "
          f"[day_base={day_base}] → {out_path}")


# --------------------------------------------------------------------------- #
# CLI                                                                          #
# --------------------------------------------------------------------------- #
def main() -> None:
    ap = argparse.ArgumentParser(
        description="Convert plans_*.jsonl into structured (id / source_id) JSONL.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Example:\n"
            "  python evaluation/convert_plans.py \\\n"
            "         --in plan-generation/plans_openrouter_deepseek_deepseek-v4-flash.jsonl \\\n"
            "         --out evaluation/structured_deepseek.jsonl\n"
            "\n"
            "  # 0-based day numbering (default is 1-based, matching \"Day N:\"):\n"
            "  python evaluation/convert_plans.py --day-base 0 \\\n"
            "         --in ... --out ...\n"
        ),
    )
    ap.add_argument("--in", dest="in_path", type=Path, required=True,
                    help="Input JSONL produced by plan-generation/generate_plans.py.")
    ap.add_argument("--out", dest="out_path", type=Path, required=True,
                    help="Output structured JSONL.")
    ap.add_argument("--day-base", type=int, choices=(0, 1), default=1,
                    help="Base index for the per-day `days` field.  "
                         "1 (default) preserves the \"Day 1:\" numbering "
                         "from the LLM output; 0 shifts each day down by "
                         "one.  Does not affect the row-level `id` / "
                         "`source_id`, which pass through verbatim.")
    args = ap.parse_args()

    convert_file(args.in_path, args.out_path, day_base=args.day_base)


if __name__ == "__main__":
    main()
