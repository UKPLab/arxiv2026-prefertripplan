#!/usr/bin/env python3
"""latex.py — cross-model LaTeX tables from ``analyze_performance.py``
plain-text reports.

Consumes one or more ``analyze_*.txt`` files (one per model / split
combination) and emits six paper-ready booktabs LaTeX tables:

    overall.tex          Overall (single bucket)
    by_days.tex          Per-day slice (3 / 5 / 7)
    by_pairing_type.tex  single / independent / overlapping
    by_pairing_subtype.tex   competing / non_competing / --
    by_paradigm.tex      Pref pass_rate per paradigm (8 columns)
    by_trivial.tex       Trivial vs non-trivial preference pass rate
    all_tables.tex       \\input{}-includes all six

Rows across every table are model×split combinations, so multiple models
sit side-by-side.  Nothing about which columns are shown is model-
dependent; the parser derives the schema from the analyze report.

Usage
-----
Auto-labelled (label derives from parent dir + filename)::

    python3 evaluation/latex.py \\
        evaluation/nemotron_test/analyze_nemotron.txt \\
        evaluation/deepseek_test/analyze_deepseek.txt \\
        evaluation/nemotron_test_large/analyze_nemotron.txt \\
        evaluation/deepseek_test_large/analyze_deepseek.txt \\
        --out-dir evaluation/latex_tables/

Explicit label::

    python3 evaluation/latex.py \\
        --input evaluation/nemotron_test/analyze_nemotron.txt:Nemotron/test \\
        --input evaluation/deepseek_test/analyze_deepseek.txt:DeepSeek/test \\
        --out-dir evaluation/latex_tables/

Every emitted ``.tex`` file uses ``booktabs``; add ``\\usepackage{booktabs}``
(and ``\\usepackage{multirow}`` for the sliced tables) to the preamble.
"""
from __future__ import annotations

import argparse
import re
from dataclasses import dataclass, field
from pathlib import Path


# --------------------------------------------------------------------------- #
# Parser                                                                      #
# --------------------------------------------------------------------------- #

_SECTION_RE = re.compile(r"^===\s+(.*?)\s+===\s*$")
_DASH_RE    = re.compile(r"^\s*-+\s*$")


@dataclass
class Report:
    """One parsed analyze_*.txt report."""
    path: Path
    label: str                                        # display label
    sections: dict[str, list[dict[str, str]]] = field(default_factory=dict)

    def section(self, name: str) -> list[dict[str, str]]:
        """Return rows for a section by exact title; empty list if absent."""
        return self.sections.get(name, [])


def _parse_report(path: Path, label: str) -> Report:
    """Walk the analyze text file section-by-section.

    Each section has this shape::

        === <title> ===
          <header line — whitespace-separated column names>
          --- separator ---
          <row1: bucket-name value1 value2 ...>
          <row2: ...>
          <blank line ends section>

    Some cross-tab sections have a second dash line before a TOTAL row;
    that TOTAL is captured as its own bucket row.
    """
    lines = path.read_text().splitlines()
    sections: dict[str, list[dict[str, str]]] = {}

    i, n = 0, len(lines)
    while i < n:
        m = _SECTION_RE.match(lines[i])
        if not m:
            i += 1; continue
        title = m.group(1)
        i += 1
        # Header line (first non-empty line inside the section).
        while i < n and not lines[i].strip():
            i += 1
        if i >= n:
            break
        header_line = lines[i].strip()
        i += 1
        # Skip the dashes.
        while i < n and _DASH_RE.match(lines[i]):
            i += 1
        # Header columns.
        header_tokens = header_line.split()
        # Body: read until blank line or next section header.
        rows: list[dict[str, str]] = []
        while i < n:
            line = lines[i]
            if not line.strip():
                break
            if _SECTION_RE.match(line):
                break
            if _DASH_RE.match(line):
                i += 1
                continue
            toks = line.split()
            if len(toks) < len(header_tokens):
                i += 1
                continue
            # First token is the bucket label; remaining align with the
            # rest of the header.  For the cross-tab tables the bucket
            # label can be "TOTAL" (footer) or a multi-word bucket
            # (never observed on our data, but keep tolerant).
            rows.append(dict(zip(header_tokens, toks[: len(header_tokens)])))
            i += 1
        sections[title] = rows

    return Report(path=path, label=label, sections=sections)


def _auto_label(path: Path) -> str:
    """Derive a display label from ``path``.  Uses the parent directory
    name (e.g. ``nemotron_test_large``) so the label carries both the
    model AND the split.  Falls back to the filename stem when the file
    isn't inside a per-model directory."""
    parent = path.parent.name
    if parent and parent != ".":
        return parent
    return path.stem


# --------------------------------------------------------------------------- #
# Helpers                                                                     #
# --------------------------------------------------------------------------- #

def _strip_pct(v: str) -> str:
    """Strip a trailing ``%`` and return the numeric string.  Missing /
    dash tokens come through unchanged."""
    if not v or v == "-":
        return "--"
    return v[:-1] if v.endswith("%") else v


def _extract_pct(v: str) -> str:
    """From ``"90.6% (85)"`` return ``"90.6"``; from plain ``"90.6%"``
    return ``"90.6"``.  Dashes come back as ``--``."""
    if not v or v == "-":
        return "--"
    # Cross-tab cells embed a "( n )" alongside the pass_rate.
    m = re.match(r"^\s*([0-9]+(?:\.[0-9]+)?)\s*%", v)
    if m:
        return m.group(1)
    if v.endswith("%"):
        return v[:-1]
    return v


def _tex_label(s: str) -> str:
    """Escape a label string for LaTeX (underscores, percent signs)."""
    return s.replace("\\", "\\\\").replace("_", r"\_").replace("%", r"\%") \
             .replace("&", r"\&").replace("#", r"\#")


def _num(v: str, default: str = "--") -> str:
    """Format a value for a LaTeX cell.  Strips percent signs; keeps
    integers integer-formatted; passes dashes through as ``--``."""
    if v is None or v == "" or v == "-" or v == "--":
        return default
    if v.endswith("%"):
        v = v[:-1]
    return v


# --------------------------------------------------------------------------- #
# Table renderers                                                             #
# --------------------------------------------------------------------------- #

_METRIC_COLS       = ("N", "deliv", "cs_μ", "cs_M", "hd_μ", "hd_M",
                        "pf_μ", "pf_M", "final", "final+p", "Δpref")
_METRIC_LABELS_TEX = ("N", "Deliv", r"CS\_$\mu$", r"CS\_M",
                        r"Hd\_$\mu$", r"Hd\_M", r"Pf\_$\mu$", r"Pf\_M",
                        "Final", "Final+p", r"$\Delta$pref")
_METRIC_LABELS_MD  = ("N", "Deliv", "CS μ", "CS M", "Hd μ", "Hd M",
                        "Pf μ", "Pf M", "Final", "Final+p", "Δpref")
_METRIC_LABELS = _METRIC_LABELS_TEX   # backwards-compat alias for the LaTeX renderers

# Preference-only subset -- used by the pairing_type / pairing_subtype
# tables where the LC / commonsense / delivery columns aren't the story
# and would just consume horizontal space.  Final (pref-less) is dropped
# too since Δpref already conveys the marginal cost of preferences on
# top of the pref-inclusive final rate.
_PREF_METRIC_COLS       = ("N", "pf_μ", "pf_M", "final+p", "Δpref")
_PREF_METRIC_LABELS_TEX = ("N", r"Pf\_$\mu$", r"Pf\_M",
                             "Final+p", r"$\Delta$pref")
_PREF_METRIC_LABELS_MD  = ("N", "Pf μ", "Pf M", "Final+p", "Δpref")


def _render_overall(reports: list[Report]) -> str:
    col_spec = "l" + "r" * len(_METRIC_COLS)
    lines = [
        r"% Overall (single bucket) — per-model summary.",
        r"\begin{table}[t]",
        r"  \centering",
        r"  \small",
        r"  \caption{Overall performance across models and splits.  Rates are shown in \%.  "
        r"Final+p = Final Pass Rate including preferences; $\Delta$pref = Final $-$ Final+p.}",
        r"  \label{tab:overall}",
        r"  \begin{tabular}{" + col_spec + r"}",
        r"    \toprule",
        r"    Model / Split & " + " & ".join(_METRIC_LABELS) + r" \\",
        r"    \midrule",
    ]
    for rp in reports:
        rows = rp.section("Overall (single bucket)")
        if not rows:
            continue
        r0 = rows[0]
        vals = [_num(r0.get(c, "--")) for c in _METRIC_COLS]
        lines.append(f"    {_tex_label(rp.label)} & " + " & ".join(vals) + r" \\")
    lines += [
        r"    \bottomrule",
        r"  \end{tabular}",
        r"\end{table}",
        "",
    ]
    return "\n".join(lines)


def _render_by_axis(reports: list[Report], section: str, axis_label: str,
                     axis_col_name: str, table_label: str,
                     caption: str,
                     metric_cols: tuple = _METRIC_COLS,
                     metric_labels: tuple = _METRIC_LABELS_TEX) -> str:
    """Sliced constraint table: one row per (model, bucket) with metrics
    as columns.  Uses \\multirow to group buckets under each model.

    axis_col_name is the token that names the bucket column in the header
    (which is always the FIRST token — ``bucket`` — since the analyze
    report uses that literal string).  Kept as a parameter for future
    flexibility.

    ``metric_cols`` / ``metric_labels`` let a caller narrow to a subset
    (e.g. preference-only columns for pairing_type / pairing_subtype
    tables where the LC / CS / delivery columns aren't the story)."""
    col_spec = "ll" + "r" * len(metric_cols)
    lines = [
        f"% {caption}",
        r"\begin{table}[t]",
        r"  \centering",
        r"  \footnotesize",
        f"  \\caption{{{caption}  Rates in \\%.}}",
        f"  \\label{{tab:{table_label}}}",
        r"  \begin{tabular}{" + col_spec + r"}",
        r"    \toprule",
        rf"    Model / Split & {axis_label} & " + " & ".join(metric_labels) + r" \\",
        r"    \midrule",
    ]
    for m_idx, rp in enumerate(reports):
        rows = rp.section(section)
        if not rows:
            continue
        n_rows = len(rows)
        # First row carries the multirow model label; subsequent rows leave it blank.
        for j, r in enumerate(rows):
            bucket = r.get("bucket", "--")
            vals   = [_num(r.get(c, "--")) for c in metric_cols]
            if j == 0:
                lines.append(
                    f"    \\multirow{{{n_rows}}}{{*}}{{{_tex_label(rp.label)}}} "
                    f"& {_tex_label(bucket)} & " + " & ".join(vals) + r" \\")
            else:
                lines.append(
                    f"                                            "
                    f"& {_tex_label(bucket)} & " + " & ".join(vals) + r" \\")
        if m_idx < len(reports) - 1:
            lines.append(r"    \midrule")
    lines += [
        r"    \bottomrule",
        r"  \end{tabular}",
        r"\end{table}",
        "",
    ]
    return "\n".join(lines)


_PARADIGMS = ("AtomicPreference", "CompositePreference",
                "ConditionalPreference", "LexicographicPreference",
                "NumericPreference", "ScopedPreference",
                "CompensatoryPreference", "TemporalPreference")
_PARADIGM_LABELS = ("Atomic", "Composite", "Conditional", "Lex",
                    "Numeric", "Scoped", "Comp", "Temporal")


def _render_by_paradigm(reports: list[Report]) -> str:
    """Preference pass\\% per paradigm — one row per model, one column
    per paradigm, plus a total-N column for sample-size context."""
    col_spec = "l" + "r" * (len(_PARADIGMS) + 1)
    lines = [
        r"% Preference pass rate per paradigm.",
        r"\begin{table}[t]",
        r"  \centering",
        r"  \small",
        r"  \caption{Per-paradigm preference pass rate (\%) with total preference count.  "
        r"Higher is better.}",
        r"  \label{tab:by_paradigm}",
        r"  \begin{tabular}{" + col_spec + r"}",
        r"    \toprule",
        r"    Model / Split & Total N & " + " & ".join(_PARADIGM_LABELS) + r" \\",
        r"    \midrule",
    ]
    for rp in reports:
        rows = rp.section("Preferences: by paradigm")
        if not rows:
            continue
        by_p = {r["bucket"]: r for r in rows if "bucket" in r}
        total_n = sum(int(r.get("n", 0) or 0) for r in rows)
        cells = []
        for p in _PARADIGMS:
            r = by_p.get(p)
            cells.append(_extract_pct(r.get("pass%", "--")) if r else "--")
        lines.append(f"    {_tex_label(rp.label)} & {total_n} & " +
                     " & ".join(cells) + r" \\")
    lines += [
        r"    \bottomrule",
        r"  \end{tabular}",
        r"\end{table}",
        "",
    ]
    return "\n".join(lines)


def _render_by_trivial(reports: list[Report]) -> str:
    """Trivial vs non-trivial preference pass rate."""
    col_spec = "lrrrr"
    lines = [
        r"% Trivial vs non-trivial preference pass rate.",
        r"\begin{table}[t]",
        r"  \centering",
        r"  \small",
        r"  \caption{Trivial-vs-non-trivial preference pass rate (\%). "
        r"A trivial preference is one whose evaluator returns a vacuous "
        r"``true'' (e.g. Atomic with no entities, Scoped with no matching "
        r"days, Conditional with false condition + no else branch).  A "
        r"large trivial pass share alongside a non-trivial gap suggests "
        r"the planner is exploiting trivial paths.}",
        r"  \label{tab:by_trivial}",
        r"  \begin{tabular}{" + col_spec + r"}",
        r"    \toprule",
        r"    Model / Split & Trivial N & Trivial \% & "
        r"Trivial pass\% & Non-trivial pass\% \\",
        r"    \midrule",
    ]
    for rp in reports:
        rows = rp.section("Preferences: by trivial-vs-non-trivial")
        if not rows:
            continue
        by_b = {r["bucket"]: r for r in rows if "bucket" in r}
        t = by_b.get("trivial", {})
        nt = by_b.get("non-trivial", {})
        try:
            t_n = int(t.get("n", 0) or 0)
        except ValueError:
            t_n = 0
        try:
            nt_n = int(nt.get("n", 0) or 0)
        except ValueError:
            nt_n = 0
        total = t_n + nt_n
        share = (100.0 * t_n / total) if total else 0.0
        lines.append(
            f"    {_tex_label(rp.label)} & {t_n} & {share:.1f} & "
            f"{_extract_pct(t.get('pass%', '--'))} & "
            f"{_extract_pct(nt.get('pass%', '--'))} \\\\")
    lines += [
        r"    \bottomrule",
        r"  \end{tabular}",
        r"\end{table}",
        "",
    ]
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# Markdown renderers                                                          #
# --------------------------------------------------------------------------- #
# Same six tables, GitHub-flavored markdown.  Multirow model grouping is
# emulated by repeating the model label in each bucket row -- GFM has
# no rowspan, and repeated cells render most cleanly across viewers.

def _md_header(headers: list[str], aligns: list[str] | None = None) -> list[str]:
    """Return the two header lines of a markdown table (title row + the
    ``|:---|:---:|---:|`` separator).  ``aligns[i]`` is ``"l"``, ``"c"``,
    or ``"r"``; defaults to left-aligned for the first column and right-
    aligned for the rest."""
    if aligns is None:
        aligns = ["l"] + ["r"] * (len(headers) - 1)
    sep = []
    for a in aligns:
        if a == "l":   sep.append(":---")
        elif a == "c": sep.append(":---:")
        else:          sep.append("---:")
    return [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(sep)     + " |",
    ]


def _md_row(cells: list[str]) -> str:
    return "| " + " | ".join(cells) + " |"


def _md_overall(reports: list[Report]) -> str:
    lines = [
        "## Overall (single bucket)",
        "",
        "Rates are shown in %.  `Final+p` = Final Pass Rate including "
        "preferences; `Δpref` = Final − Final+p.",
        "",
    ]
    lines.extend(_md_header(["Model / Split", *_METRIC_LABELS_MD]))
    for rp in reports:
        rows = rp.section("Overall (single bucket)")
        if not rows:
            continue
        r0 = rows[0]
        vals = [_num(r0.get(c, "--")) for c in _METRIC_COLS]
        lines.append(_md_row([rp.label, *vals]))
    lines.append("")
    return "\n".join(lines)


def _md_by_axis(reports: list[Report], section: str, axis_label: str,
                 heading: str, caption: str,
                 metric_cols: tuple = _METRIC_COLS,
                 metric_labels_md: tuple = _METRIC_LABELS_MD) -> str:
    """Sliced tables: model label is repeated on every bucket row (GFM
    has no rowspan).  ``metric_cols`` / ``metric_labels_md`` let a
    caller narrow to a subset (e.g. preference-only columns)."""
    lines = [
        f"## {heading}",
        "",
        caption,
        "",
    ]
    lines.extend(_md_header(["Model / Split", axis_label, *metric_labels_md]))
    for rp in reports:
        rows = rp.section(section)
        if not rows:
            continue
        for r in rows:
            bucket = r.get("bucket", "--")
            vals   = [_num(r.get(c, "--")) for c in metric_cols]
            lines.append(_md_row([rp.label, bucket, *vals]))
    lines.append("")
    return "\n".join(lines)


def _md_by_paradigm(reports: list[Report]) -> str:
    lines = [
        "## Preference pass rate per paradigm",
        "",
        "Values are pass rate (%).  `Total N` = total preferences across "
        "the split.",
        "",
    ]
    lines.extend(_md_header(["Model / Split", "Total N", *_PARADIGM_LABELS]))
    for rp in reports:
        rows = rp.section("Preferences: by paradigm")
        if not rows:
            continue
        by_p = {r["bucket"]: r for r in rows if "bucket" in r}
        total_n = sum(int(r.get("n", 0) or 0) for r in rows)
        cells = [rp.label, str(total_n)]
        for p in _PARADIGMS:
            r = by_p.get(p)
            cells.append(_extract_pct(r.get("pass%", "--")) if r else "--")
        lines.append(_md_row(cells))
    lines.append("")
    return "\n".join(lines)


def _md_by_trivial(reports: list[Report]) -> str:
    lines = [
        "## Trivial vs non-trivial preferences",
        "",
        "A trivial preference is one whose evaluator returns a vacuous "
        "\"true\" (e.g. Atomic with no entities, Scoped with no matching "
        "days, Conditional with false condition and no `else` branch).  "
        "A large trivial share alongside a non-trivial pass gap suggests "
        "the planner is exploiting trivial paths.",
        "",
    ]
    lines.extend(_md_header(["Model / Split", "Trivial N", "Trivial %",
                              "Trivial pass%", "Non-trivial pass%"]))
    for rp in reports:
        rows = rp.section("Preferences: by trivial-vs-non-trivial")
        if not rows:
            continue
        by_b = {r["bucket"]: r for r in rows if "bucket" in r}
        t  = by_b.get("trivial", {})
        nt = by_b.get("non-trivial", {})
        try:
            t_n  = int(t.get("n", 0) or 0)
        except ValueError: t_n = 0
        try:
            nt_n = int(nt.get("n", 0) or 0)
        except ValueError: nt_n = 0
        total = t_n + nt_n
        share = (100.0 * t_n / total) if total else 0.0
        lines.append(_md_row([
            rp.label, str(t_n), f"{share:.1f}",
            _extract_pct(t.get("pass%", "--")),
            _extract_pct(nt.get("pass%", "--")),
        ]))
    lines.append("")
    return "\n".join(lines)


def _md_all_bundle(sections: list[tuple[str, str]]) -> str:
    """Concatenate every markdown table into a single browsable
    ``all_tables.md``.  Suitable for pasting into a GitHub PR or a
    single-page render."""
    header = [
        "# Cross-model performance tables",
        "",
        "Generated by `evaluation/latex.py --md-out-dir`.",
        "",
    ]
    body = []
    for _, tex in sections:
        body.append(tex)
    return "\n".join(header) + "\n".join(body)


# --------------------------------------------------------------------------- #
# LaTeX all-tables wrapper                                                    #
# --------------------------------------------------------------------------- #

def _render_all_include(names: list[str]) -> str:
    """Emit a helper file that ``\\input``s every generated table so a
    single ``\\input{all_tables.tex}`` pulls the whole set into the
    paper."""
    lines = [
        r"% Auto-generated include-all wrapper.",
        r"% Add to your paper preamble:",
        r"%   \usepackage{booktabs}",
        r"%   \usepackage{multirow}",
    ]
    for n in names:
        lines.append(f"\\input{{{n}}}")
    lines.append("")
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# Main                                                                        #
# --------------------------------------------------------------------------- #

def _parse_inputs(positional: list[str],
                  labelled: list[str]) -> list[tuple[Path, str]]:
    """Merge positional ``PATH`` entries and ``--input PATH:LABEL`` entries
    into a single ordered list of (path, label) tuples.  Positional
    entries auto-label from the parent directory name."""
    out: list[tuple[Path, str]] = []
    for spec in labelled or []:
        if ":" in spec:
            p, lbl = spec.rsplit(":", 1)
            out.append((Path(p), lbl))
        else:
            out.append((Path(spec), _auto_label(Path(spec))))
    for p in positional or []:
        out.append((Path(p), _auto_label(Path(p))))
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0],
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("positional", nargs="*",
                    help="Analyze report paths (auto-labelled from parent dir).")
    ap.add_argument("--input", action="append", metavar="PATH[:LABEL]",
                    help="Explicit input with optional label; may be repeated.")
    ap.add_argument("--out-dir", type=Path, required=True,
                    help="Directory where the per-table .tex files land.")
    ap.add_argument("--md-out-dir", type=Path, default=None,
                    help="If set, also emit GitHub-flavored markdown "
                         "versions of the six tables into this directory, "
                         "plus a bundled all_tables.md.")
    args = ap.parse_args()

    inputs = _parse_inputs(args.positional, args.input)
    if not inputs:
        ap.error("At least one input file is required.")

    reports: list[Report] = []
    for path, label in inputs:
        if not path.exists():
            print(f"[warn] {path} does not exist; skipping")
            continue
        rp = _parse_report(path, label)
        print(f"[in]  {path}  →  label={label!r}  "
              f"sections={len(rp.sections)}")
        reports.append(rp)

    if not reports:
        ap.error("No inputs were readable.")

    args.out_dir.mkdir(parents=True, exist_ok=True)

    outputs = [
        ("overall.tex",             _render_overall(reports)),
        ("by_days.tex",             _render_by_axis(reports,
                                        section="By days",
                                        axis_label="Days",
                                        axis_col_name="bucket",
                                        table_label="by_days",
                                        caption=r"Performance sliced by trip duration.")),
        ("by_pairing_type.tex",     _render_by_axis(reports,
                                        section="By pairing_type",
                                        axis_label="Pairing type",
                                        axis_col_name="bucket",
                                        table_label="by_pairing_type",
                                        caption=r"Preference performance by pairing type "
                                                 r"(single / independent / overlapping).  "
                                                 r"Columns are preference-only; delivery / "
                                                 r"commonsense / hard columns omitted.",
                                        metric_cols=_PREF_METRIC_COLS,
                                        metric_labels=_PREF_METRIC_LABELS_TEX)),
        ("by_pairing_subtype.tex",  _render_by_axis(reports,
                                        section="By pairing_subtype",
                                        axis_label="Pairing subtype",
                                        axis_col_name="bucket",
                                        table_label="by_pairing_subtype",
                                        caption=r"Preference performance by pairing subtype "
                                                 r"(competing / non\_competing / --).  "
                                                 r"Columns are preference-only; delivery / "
                                                 r"commonsense / hard columns omitted.",
                                        metric_cols=_PREF_METRIC_COLS,
                                        metric_labels=_PREF_METRIC_LABELS_TEX)),
        ("by_paradigm.tex",         _render_by_paradigm(reports)),
        ("by_trivial.tex",          _render_by_trivial(reports)),
    ]
    for name, tex in outputs:
        (args.out_dir / name).write_text(tex)
        print(f"[out] {args.out_dir / name}")
    (args.out_dir / "all_tables.tex").write_text(
        _render_all_include([n for n, _ in outputs]))
    print(f"[out] {args.out_dir / 'all_tables.tex'}")

    # Markdown output (optional).
    if args.md_out_dir is not None:
        args.md_out_dir.mkdir(parents=True, exist_ok=True)
        md_outputs = [
            ("overall.md",            _md_overall(reports)),
            ("by_days.md",            _md_by_axis(reports,
                                          section="By days",
                                          axis_label="Days",
                                          heading="By trip duration",
                                          caption=r"Performance sliced by trip duration.  Rates in %.")),
            ("by_pairing_type.md",    _md_by_axis(reports,
                                          section="By pairing_type",
                                          axis_label="Pairing type",
                                          heading="By preference-pairing type",
                                          caption=r"single / independent / overlapping.  "
                                                   r"Preference-only columns (delivery / "
                                                   r"commonsense / hard omitted).  Rates in %.",
                                          metric_cols=_PREF_METRIC_COLS,
                                          metric_labels_md=_PREF_METRIC_LABELS_MD)),
            ("by_pairing_subtype.md", _md_by_axis(reports,
                                          section="By pairing_subtype",
                                          axis_label="Pairing subtype",
                                          heading="By preference-pairing subtype",
                                          caption=r"competing / non_competing / --.  "
                                                   r"Preference-only columns (delivery / "
                                                   r"commonsense / hard omitted).  Rates in %.",
                                          metric_cols=_PREF_METRIC_COLS,
                                          metric_labels_md=_PREF_METRIC_LABELS_MD)),
            ("by_paradigm.md",        _md_by_paradigm(reports)),
            ("by_trivial.md",         _md_by_trivial(reports)),
        ]
        for name, md in md_outputs:
            (args.md_out_dir / name).write_text(md)
            print(f"[out] {args.md_out_dir / name}")
        (args.md_out_dir / "all_tables.md").write_text(
            _md_all_bundle(md_outputs))
        print(f"[out] {args.md_out_dir / 'all_tables.md'}")


if __name__ == "__main__":
    main()
