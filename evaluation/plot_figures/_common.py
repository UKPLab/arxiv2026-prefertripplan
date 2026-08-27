"""_common.py — shared styling and helpers for the figure scripts in
this directory.

Both ``plot_paradigm_details.py`` (per-model paradigm figures) and
``plot_pairing_slope.py`` (cross-model slope chart) import from here so
the two produce visually consistent output and neither has to carry a
copy of the save / mean plumbing.

Importing this module applies the shared ``rcParams``; each script is
free to override individual keys afterwards.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")            # non-interactive backend
import matplotlib.pyplot as plt


# --------------------------------------------------------------------------- #
# Shared styling                                                              #
# --------------------------------------------------------------------------- #

plt.rcParams.update({
    "figure.figsize":  (7.0, 4.5),
    "figure.dpi":      120,
    "font.size":       10,
    "axes.grid":       True,
    "grid.alpha":      0.25,
    "axes.spines.top": False,
    "axes.spines.right": False,
    # Default hatch strokes are hairline and vanish when a figure is
    # scaled down for a two-column layout.
    "hatch.linewidth": 1.5,
})


# --------------------------------------------------------------------------- #
# Helpers                                                                     #
# --------------------------------------------------------------------------- #

def save(fig, out_path: Path, fmt: str) -> None:
    """Write ``fig`` to ``out_path`` with the given extension, creating
    the parent directory if needed, then close the figure."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path.with_suffix(f".{fmt}"), bbox_inches="tight")
    plt.close(fig)


def mean(v: list[float]) -> float:
    """Arithmetic mean; ``0.0`` on an empty sequence."""
    return (sum(v) / len(v)) if v else 0.0


# --------------------------------------------------------------------------- #
# Cross-model figure vocabulary                                               #
# --------------------------------------------------------------------------- #
# Model identity (display name + colour), split encoding, the categorical
# axis definitions, and the analyze-report cell reader.  Everything the
# cross-model figures need to stay mutually consistent lives here, so a
# model added or an axis reordered lands in every figure at once.

# Display names for the legend.  Anything absent falls back to the raw
# token derived from the run-directory name.
_MODEL_DISPLAY_NAME = {
    "deepseek":        "deepseek-v4-flash",
    "nemotron":        "nemotron-3-ultra",
    "gemma-4-26b-a4b": "gemma-4-26b",
    "qwen3.8-27b":     "qwen3.8-27b",
}


def display_model(model: str) -> str:
    return _MODEL_DISPLAY_NAME.get(model, model)


# Fixed model -> colour so a model keeps its colour across every figure
# and regardless of --report ordering.  Unlisted models draw from the
# palette without reusing a colour a listed model already owns.
_MODEL_COLOR = {
    "deepseek":        "#4c78a8",   # blue
    "nemotron":        "#e45756",   # red
    "gemma-4-26b-a4b": "#54a24b",   # green
    "gpt-5.6-terra":   "#b279a2",   # purple
    "qwen3.8-27b":     "#f58518",   # orange
}
_MODEL_PALETTE = ["#4c78a8", "#e45756", "#54a24b", "#b279a2",
                  "#f58518", "#72b7b2", "#eeca3b", "#9d755d"]

# Fixed left-to-right / legend order for the models.  Anything not listed
# keeps its first-seen position after these.
# qwen3.8-27b sits with the mid-pack models (preference micro 86.2% on
# test_large, effectively tied with deepseek) rather than at the end, so
# gemma-4-26b stays the visual outlier in last position.
MODEL_ORDER = ("gpt-5.6-terra", "nemotron", "deepseek", "qwen3.8-27b",
               "gemma-4-26b-a4b")


def order_models(models: list[str]) -> list[str]:
    """Sort ``models`` into ``MODEL_ORDER``, appending unlisted models in
    their first-seen order so a new run still plots."""
    known = [m for m in MODEL_ORDER if m in models]
    rest  = [m for m in models if m not in MODEL_ORDER]
    return known + rest


def model_colours(models: list[str]) -> dict[str, str]:
    """Return ``{model: colour}``, honouring ``_MODEL_COLOR`` first."""
    taken = {c for m, c in _MODEL_COLOR.items() if m in models}
    spare = [c for c in _MODEL_PALETTE if c not in taken]
    out, k = {}, 0
    for m in models:
        if m in _MODEL_COLOR:
            out[m] = _MODEL_COLOR[m]
        else:
            out[m] = spare[k % len(spare)] if spare else _MODEL_PALETTE[0]
            k += 1
    return out


# Split encoding.  Colour is spent on the model, so the split is carried
# by fill weight AND hatch together -- either alone is easy to lose, and
# the pair survives greyscale printing.
#
# ``test_large`` is the larger, more reliable split, so it reads as the
# primary series: a solid, fully-opaque fill with no hatch.  ``test`` is
# the secondary one: a translucent fill behind a crisp outline and hatch.
#
# ``face_alpha`` is baked into the facecolor as RGBA rather than passed
# as the patch-wide ``alpha``, because ``alpha`` fades the edge and hatch
# too -- which is exactly what made the earlier hatching invisible.  This
# way the fill is translucent while the border and stripes stay at full
# strength.
SPLIT_STYLE = {
    "test":       {"face_alpha": 0.22, "hatch": "///", "lw": 1.1},
    "test_large": {"face_alpha": 1.00, "hatch": None,  "lw": 0.0},
}
SPLIT_ORDER = ("test", "test_large")


def split_model_and_split(dir_name: str) -> tuple[str, str]:
    """Derive ``(model, split)`` from a run-directory name such as
    ``nemotron_test_large`` -> ``("nemotron", "test_large")``.  The longer
    suffix is stripped first so ``_test_large`` isn't read as ``_test``."""
    for suffix in ("_test_large", "_test"):
        if dir_name.endswith(suffix):
            return dir_name[: -len(suffix)], suffix.lstrip("_")
    return dir_name, "test"


def parse_report_spec(spec: str) -> tuple[Path, str, str]:
    """Parse a ``--report`` argument into ``(path, model, split)``.

    Accepted forms::

        PATH                 -> model / split from the parent directory
                                name (e.g. ``nemotron_test_large``)
        PATH:MODEL:SPLIT     -> explicit override
    """
    bits = spec.split(":")
    if len(bits) >= 3:
        return Path(":".join(bits[:-2])), bits[-2], bits[-1]
    path = Path(spec)
    model, split = split_model_and_split(path.parent.name)
    return path, model, split


def metric_from_report(report, section: str, bucket: str,
                       column: str) -> float | None:
    """Pull one numeric cell out of a parsed analyze report.  Percent
    signs are stripped; ``None`` when the row or column is absent."""
    for row in report.section(section):
        if row.get("bucket") != bucket:
            continue
        raw = row.get(column)
        if raw is None:
            return None
        raw = raw[:-1] if raw.endswith("%") else raw
        try:
            return float(raw)
        except ValueError:
            return None
    return None


# --------------------------------------------------------------------------- #
# Categorical axis definitions                                                #
# --------------------------------------------------------------------------- #
# Each entry is ``(x_label, report_section, bucket)``.  Order and labels
# carried over verbatim from the earlier per-axis scripts.

_PAIRING_SECTION_T = "By pairing_type"
_PAIRING_SECTION_S = "By pairing_subtype"
PAIR_AXIS = [
    ("single",                     _PAIRING_SECTION_T, "single"),
    ("independent",                _PAIRING_SECTION_T, "independent"),
    ("overlapping\nnon-competing", _PAIRING_SECTION_S, "non_competing"),
    ("overlapping\ncompeting",     _PAIRING_SECTION_S, "competing"),
]

_PARADIGM_SECTION = "Preferences: by paradigm"
PARADIGM_AXIS = [
    ("atom.",   _PARADIGM_SECTION, "AtomicPreference"),
    ("compos.", _PARADIGM_SECTION, "CompositePreference"),
    ("cond.",   _PARADIGM_SECTION, "ConditionalPreference"),
    ("lexico.", _PARADIGM_SECTION, "LexicographicPreference"),
    ("compen.", _PARADIGM_SECTION, "CompensatoryPreference"),
    ("scoped",  _PARADIGM_SECTION, "ScopedPreference"),
    ("numeric", _PARADIGM_SECTION, "NumericPreference"),
    ("tempo.",  _PARADIGM_SECTION, "TemporalPreference"),
]

_SUBPARADIGM_SECTION = "Preferences: by (paradigm, sub_paradigm)"
TEMPORAL_AXIS = [
    ("sometime",         _SUBPARADIGM_SECTION, "TemporalPreference[sometime]"),
    ("always",           _SUBPARADIGM_SECTION, "TemporalPreference[always]"),
    ("atmost\nonce",     _SUBPARADIGM_SECTION, "TemporalPreference[atmost_once]"),
    ("within",           _SUBPARADIGM_SECTION, "TemporalPreference[within]"),
    ("sometime\nbefore", _SUBPARADIGM_SECTION, "TemporalPreference[sometime_before]"),
    ("sometime\nafter",  _SUBPARADIGM_SECTION, "TemporalPreference[sometime_after]"),
    ("always\nwithin",   _SUBPARADIGM_SECTION, "TemporalPreference[always_within]"),
    ("hold\nduring",     _SUBPARADIGM_SECTION, "TemporalPreference[hold_during]"),
    ("hold\nafter",      _SUBPARADIGM_SECTION, "TemporalPreference[hold_after]"),
]

# chart name -> (axis spec, default metric column, noun for the title)
CHARTS = {
    "pairing":  (PAIR_AXIS,     "pf_M",  "pairing structures"),
    "paradigm": (PARADIGM_AXIS, "pass%", "preference paradigms"),
    "temporal": (TEMPORAL_AXIS, "pass%", "temporal sub-preferences"),
}
