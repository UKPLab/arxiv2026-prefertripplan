"""Repo paths, sys.path shim, and dataset access.

The agentic package is self-contained: it imports nothing from
``data-generation/`` or ``preferences.py``, and reads no files from them
either. The benchmark comes from the published dataset on the Hub
(``UKPLab/PreferTripPlan``), which is the same source the rest of the pipeline
uses (``generate_plans.py --dataset``, ``eval.py --dataset``), so the agentic
track and the existing results are scored against identical rows.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DB_DIR = ROOT / "database"

DATASET = "UKPLab/PreferTripPlan"
SPLITS = ("test", "test_large")

# Two paths only.
#   plan-generation/     -- ``llm_backends``, the shared provider clients and
#                           retry ladder, used by both the direct planner and
#                           this package so models are driven identically.
#   travelplanner-ports/ -- the ported tool APIs, used by the diagnostics to
#                           check our tools against what the grader reads.
# ``data-generation/`` and ``preferences.py`` are deliberately absent;
# ``evaluation/`` is invoked as a CLI, never imported.
for _d in ("plan-generation", "travelplanner-ports"):
    _p = str(ROOT / _d)
    if _p not in sys.path:
        sys.path.insert(0, _p)


def load_split(split: str, dataset: str = DATASET) -> list[dict]:
    """Return one split as a list of plain row dicts.

    ``dataset`` is anything ``datasets.load_dataset`` accepts: the Hub id
    (default) or a path to a local HF-compatible directory such as
    ``data-generation/HF-data``, which ships a README declaring the same
    splits. Verified: the local directory and the Hub return byte-identical
    rows, so no separate local-file path is needed.
    """
    if split not in SPLITS:
        raise SystemExit(f"unknown split {split!r}; expected one of {list(SPLITS)}")
    from datasets import load_dataset
    return [_normalise(dict(r)) for r in load_dataset(dataset, split=split)]


def _normalise(row: dict) -> dict:
    """Undo a loader artefact in the ``date`` column.

    Any load through ``datasets`` returns ``date`` as
    ``"2025-11-02 00:00:00"``, Hub or local directory alike, while the source
    JSONL holds ``"2025-11-02"``. Arrow infers ``timestamp[s]`` from the ISO
    strings, the dataset README declares the column as ``string``, and the
    resulting cast stringifies the datetime. ``date.fromisoformat`` rejects the
    result, so strip the time component here.

    Verified on ``test``: with this applied, every field of every row matches
    the source JSONL exactly. See DESIGN.md fact 6.
    """
    d = row.get("date")
    if isinstance(d, list):
        row["date"] = [str(x).split(" ")[0] for x in d]
    return row


def add_split_args(parser) -> None:
    """Shared ``--dataset / --split`` flags for the CLIs."""
    parser.add_argument("--dataset", default=DATASET,
                        help=f"HF dataset id or local HF-compatible dir "
                             f"(default: {DATASET})")
    parser.add_argument("--split", default="test", choices=list(SPLITS))


def split_from_args(a) -> list[dict]:
    return load_split(a.split, dataset=a.dataset)
