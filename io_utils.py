"""Readers for the released evaluation data.

Two things about the released layout differ from the layout the analysis scripts were
written against, and both are handled here rather than at every call site:

  * the model responses live under ``data/responses/<model>/<task>/output.json``
    rather than ``se_results/<model>/<task>/output.json``;
  * responses and diagnosis records ship gzipped, because uncompressed they are
    611 MB against 104 MB compressed.

``find`` matches a pattern in either location and in either form, so code that asks for
``{BASE}/se_results/Chem-R/cap2mol/output.json`` gets the released file. ``open_text``
opens a match whichever form it turned out to be. Neither helper needs the caller to
know which layout it is looking at.

Model directories use the release display tokens (``Chem-R-Faithful``, ``process``,
``SFT``, ``base-a``, …); ``data/raw/README.md`` maps them to the internal training
codenames used while the study was run.
"""
import glob as _glob
import gzip
import os

__all__ = ["find", "open_text", "load_json", "iter_jsonl"]

# se_results/ was the working-repo location; data/responses/ is the released one.
_RELOCATED = (("/se_results/", "/data/responses/"),)


def _variants(pattern):
    pats = [pattern]
    for old, new in _RELOCATED:
        if old in pattern:
            pats.append(pattern.replace(old, new))
    return [p for pat in pats for p in (pat, pat + ".gz")]


def find(pattern):
    """glob(), also matching the relocated and/or gzipped form of the same pattern.

    Returns the first non-empty match set, so an uncompressed file already in place
    wins over a compressed one and the caller sees the layout it expected.
    """
    for p in _variants(pattern):
        hits = sorted(_glob.glob(p))
        if hits:
            return hits
    return []


def open_text(path):
    """Open a data file for reading text, transparently decompressing .gz."""
    if path.endswith(".gz"):
        return gzip.open(path, "rt", encoding="utf-8")
    return open(path, encoding="utf-8")


def load_json(path):
    import json
    with open_text(path) as f:
        return json.load(f)


def iter_jsonl(path):
    """Yield one parsed record per line of a (possibly gzipped) JSONL file."""
    import json
    with open_text(path) as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)
