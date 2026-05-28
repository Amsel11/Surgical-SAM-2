"""Closed-vocabulary loader for the OCR slot-timeline scripts.

Pure JSON parsing, no heavy deps — both ocr_slots_timeline.py (Qwen) and
ocr_slots_timeline_paddle.py import `load_vocab` from here so neither pulls in
a model/runtime dependency just to read the vocab file.

Vocab file: configs/cardiac_whip_vocabulary.json — either the structured form
(`canonical_instruments: [{canonical, ui_variants}]` + `mode_indicators_to_ignore`)
or a legacy flat `all_queries_flat` list.
"""
from __future__ import annotations

import json
from pathlib import Path


def load_vocab(queries_file: Path):
    """Return (canonical_names, ui_to_canonical_map, mode_indicators)."""
    data = json.loads(Path(queries_file).read_text())
    canonical_names: list[str] = []
    ui_to_canonical: dict[str, str] = {}
    if "canonical_instruments" in data:
        for entry in data["canonical_instruments"]:
            c = entry["canonical"].strip().lower()
            canonical_names.append(c)
            for v in entry.get("ui_variants", []):
                ui_to_canonical[v.strip().lower()] = c
            ui_to_canonical[c] = c  # canonical itself always matches
    else:
        canonical_names = [q.strip().lower() for q in data["all_queries_flat"]]
        for c in canonical_names:
            ui_to_canonical[c] = c
    modes = [m.lower() for m in data.get("mode_indicators_to_ignore", [])]
    return canonical_names, ui_to_canonical, modes


__all__ = ["load_vocab"]
