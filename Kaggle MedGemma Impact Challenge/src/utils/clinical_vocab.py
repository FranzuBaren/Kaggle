"""
Clinical Vocabulary — Synonym-Aware Entity Extraction.

A 20-entry clinical dictionary for normalizing radiological findings
across synonyms, abbreviations, and variant phrasing. Ensures that
"pleural effusion", "fluid in pleural space", and "effusion" all map
to the same canonical entity.
"""

from __future__ import annotations

import re
from collections import defaultdict


# ── Canonical clinical entities with synonym groups ──
SYNONYM_MAP: dict[str, list[str]] = {
    "cardiomegaly": [
        "cardiomegaly", "enlarged heart", "cardiac enlargement",
        "enlarged cardiac silhouette", "heart enlargement",
    ],
    "effusion": [
        "effusion", "pleural effusion", "fluid collection",
        "pleural fluid", "fluid in pleural space",
    ],
    "consolidation": [
        "consolidation", "airspace consolidation", "lobar consolidation",
        "air space disease", "airspace disease", "airspace opacity",
    ],
    "atelectasis": [
        "atelectasis", "volume loss", "lung collapse",
        "subsegmental atelectasis", "basilar atelectasis",
    ],
    "opacity": [
        "opacity", "opacification", "hazy opacity",
        "ground glass opacity", "ground-glass opacity", "ggo",
    ],
    "nodule": [
        "nodule", "pulmonary nodule", "lung nodule",
        "solitary nodule", "nodular density",
    ],
    "mass": [
        "mass", "lung mass", "pulmonary mass", "mediastinal mass",
    ],
    "pneumothorax": [
        "pneumothorax", "collapsed lung", "ptx",
    ],
    "pneumonia": [
        "pneumonia", "pneumonic", "infectious process",
        "pulmonary infection",
    ],
    "edema": [
        "edema", "pulmonary edema", "vascular congestion",
        "cephalization", "pulmonary congestion",
    ],
    "infiltrate": [
        "infiltrate", "pulmonary infiltrate", "infiltrative process",
    ],
    "thickening": [
        "thickening", "pleural thickening", "interstitial thickening",
    ],
    "calcification": [
        "calcification", "calcified", "calcific density",
        "granuloma", "calcified granuloma",
    ],
    "hernia": [
        "hernia", "hiatal hernia", "diaphragmatic hernia",
    ],
    "scoliosis": [
        "scoliosis", "spinal curvature", "lateral curvature",
    ],
    "fracture": [
        "fracture", "rib fracture", "compression fracture",
    ],
    "degenerative": [
        "degenerative", "degenerative changes", "spondylosis",
        "degenerative disc disease", "osteophyte",
    ],
    "emphysema": [
        "emphysema", "hyperinflation", "hyperinflated lungs",
        "copd", "bullous",
    ],
    "normal": [
        "normal", "unremarkable", "no acute", "clear lungs",
        "no active disease", "within normal limits",
    ],
    "device": [
        "device", "pacemaker", "sternal wires", "support device",
        "central line", "picc", "endotracheal tube", "ett",
    ],
}


class ClinicalVocabulary:
    """Synonym-aware clinical entity extraction and normalization.

    Builds a reverse lookup from every synonym variant to its canonical form,
    enabling robust entity matching across different phrasings in radiologist
    reports and MedGemma outputs.
    """

    def __init__(self, synonym_map: dict[str, list[str]] | None = None):
        self.synonym_map = synonym_map or SYNONYM_MAP
        self._build_reverse_map()

    def _build_reverse_map(self) -> None:
        """Build reverse lookup: variant → canonical entity."""
        self._reverse: dict[str, str] = {}
        self._patterns: list[tuple[re.Pattern, str]] = []

        for canonical, variants in self.synonym_map.items():
            for variant in variants:
                self._reverse[variant.lower()] = canonical
                # Create regex pattern for multi-word variants
                pattern = re.compile(r"\b" + re.escape(variant.lower()) + r"\b")
                self._patterns.append((pattern, canonical))

        # Sort patterns by length (longest first) for greedy matching
        self._patterns.sort(key=lambda x: len(x[0].pattern), reverse=True)

    def normalize(self, entity: str) -> str:
        """Normalize an entity string to its canonical form.

        Args:
            entity: Raw entity string (e.g., "pleural effusion").

        Returns:
            Canonical form (e.g., "effusion"), or lowercased input if unknown.
        """
        key = entity.lower().strip()
        return self._reverse.get(key, key)

    def extract_entities(self, text: str) -> list[str]:
        """Extract all clinical entities from free text.

        Uses greedy matching with longest-first priority to handle
        overlapping patterns (e.g., "pleural effusion" before "effusion").

        Args:
            text: Clinical text (findings, impression, or MedGemma output).

        Returns:
            List of unique canonical entities found in the text.
        """
        text_lower = text.lower()
        found: set[str] = set()

        for pattern, canonical in self._patterns:
            if pattern.search(text_lower):
                found.add(canonical)

        return sorted(found)

    @property
    def all_entities(self) -> list[str]:
        """Return all canonical entity names."""
        return sorted(self.synonym_map.keys())

    @property
    def size(self) -> int:
        """Number of canonical entities in the vocabulary."""
        return len(self.synonym_map)
