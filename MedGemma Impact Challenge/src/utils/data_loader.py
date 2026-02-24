"""
Data Loader — OpenI / Indiana University Chest X-Ray Dataset.

Loads 3,955 radiology reports paired with 7,470 CXR images.
Implements stratified sampling across 5 pathology categories:
Normal, Cardiomegaly, Pneumonia, Effusion, Mass (4 cases each = 20 total).
"""

from __future__ import annotations

import csv
import logging
import re
from dataclasses import dataclass, field
from enum import Enum, auto
from pathlib import Path
from typing import Optional

from PIL import Image

logger = logging.getLogger(__name__)


class SeverityLevel(Enum):
    """Clinical severity classification for stratified sampling."""

    NORMAL = auto()
    MILD = auto()
    MODERATE = auto()
    SEVERE = auto()
    CRITICAL = auto()


@dataclass
class ClinicalCase:
    """A single clinical case from the OpenI dataset."""

    case_id: str
    findings: str
    impression: str
    indication: str = ""
    image_paths: list[str] = field(default_factory=list)
    pathology_category: str = "unknown"
    ground_truth_severity: SeverityLevel = SeverityLevel.NORMAL
    ground_truth_entities: list[str] = field(default_factory=list)


# ── Pathology classification rules ──
PATHOLOGY_KEYWORDS = {
    "Cardiomegaly": ["cardiomegaly", "enlarged heart", "cardiac enlargement"],
    "Effusion": ["effusion", "pleural effusion", "pleural fluid"],
    "Pneumonia": ["pneumonia", "consolidation", "infectious", "infiltrate"],
    "Mass": ["mass", "nodule", "tumor", "lesion"],
    "Normal": ["normal", "unremarkable", "no acute", "clear"],
}


class OpenIDataLoader:
    """Loader for the OpenI / Indiana University CXR dataset.

    Supports loading from Kaggle dataset directory or pre-downloaded CSVs.
    Implements stratified cohort selection for balanced validation.
    """

    def __init__(self, data_dir: str | Path):
        self.data_dir = Path(data_dir)
        self.reports_path = self.data_dir / "indiana_reports.csv"
        self.projections_path = self.data_dir / "indiana_projections.csv"

    def load_reports(self) -> list[ClinicalCase]:
        """Load all reports from the OpenI CSV files.

        Returns:
            List of ClinicalCase objects with findings and impressions.
        """
        if not self.reports_path.exists():
            raise FileNotFoundError(
                f"Reports CSV not found at {self.reports_path}. "
                "Download from: https://www.kaggle.com/datasets/raddar/chest-xrays-indiana-university"
            )

        cases = []
        with open(self.reports_path, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                uid = row.get("uid", "").strip()
                findings = row.get("findings", "").strip()
                impression = row.get("impression", "").strip()
                indication = row.get("indication", "").strip()

                if not findings or not impression:
                    continue

                category = self._classify_pathology(findings + " " + impression)
                severity = self._estimate_severity(category, impression)

                cases.append(
                    ClinicalCase(
                        case_id=f"IU-{uid}",
                        findings=findings,
                        impression=impression,
                        indication=indication,
                        pathology_category=category,
                        ground_truth_severity=severity,
                    )
                )

        logger.info(f"Loaded {len(cases)} reports from OpenI dataset")
        return cases

    def select_cohort(
        self,
        cases: list[ClinicalCase],
        n_per_category: int = 4,
        seed: int = 42,
    ) -> list[ClinicalCase]:
        """Select a stratified cohort across pathology categories.

        Args:
            cases: Full list of loaded cases.
            n_per_category: Number of cases per pathology category.
            seed: Random seed for reproducible selection.

        Returns:
            Stratified subset of cases (default: 5 categories × 4 = 20).
        """
        import random

        rng = random.Random(seed)
        by_category: dict[str, list[ClinicalCase]] = {}

        for case in cases:
            cat = case.pathology_category
            by_category.setdefault(cat, []).append(case)

        cohort = []
        for category in PATHOLOGY_KEYWORDS:
            pool = by_category.get(category, [])
            if len(pool) < n_per_category:
                logger.warning(
                    f"Category '{category}' has only {len(pool)} cases "
                    f"(need {n_per_category})"
                )
                cohort.extend(pool)
            else:
                cohort.extend(rng.sample(pool, n_per_category))

        logger.info(
            f"Selected cohort: {len(cohort)} cases across "
            f"{len(set(c.pathology_category for c in cohort))} categories"
        )
        return cohort

    def _classify_pathology(self, text: str) -> str:
        """Classify a case into one of 5 pathology categories."""
        text_lower = text.lower()
        for category, keywords in PATHOLOGY_KEYWORDS.items():
            if category == "Normal":
                continue
            if any(kw in text_lower for kw in keywords):
                return category
        return "Normal"

    def _estimate_severity(self, category: str, impression: str) -> SeverityLevel:
        """Estimate clinical severity from pathology category and impression."""
        if category == "Normal":
            return SeverityLevel.NORMAL
        if category in ("Mass",):
            return SeverityLevel.SEVERE

        text_lower = impression.lower()
        if any(w in text_lower for w in ("large", "massive", "bilateral", "severe")):
            return SeverityLevel.SEVERE
        if any(w in text_lower for w in ("moderate", "significant")):
            return SeverityLevel.MODERATE
        return SeverityLevel.MILD
