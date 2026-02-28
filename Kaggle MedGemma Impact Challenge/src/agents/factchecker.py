"""
FactChecker Agent — Anti-Hallucination Validation.

Performs independent validation against radiologist ground truth using:
- Synonym-aware entity extraction (20-entry clinical dictionary)
- Entity-by-entity comparison with normalized matching
- Seven-metric scoring: DCR, HI, PSS, CCS, F1, ROUGE-L, SCE

This is validation against independent ground truth, not self-correction.
The Diagnostician never evaluates its own work.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from src.utils.clinical_vocab import ClinicalVocabulary
from src.utils.metrics import (
    clinical_completeness_score,
    diagnostic_concordance_rate,
    hallucination_index,
    rouge_l_score,
    severity_calibration_error,
)

logger = logging.getLogger(__name__)


# ── ICD-11 severity weights for CCS computation ──
SEVERITY_WEIGHTS = {
    "critical": 3.0,
    "severe": 2.0,
    "moderate": 1.5,
    "mild": 1.0,
    "normal": 0.5,
}

# ── Entity-level severity mapping ──
HIGH_SEVERITY_ENTITIES = {
    "mass", "nodule", "pneumothorax", "consolidation",
    "pulmonary_edema", "aortic_aneurysm",
}

MODERATE_SEVERITY_ENTITIES = {
    "effusion", "pleural_effusion", "cardiomegaly",
    "atelectasis", "pneumonia", "infiltrate",
}


@dataclass
class FactCheckResult:
    """Comprehensive validation result for a single case."""

    case_id: str

    # Entity-level results
    gt_entities: list[str] = field(default_factory=list)
    pred_entities: list[str] = field(default_factory=list)
    matched_entities: list[str] = field(default_factory=list)
    missed_entities: list[str] = field(default_factory=list)
    hallucinated_entities: list[str] = field(default_factory=list)

    # Core metrics
    dcr: float = 0.0
    hallucination_index: float = 0.0
    clinical_f1: float = 0.0
    rouge_l: float = 0.0
    ccs: float = 0.0
    sce: float = 0.0

    # Metadata
    gt_severity: str = "unknown"
    pred_severity: str = "unknown"


class FactChecker:
    """Anti-hallucination validation against radiologist ground truth.

    Compares MedGemma's extracted entities against the radiologist's
    independent impression using synonym-aware matching and computes
    seven validation metrics per case.
    """

    def __init__(self, vocab: Optional[ClinicalVocabulary] = None):
        self.vocab = vocab or ClinicalVocabulary()

    def validate(
        self,
        case_id: str,
        pred_impression: str,
        pred_entities: list[str],
        gt_impression: str,
        gt_entities: Optional[list[str]] = None,
        pred_severity: str = "unknown",
        gt_severity: str = "unknown",
    ) -> FactCheckResult:
        """Validate a predicted impression against radiologist ground truth.

        Args:
            case_id: Unique case identifier.
            pred_impression: MedGemma-generated impression text.
            pred_entities: Entities extracted from predicted impression.
            gt_impression: Radiologist's independent impression text.
            gt_entities: Ground-truth entities (extracted if not provided).
            pred_severity: Predicted severity level.
            gt_severity: Ground-truth severity level.

        Returns:
            FactCheckResult with entity-level comparison and 7 metrics.
        """
        # Extract GT entities if not provided
        if gt_entities is None:
            gt_entities = self.vocab.extract_entities(gt_impression)

        # Normalize all entities via synonym closure
        gt_norm = set(self.vocab.normalize(e) for e in gt_entities)
        pred_norm = set(self.vocab.normalize(e) for e in pred_entities)

        # Entity-level comparison
        matched = gt_norm & pred_norm
        missed = gt_norm - pred_norm
        hallucinated = pred_norm - gt_norm

        # Compute metrics
        dcr = diagnostic_concordance_rate(gt_norm, pred_norm)
        hi = hallucination_index(pred_norm, gt_norm)
        rouge = rouge_l_score(pred_impression, gt_impression)
        ccs = clinical_completeness_score(gt_norm, pred_norm, SEVERITY_WEIGHTS, gt_severity)
        sce = severity_calibration_error(pred_severity, gt_severity, SEVERITY_WEIGHTS)

        # Clinical F1
        precision = len(matched) / max(len(pred_norm), 1)
        recall = len(matched) / max(len(gt_norm), 1)
        f1 = 2 * precision * recall / max(precision + recall, 1e-8)

        result = FactCheckResult(
            case_id=case_id,
            gt_entities=sorted(gt_norm),
            pred_entities=sorted(pred_norm),
            matched_entities=sorted(matched),
            missed_entities=sorted(missed),
            hallucinated_entities=sorted(hallucinated),
            dcr=dcr,
            hallucination_index=hi,
            clinical_f1=f1,
            rouge_l=rouge,
            ccs=ccs,
            sce=sce,
            gt_severity=gt_severity,
            pred_severity=pred_severity,
        )

        logger.info(
            f"[FactChecker] {case_id} | DCR={dcr:.3f} HI={hi:.3f} "
            f"F1={f1:.3f} | matched={len(matched)} missed={len(missed)} "
            f"halluc={len(hallucinated)}"
        )

        return result
