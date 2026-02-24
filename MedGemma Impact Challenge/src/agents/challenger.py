"""
Challenger Agent — Adversarial Stress-Testing Engine.

Tests MedGemma's diagnostic robustness via three perturbation types:
1. Entity removal — drops key findings to test recall
2. Confounder injection — adds misleading clinical context
3. Severity contradiction — flips severity descriptors

Measures Perturbation Stability Score (PSS): the fraction of core
diagnoses maintained under adversarial input modifications.

PSS is formalized as a Lipschitz bound: ‖f(x) - f(x+δ)‖ ≤ L‖δ‖
"""

from __future__ import annotations

import logging
import random
import re
import time
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Optional

from src.utils.clinical_vocab import ClinicalVocabulary

logger = logging.getLogger(__name__)


class PerturbationType(Enum):
    """Types of adversarial perturbation applied to clinical findings."""

    ENTITY_REMOVAL = auto()
    CONFOUNDER_INJECTION = auto()
    SEVERITY_CONTRADICTION = auto()


# ── Confounder templates ──

CONFOUNDERS = [
    "Patient also reports acute chest pain radiating to the left arm.",
    "History of recent international travel with possible TB exposure.",
    "Incidental note: patient is post-cardiac surgery with sternal wires.",
    "Patient has known history of lung cancer, currently on immunotherapy.",
    "Recent CT showed possible mediastinal lymphadenopathy.",
]

SEVERITY_FLIPS = {
    "mild": "critical",
    "moderate": "severe",
    "severe": "mild",
    "small": "massive",
    "large": "trace",
    "bilateral": "unilateral",
    "acute": "chronic",
    "chronic": "acute",
}


@dataclass
class PerturbationResult:
    """Result of a single adversarial perturbation trial."""

    perturbation_type: PerturbationType
    original_entities: list[str]
    perturbed_entities: list[str]
    original_text: str
    perturbed_text: str
    entities_maintained: list[str] = field(default_factory=list)
    entities_lost: list[str] = field(default_factory=list)
    entities_gained: list[str] = field(default_factory=list)
    stability_score: float = 0.0
    inference_time_sec: float = 0.0


@dataclass
class ChallengerResult:
    """Aggregated results from all perturbation trials for a single case."""

    case_id: str
    perturbation_results: list[PerturbationResult] = field(default_factory=list)
    mean_pss: float = 0.0
    min_pss: float = 0.0
    n_trials: int = 0


class Challenger:
    """Adversarial perturbation engine for clinical robustness testing.

    For each case, applies multiple perturbation types to the input findings
    and measures how many core diagnoses the model maintains. This tests
    robustness at inference time, not just during development.
    """

    def __init__(
        self,
        engine: Any,
        vocab: Optional[ClinicalVocabulary] = None,
        max_new_tokens: int = 512,
        seed: int = 42,
    ):
        self.engine = engine
        self.vocab = vocab or ClinicalVocabulary()
        self.max_new_tokens = max_new_tokens
        self.rng = random.Random(seed)
        self.call_count = 0

    def challenge(
        self,
        case_id: str,
        findings: str,
        original_entities: list[str],
        image: Any = None,
    ) -> ChallengerResult:
        """Run adversarial perturbation battery on a single case.

        Args:
            case_id: Unique case identifier.
            findings: Original radiologist findings text.
            original_entities: Entities from Diagnostician's original output.
            image: Optional CXR image for multimodal perturbation.

        Returns:
            ChallengerResult with per-perturbation stability scores.
        """
        results = []

        for ptype in PerturbationType:
            perturbed_text = self._apply_perturbation(findings, ptype)

            t0 = time.perf_counter()
            self.call_count += 1

            prompt = (
                "You are a board-certified radiologist. Based on these findings, "
                "list the clinical entities present as JSON: "
                f'{{"entities": ["..."]}}\n\nFindings: {perturbed_text}'
            )

            raw_output, _ = self.engine.generate(
                prompt=prompt,
                image=image,
                max_new_tokens=self.max_new_tokens,
                return_entropy=True,
            )

            perturbed_entities = self.vocab.extract_entities(raw_output)

            # Compute stability
            orig_set = set(self.vocab.normalize(e) for e in original_entities)
            pert_set = set(self.vocab.normalize(e) for e in perturbed_entities)

            maintained = orig_set & pert_set
            lost = orig_set - pert_set
            gained = pert_set - orig_set

            stability = len(maintained) / max(len(orig_set), 1)

            results.append(
                PerturbationResult(
                    perturbation_type=ptype,
                    original_entities=list(orig_set),
                    perturbed_entities=list(pert_set),
                    original_text=findings,
                    perturbed_text=perturbed_text,
                    entities_maintained=list(maintained),
                    entities_lost=list(lost),
                    entities_gained=list(gained),
                    stability_score=stability,
                    inference_time_sec=time.perf_counter() - t0,
                )
            )

        pss_scores = [r.stability_score for r in results]

        logger.info(
            f"[Challenger] {case_id} | trials={len(results)} | "
            f"mean_pss={sum(pss_scores)/len(pss_scores):.3f}"
        )

        return ChallengerResult(
            case_id=case_id,
            perturbation_results=results,
            mean_pss=sum(pss_scores) / len(pss_scores) if pss_scores else 0.0,
            min_pss=min(pss_scores) if pss_scores else 0.0,
            n_trials=len(results),
        )

    def _apply_perturbation(self, findings: str, ptype: PerturbationType) -> str:
        """Apply a specific perturbation type to the findings text."""
        if ptype == PerturbationType.ENTITY_REMOVAL:
            return self._remove_entity(findings)
        elif ptype == PerturbationType.CONFOUNDER_INJECTION:
            return self._inject_confounder(findings)
        elif ptype == PerturbationType.SEVERITY_CONTRADICTION:
            return self._contradict_severity(findings)
        return findings

    def _remove_entity(self, text: str) -> str:
        """Remove a random clinical entity from the findings text."""
        entities = self.vocab.extract_entities(text)
        if not entities:
            return text
        target = self.rng.choice(entities)
        return re.sub(re.escape(target), "", text, flags=re.IGNORECASE).strip()

    def _inject_confounder(self, text: str) -> str:
        """Add a misleading clinical context to the findings."""
        confounder = self.rng.choice(CONFOUNDERS)
        return f"{text} {confounder}"

    def _contradict_severity(self, text: str) -> str:
        """Flip severity descriptors in the findings text."""
        result = text
        for original, flipped in SEVERITY_FLIPS.items():
            if original.lower() in result.lower():
                pattern = re.compile(re.escape(original), re.IGNORECASE)
                result = pattern.sub(flipped, result, count=1)
                break
        return result
