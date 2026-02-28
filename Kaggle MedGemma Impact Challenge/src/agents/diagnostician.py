"""
Diagnostician Agent — MedGemma 1.5 4B Multimodal Impression Generation.

Receives a CXR image (384px, SigLIP-encoded) + radiologist findings and
generates a structured clinical impression via severity-aware Chain-of-Thought.

Cross-attends between visual features (opacity, cardiac silhouette, costophrenic
blunting) and textual entities to produce JSON output with:
- Primary/secondary diagnoses
- Severity classification
- Recommended follow-up actions
- Extracted clinical entities
"""

from __future__ import annotations

import json
import logging
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import torch
from PIL import Image

from src.utils.clinical_vocab import ClinicalVocabulary

logger = logging.getLogger(__name__)


@dataclass
class DiagnosticResult:
    """Structured output from the Diagnostician agent."""

    case_id: str
    impression: str
    entities: list[str] = field(default_factory=list)
    severity: str = "unknown"
    primary_diagnosis: str = ""
    secondary_diagnoses: list[str] = field(default_factory=list)
    follow_up: str = ""
    confidence: float = 0.0
    entropy_profile: list[float] = field(default_factory=list)
    inference_time_sec: float = 0.0
    token_count: int = 0


# ── Prompt templates ──

MULTIMODAL_PROMPT = """You are a board-certified radiologist synthesizing a clinical impression.

**Patient Indication:** {indication}
**Radiologist Findings:** {findings}
**CXR Image:** [attached]

Analyze the chest X-ray image and the findings. Provide your assessment as JSON:
{{
  "impression": "<narrative clinical impression>",
  "primary_diagnosis": "<most significant finding>",
  "secondary_diagnoses": ["<other findings>"],
  "severity": "<mild|moderate|severe|critical>",
  "entities": ["<list of clinical entities detected>"],
  "follow_up": "<recommended actions>"
}}

Think step-by-step:
1. Enumerate each finding from text and image
2. Assign severity to each
3. Synthesize into a coherent impression
"""

IMAGE_ONLY_PROMPT = """You are a board-certified radiologist. Analyze this chest X-ray image.
Describe ALL findings visible in the image. Provide your assessment as JSON:
{{
  "impression": "<narrative clinical impression>",
  "entities": ["<list of clinical entities detected>"],
  "severity": "<mild|moderate|severe|critical>"
}}
"""

TEXT_ONLY_PROMPT = """You are a board-certified radiologist synthesizing a clinical impression.

**Patient Indication:** {indication}
**Radiologist Findings:** {findings}

Provide your assessment as JSON:
{{
  "impression": "<narrative clinical impression>",
  "primary_diagnosis": "<most significant finding>",
  "secondary_diagnoses": ["<other findings>"],
  "severity": "<mild|moderate|severe|critical>",
  "entities": ["<list of clinical entities detected>"],
  "follow_up": "<recommended actions>"
}}
"""


class Diagnostician:
    """MedGemma-powered multimodal clinical impression generator.

    Supports three inference modes:
    - multimodal: CXR image + text findings (primary pipeline)
    - image_only: CXR image without text (ablation track)
    - text_only: Findings text without image (ablation track)
    """

    def __init__(
        self,
        engine: Any,
        vocab: Optional[ClinicalVocabulary] = None,
        max_new_tokens: int = 512,
    ):
        self.engine = engine
        self.vocab = vocab or ClinicalVocabulary()
        self.max_new_tokens = max_new_tokens
        self.call_count = 0

    def generate_impression(
        self,
        case_id: str,
        findings: str,
        indication: str = "",
        image: Optional[Image.Image] = None,
        mode: str = "multimodal",
    ) -> DiagnosticResult:
        """Generate a clinical impression for a single case.

        Args:
            case_id: Unique case identifier.
            findings: Radiologist-dictated findings text.
            indication: Patient indication / reason for study.
            image: PIL Image of the chest X-ray (384px recommended).
            mode: One of 'multimodal', 'image_only', 'text_only'.

        Returns:
            DiagnosticResult with structured clinical assessment.
        """
        t0 = time.perf_counter()
        self.call_count += 1

        # Select prompt template
        if mode == "image_only":
            prompt = IMAGE_ONLY_PROMPT
            use_image = True
        elif mode == "text_only":
            prompt = TEXT_ONLY_PROMPT.format(
                findings=findings, indication=indication or "Not provided"
            )
            use_image = False
        else:
            prompt = MULTIMODAL_PROMPT.format(
                findings=findings, indication=indication or "Not provided"
            )
            use_image = True

        # Run MedGemma inference
        raw_output, entropy_profile = self.engine.generate(
            prompt=prompt,
            image=image if use_image else None,
            max_new_tokens=self.max_new_tokens,
            return_entropy=True,
        )

        # Parse structured output
        result = self._parse_output(raw_output, case_id)
        result.entropy_profile = entropy_profile
        result.inference_time_sec = time.perf_counter() - t0

        # Entity extraction with synonym normalization
        if not result.entities:
            result.entities = self.vocab.extract_entities(raw_output)

        logger.info(
            f"[Diagnostician] {case_id} | mode={mode} | "
            f"entities={len(result.entities)} | {result.inference_time_sec:.1f}s"
        )

        return result

    def _parse_output(self, raw: str, case_id: str) -> DiagnosticResult:
        """Parse JSON output from MedGemma, with fallback for malformed responses."""
        # Try to extract JSON block
        json_match = re.search(r"\{.*\}", raw, re.DOTALL)
        if json_match:
            try:
                data = json.loads(json_match.group())
                return DiagnosticResult(
                    case_id=case_id,
                    impression=data.get("impression", raw),
                    entities=data.get("entities", []),
                    severity=data.get("severity", "unknown"),
                    primary_diagnosis=data.get("primary_diagnosis", ""),
                    secondary_diagnoses=data.get("secondary_diagnoses", []),
                    follow_up=data.get("follow_up", ""),
                    confidence=data.get("confidence", 0.0),
                    token_count=len(raw.split()),
                )
            except json.JSONDecodeError:
                logger.warning(f"[Diagnostician] {case_id}: JSON parse failed, using fallback")

        # Fallback: extract entities from raw text
        entities = self.vocab.extract_entities(raw)
        return DiagnosticResult(
            case_id=case_id,
            impression=raw.strip(),
            entities=entities,
            token_count=len(raw.split()),
        )
