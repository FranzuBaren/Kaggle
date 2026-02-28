"""
SentinelPipeline — DAG Execution Engine.

Orchestrates the three-agent adversarial validation pipeline:
  Diagnostician → Challenger → FactChecker → HITL Gate → TxGemma

Each case flows through all agents sequentially. The HITL gate
evaluates composite risk and routes to auto-approval or expert review.
All agent outputs and routing decisions are logged for regulatory audit.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any, Optional

from PIL import Image

from src.agents.diagnostician import Diagnostician, DiagnosticResult
from src.agents.challenger import Challenger, ChallengerResult
from src.agents.factchecker import FactChecker, FactCheckResult
from src.pipeline.hitl_gate import HITLGate, HITLDecision

logger = logging.getLogger(__name__)


@dataclass
class CaseInput:
    """Input data for a single clinical case."""

    case_id: str
    findings: str
    indication: str = ""
    gt_impression: str = ""
    gt_entities: Optional[list[str]] = None
    gt_severity: str = "unknown"
    image: Optional[Image.Image] = None


@dataclass
class PipelineResult:
    """Complete pipeline output for a single case, including audit trail."""

    case_id: str
    pipeline_status: str = "OK"  # OK or ERROR

    # Agent outputs
    diagnostician_result: Optional[DiagnosticResult] = None
    challenger_result: Optional[ChallengerResult] = None
    factchecker_result: Optional[FactCheckResult] = None

    # Routing
    hitl_decision: Optional[HITLDecision] = None

    # Timing
    total_time_sec: float = 0.0
    error_message: str = ""


class SentinelPipeline:
    """Three-agent adversarial validation pipeline with HITL safety gate.

    Usage:
        pipeline = SentinelPipeline(diagnostician, challenger, factchecker)
        results = pipeline.run_batch(cases)
    """

    def __init__(
        self,
        diagnostician: Diagnostician,
        challenger: Challenger,
        factchecker: FactChecker,
        hitl_gate: Optional[HITLGate] = None,
    ):
        self.diagnostician = diagnostician
        self.challenger = challenger
        self.factchecker = factchecker
        self.hitl_gate = hitl_gate or HITLGate()

    def run_case(self, case: CaseInput) -> PipelineResult:
        """Execute the full pipeline for a single case.

        Args:
            case: CaseInput with findings, image, and ground truth.

        Returns:
            PipelineResult with all agent outputs and routing decision.
        """
        t0 = time.perf_counter()

        try:
            # Stage 1: Diagnostician — generate impression
            dx_result = self.diagnostician.generate_impression(
                case_id=case.case_id,
                findings=case.findings,
                indication=case.indication,
                image=case.image,
                mode="multimodal" if case.image is not None else "text_only",
            )

            # Stage 2: Challenger — adversarial stress-test
            ch_result = self.challenger.challenge(
                case_id=case.case_id,
                findings=case.findings,
                original_entities=dx_result.entities,
                image=case.image,
            )

            # Stage 3: FactChecker — validate against ground truth
            fc_result = self.factchecker.validate(
                case_id=case.case_id,
                pred_impression=dx_result.impression,
                pred_entities=dx_result.entities,
                gt_impression=case.gt_impression,
                gt_entities=case.gt_entities,
                pred_severity=dx_result.severity,
                gt_severity=case.gt_severity,
            )

            # Stage 4: HITL Gate — route to approval or review
            hitl = self.hitl_gate.evaluate(
                hi=fc_result.hallucination_index,
                pss=ch_result.mean_pss,
                sce=fc_result.sce,
            )

            total = time.perf_counter() - t0
            logger.info(
                f"[Pipeline] {case.case_id} | {hitl.action} | "
                f"DCR={fc_result.dcr:.3f} | {total:.1f}s"
            )

            return PipelineResult(
                case_id=case.case_id,
                pipeline_status="OK",
                diagnostician_result=dx_result,
                challenger_result=ch_result,
                factchecker_result=fc_result,
                hitl_decision=hitl,
                total_time_sec=total,
            )

        except Exception as e:
            logger.error(f"[Pipeline] {case.case_id} | ERROR: {e}")
            return PipelineResult(
                case_id=case.case_id,
                pipeline_status="ERROR",
                total_time_sec=time.perf_counter() - t0,
                error_message=str(e),
            )

    def run_batch(self, cases: list[CaseInput]) -> list[PipelineResult]:
        """Execute the pipeline for a batch of cases.

        Args:
            cases: List of CaseInput objects.

        Returns:
            List of PipelineResult objects, one per case.
        """
        results = []
        for i, case in enumerate(cases):
            logger.info(f"[Pipeline] Processing case {i+1}/{len(cases)}: {case.case_id}")
            results.append(self.run_case(case))
        return results

    def get_summary_stats(self, results: list[PipelineResult]) -> dict:
        """Compute aggregate statistics across all pipeline results."""
        ok_results = [r for r in results if r.pipeline_status == "OK"]
        if not ok_results:
            return {"n_cases": 0, "n_errors": len(results)}

        fc_results = [r.factchecker_result for r in ok_results if r.factchecker_result]
        ch_results = [r.challenger_result for r in ok_results if r.challenger_result]
        hitl_results = [r.hitl_decision for r in ok_results if r.hitl_decision]

        return {
            "n_cases": len(ok_results),
            "n_errors": len(results) - len(ok_results),
            "mean_dcr": sum(f.dcr for f in fc_results) / len(fc_results),
            "mean_hi": sum(f.hallucination_index for f in fc_results) / len(fc_results),
            "mean_f1": sum(f.clinical_f1 for f in fc_results) / len(fc_results),
            "mean_pss": sum(c.mean_pss for c in ch_results) / len(ch_results),
            "mean_ccs": sum(f.ccs for f in fc_results) / len(fc_results),
            "hitl_rate": sum(1 for h in hitl_results if h.action == "EXPERT_REVIEW")
            / len(hitl_results),
            "mean_time_sec": sum(r.total_time_sec for r in ok_results) / len(ok_results),
        }
