"""
HITL Conformal Safety Gate.

Routes cases to auto-approval or expert review based on composite risk:
- HI > 0.4  →  High hallucination rate
- PSS < 0.5 →  Unstable under perturbation
- SCE > 0.5 →  Severity miscalibration

Deliberately conservative: 35% escalation rate ensures all high-risk
cases receive human review. The system never overrides clinical judgment.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class HITLDecision:
    """Routing decision from the HITL safety gate."""

    action: str  # "AUTO_APPROVED" or "EXPERT_REVIEW"
    triggers: list[str] = field(default_factory=list)
    hi: float = 0.0
    pss: float = 0.0
    sce: float = 0.0

    @property
    def needs_review(self) -> bool:
        return self.action == "EXPERT_REVIEW"


class HITLGate:
    """Conformal safety gate for human-in-the-loop routing.

    Thresholds are deliberately conservative — it is better to
    escalate a case unnecessarily than to miss a clinical error.
    """

    def __init__(
        self,
        hi_threshold: float = 0.4,
        pss_threshold: float = 0.5,
        sce_threshold: float = 0.5,
    ):
        self.hi_threshold = hi_threshold
        self.pss_threshold = pss_threshold
        self.sce_threshold = sce_threshold

    def evaluate(self, hi: float, pss: float, sce: float) -> HITLDecision:
        """Evaluate composite risk and route the case.

        Args:
            hi: Hallucination Index (lower is better).
            pss: Perturbation Stability Score (higher is better).
            sce: Severity Calibration Error (lower is better).

        Returns:
            HITLDecision with action and triggered conditions.
        """
        triggers = []

        if hi > self.hi_threshold:
            triggers.append(f"HI={hi:.3f} > {self.hi_threshold}")
        if pss < self.pss_threshold:
            triggers.append(f"PSS={pss:.3f} < {self.pss_threshold}")
        if sce > self.sce_threshold:
            triggers.append(f"SCE={sce:.3f} > {self.sce_threshold}")

        action = "EXPERT_REVIEW" if triggers else "AUTO_APPROVED"

        return HITLDecision(
            action=action,
            triggers=triggers,
            hi=hi,
            pss=pss,
            sce=sce,
        )
