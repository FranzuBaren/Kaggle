"""Integration tests for CXR-Sentinel pipeline (without GPU)."""

import pytest

from src.pipeline.hitl_gate import HITLGate
from src.pipeline.txgemma_pharma import (
    CONDITION_DRUGS,
    DRUG_SMILES,
    TxGemmaPharmacovigilance,
)


class TestTxGemmaPharmacovigilance:
    """Test pharmacovigilance module (without model - uses fallback)."""

    def setup_method(self):
        self.pharma = TxGemmaPharmacovigilance()

    def test_known_condition_mapping(self):
        result = self.pharma.assess_case("test-001", ["effusion"])
        assert result.n_drugs_tested > 0
        assert any(r.drug == "furosemide" for r in result.drug_results)

    def test_multiple_conditions(self):
        result = self.pharma.assess_case(
            "test-002", ["effusion", "pneumonia", "cardiomegaly"]
        )
        assert result.n_drugs_tested >= 3

    def test_unknown_condition(self):
        result = self.pharma.assess_case("test-003", ["xyzzy_disease"])
        assert result.n_drugs_tested == 0

    def test_empty_conditions(self):
        result = self.pharma.assess_case("test-004", [])
        assert result.n_drugs_tested == 0
        assert not result.any_toxic

    def test_drug_smiles_coverage(self):
        """All drugs in condition map should have SMILES."""
        for drugs in CONDITION_DRUGS.values():
            for drug in drugs:
                assert drug in DRUG_SMILES, f"Missing SMILES for {drug}"

    def test_no_duplicate_drug_testing(self):
        """Same drug from multiple conditions should only be tested once."""
        result = self.pharma.assess_case(
            "test-005", ["effusion", "pleural_effusion"]
        )
        drug_names = [r.drug for r in result.drug_results]
        assert len(drug_names) == len(set(drug_names))


class TestHITLIntegration:
    """Test HITL gate with realistic clinical scenarios."""

    def setup_method(self):
        self.gate = HITLGate()

    def test_mild_normal_case(self):
        """Normal CXR should auto-approve."""
        decision = self.gate.evaluate(hi=0.0, pss=1.0, sce=0.0)
        assert not decision.needs_review

    def test_severe_discrepancy(self):
        """Severe case with high hallucination should escalate."""
        decision = self.gate.evaluate(hi=0.6, pss=0.4, sce=0.8)
        assert decision.needs_review
        assert len(decision.triggers) == 3

    def test_borderline_case(self):
        """Case just under thresholds should auto-approve."""
        decision = self.gate.evaluate(hi=0.39, pss=0.51, sce=0.49)
        assert not decision.needs_review
