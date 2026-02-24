"""Unit tests for CXR-Sentinel agents."""

import pytest

from src.utils.clinical_vocab import ClinicalVocabulary
from src.agents.factchecker import FactChecker
from src.pipeline.hitl_gate import HITLGate, HITLDecision


class TestClinicalVocabulary:
    def setup_method(self):
        self.vocab = ClinicalVocabulary()

    def test_normalize_canonical(self):
        assert self.vocab.normalize("cardiomegaly") == "cardiomegaly"

    def test_normalize_synonym(self):
        assert self.vocab.normalize("enlarged heart") == "cardiomegaly"
        assert self.vocab.normalize("pleural effusion") == "effusion"

    def test_normalize_case_insensitive(self):
        assert self.vocab.normalize("Cardiomegaly") == "cardiomegaly"
        assert self.vocab.normalize("PLEURAL EFFUSION") == "effusion"

    def test_normalize_unknown(self):
        assert self.vocab.normalize("xyzzy") == "xyzzy"

    def test_extract_entities_basic(self):
        text = "There is cardiomegaly with bilateral pleural effusion."
        entities = self.vocab.extract_entities(text)
        assert "cardiomegaly" in entities
        assert "effusion" in entities

    def test_extract_entities_synonym(self):
        text = "The heart appears to show cardiac enlargement with fluid in the pleural space."
        entities = self.vocab.extract_entities(text)
        assert "cardiomegaly" in entities

    def test_extract_entities_empty(self):
        assert self.vocab.extract_entities("") == []

    def test_extract_entities_no_clinical(self):
        text = "The patient arrived at the hospital this morning."
        # Should not extract clinical entities from non-clinical text
        entities = self.vocab.extract_entities(text)
        # May or may not find anything, but should not crash
        assert isinstance(entities, list)

    def test_vocab_size(self):
        assert self.vocab.size == 20

    def test_all_entities(self):
        entities = self.vocab.all_entities
        assert "cardiomegaly" in entities
        assert "effusion" in entities
        assert len(entities) == 20


class TestFactChecker:
    def setup_method(self):
        self.fc = FactChecker()

    def test_perfect_match(self):
        result = self.fc.validate(
            case_id="test-001",
            pred_impression="Cardiomegaly with effusion.",
            pred_entities=["cardiomegaly", "effusion"],
            gt_impression="Cardiomegaly with pleural effusion.",
            gt_entities=["cardiomegaly", "effusion"],
            pred_severity="moderate",
            gt_severity="moderate",
        )
        assert result.dcr == 1.0
        assert result.hallucination_index == 0.0
        assert result.clinical_f1 == 1.0
        assert len(result.missed_entities) == 0
        assert len(result.hallucinated_entities) == 0

    def test_missed_entity(self):
        result = self.fc.validate(
            case_id="test-002",
            pred_impression="Cardiomegaly.",
            pred_entities=["cardiomegaly"],
            gt_impression="Cardiomegaly with effusion.",
            gt_entities=["cardiomegaly", "effusion"],
        )
        assert result.dcr == 0.5
        assert "effusion" in result.missed_entities

    def test_hallucinated_entity(self):
        result = self.fc.validate(
            case_id="test-003",
            pred_impression="Cardiomegaly with pneumothorax.",
            pred_entities=["cardiomegaly", "pneumothorax"],
            gt_impression="Cardiomegaly.",
            gt_entities=["cardiomegaly"],
        )
        assert result.hallucination_index == 0.5
        assert "pneumothorax" in result.hallucinated_entities


class TestHITLGate:
    def setup_method(self):
        self.gate = HITLGate()

    def test_auto_approved(self):
        decision = self.gate.evaluate(hi=0.1, pss=0.9, sce=0.2)
        assert decision.action == "AUTO_APPROVED"
        assert not decision.needs_review
        assert len(decision.triggers) == 0

    def test_hi_trigger(self):
        decision = self.gate.evaluate(hi=0.5, pss=0.9, sce=0.2)
        assert decision.action == "EXPERT_REVIEW"
        assert decision.needs_review
        assert any("HI" in t for t in decision.triggers)

    def test_pss_trigger(self):
        decision = self.gate.evaluate(hi=0.1, pss=0.3, sce=0.2)
        assert decision.needs_review
        assert any("PSS" in t for t in decision.triggers)

    def test_sce_trigger(self):
        decision = self.gate.evaluate(hi=0.1, pss=0.9, sce=0.8)
        assert decision.needs_review
        assert any("SCE" in t for t in decision.triggers)

    def test_multiple_triggers(self):
        decision = self.gate.evaluate(hi=0.6, pss=0.3, sce=0.8)
        assert decision.needs_review
        assert len(decision.triggers) == 3

    def test_boundary_values(self):
        # At exact thresholds, should NOT trigger (> not >=)
        decision = self.gate.evaluate(hi=0.4, pss=0.5, sce=0.5)
        assert decision.action == "AUTO_APPROVED"
