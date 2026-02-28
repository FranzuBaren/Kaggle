"""Unit tests for CXR-Sentinel metrics computation."""

import pytest

from src.utils.metrics import (
    bootstrap_ci,
    clinical_f1,
    diagnostic_concordance_rate,
    hallucination_index,
    rouge_l_score,
    severity_calibration_error,
)


class TestDCR:
    def test_perfect_recall(self):
        assert diagnostic_concordance_rate({"a", "b"}, {"a", "b", "c"}) == 1.0

    def test_partial_recall(self):
        assert diagnostic_concordance_rate({"a", "b"}, {"a"}) == 0.5

    def test_zero_recall(self):
        assert diagnostic_concordance_rate({"a", "b"}, {"c"}) == 0.0

    def test_empty_gt(self):
        assert diagnostic_concordance_rate(set(), set()) == 1.0

    def test_empty_gt_nonempty_pred(self):
        assert diagnostic_concordance_rate(set(), {"a"}) == 0.0


class TestHI:
    def test_no_hallucination(self):
        assert hallucination_index({"a", "b"}, {"a", "b", "c"}) == 0.0

    def test_all_hallucination(self):
        assert hallucination_index({"x", "y"}, {"a", "b"}) == 1.0

    def test_partial_hallucination(self):
        assert hallucination_index({"a", "x"}, {"a", "b"}) == 0.5

    def test_empty_pred(self):
        assert hallucination_index(set(), {"a"}) == 0.0


class TestF1:
    def test_perfect_f1(self):
        assert clinical_f1({"a", "b"}, {"a", "b"}) == 1.0

    def test_zero_f1(self):
        assert clinical_f1({"a"}, {"b"}) == 0.0

    def test_both_empty(self):
        assert clinical_f1(set(), set()) == 1.0

    def test_partial_overlap(self):
        f1 = clinical_f1({"a", "b", "c"}, {"a", "b", "d"})
        # P = 2/3, R = 2/3, F1 = 2/3
        assert abs(f1 - 2 / 3) < 1e-6


class TestROUGEL:
    def test_identical(self):
        assert rouge_l_score("hello world", "hello world") == 1.0

    def test_no_overlap(self):
        assert rouge_l_score("hello world", "foo bar") == 0.0

    def test_partial_overlap(self):
        score = rouge_l_score("the cat sat on the mat", "the cat on the mat")
        assert 0.5 < score < 1.0

    def test_empty(self):
        assert rouge_l_score("", "hello") == 0.0


class TestSCE:
    weights = {"mild": 1.0, "moderate": 1.5, "severe": 2.0, "critical": 3.0}

    def test_same_severity(self):
        assert severity_calibration_error("mild", "mild", self.weights) == 0.0

    def test_different_severity(self):
        sce = severity_calibration_error("mild", "severe", self.weights)
        assert sce > 0

    def test_max_difference(self):
        sce = severity_calibration_error("mild", "critical", self.weights)
        # (3.0 - 1.0) / 3.0 = 0.667
        assert abs(sce - 2.0 / 3.0) < 1e-6


class TestBootstrapCI:
    def test_constant_values(self):
        mean, lo, hi = bootstrap_ci([5.0] * 100)
        assert mean == 5.0
        assert abs(lo - 5.0) < 0.01
        assert abs(hi - 5.0) < 0.01

    def test_returns_tuple(self):
        result = bootstrap_ci([1.0, 2.0, 3.0])
        assert len(result) == 3

    def test_ci_contains_mean(self):
        mean, lo, hi = bootstrap_ci([1, 2, 3, 4, 5])
        assert lo <= mean <= hi

    def test_reproducible(self):
        r1 = bootstrap_ci([1, 2, 3, 4, 5], seed=42)
        r2 = bootstrap_ci([1, 2, 3, 4, 5], seed=42)
        assert r1 == r2
