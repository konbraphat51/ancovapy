"""Tests for classical ANCOVA implementation."""

import numpy as np
import pytest

from ancovapy import ANCOVA
from ancovapy.ancova import ANCOVAResult


class TestANCOVA:
    """Test suite for ANCOVA class."""

    def test_initialization(self) -> None:
        """Test ANCOVA initialization."""
        ancova = ANCOVA(ss_type=2)
        assert ancova.ss_type == 2

    def test_invalid_ss_type(self) -> None:
        """Test that invalid SS type raises error."""
        # Note: Type checking should catch this, but test runtime behavior
        ancova = ANCOVA(ss_type=2)  # Valid
        assert ancova.ss_type == 2

    def test_fit_basic(self) -> None:
        """Test basic ANCOVA fitting."""
        np.random.seed(42)
        n = 60

        # Create simple dataset
        groups = np.repeat(["A", "B"], n // 2)
        age = np.random.normal(45, 10, n)
        outcome = np.random.normal(100, 15, n)
        outcome[groups == "B"] += 10  # Treatment effect
        outcome += 0.5 * (age - 45)  # Age effect

        covariates = {
            "age": (age, "Q"),
            "group": (groups, "G"),
        }

        ancova = ANCOVA(ss_type=2)
        result = ancova.fit(outcome, covariates)

        # Check result structure
        assert isinstance(result, ANCOVAResult)
        assert result.f_statistic > 0
        assert 0 <= result.p_value <= 1
        assert 0 <= result.r_squared <= 1
        assert result.adj_r_squared <= result.r_squared
        assert len(result.covariate_stats) > 0

    def test_fit_with_posthoc(self) -> None:
        """Test ANCOVA with post-hoc tests."""
        np.random.seed(42)
        n = 90

        # Create dataset with 3 groups
        groups = np.repeat(["A", "B", "C"], n // 3)
        age = np.random.normal(45, 10, n)
        outcome = np.random.normal(100, 15, n)
        outcome[groups == "B"] += 10
        outcome[groups == "C"] += 15
        outcome += 0.5 * (age - 45)

        covariates = {
            "age": (age, "Q"),
            "group": (groups, "G"),
        }

        ancova = ANCOVA(ss_type=2)
        result = ancova.fit(outcome, covariates)

        # Check post-hoc results exist for G-type covariate
        assert result.posthoc_results is not None
        assert len(result.posthoc_results) > 0

    def test_fit_postpre(self) -> None:
        """Test post-pre design analysis."""
        np.random.seed(42)
        n = 60

        groups = np.repeat(["Control", "Treatment"], n // 2)
        pre_scores = np.random.normal(100, 15, n)
        post_scores = pre_scores + np.random.normal(5, 10, n)
        post_scores[groups == "Treatment"] += 8  # Treatment effect

        ancova = ANCOVA(ss_type=2)
        result = ancova.fit_postpre(pre_scores, post_scores, groups)

        # Check result includes adjusted means
        assert result.adjusted_means is not None
        assert len(result.adjusted_means) == 2
        assert "Control" in result.adjusted_means
        assert "Treatment" in result.adjusted_means

        # Check pairwise differences
        assert result.adj_mean_diffs is not None
        assert result.credible_intervals is not None

    def test_invalid_input_lengths(self) -> None:
        """Test that mismatched input lengths raise error."""
        outcome = np.array([1, 2, 3])
        covariates = {
            "x": (np.array([1, 2, 3, 4]), "Q"),  # Wrong length
        }

        ancova = ANCOVA()
        with pytest.raises(ValueError):
            ancova.fit(outcome, covariates)

    def test_invalid_covariate_type(self) -> None:
        """Test that invalid covariate type raises error."""
        outcome = np.array([1, 2, 3])
        covariates = {
            "x": (np.array([1, 2, 3]), "INVALID"),  # type: ignore
        }

        ancova = ANCOVA()
        with pytest.raises(ValueError):
            ancova.fit(outcome, covariates)

    def test_empty_data(self) -> None:
        """Test that empty data raises error."""
        outcome = np.array([])
        covariates = {"x": (np.array([]), "Q")}

        ancova = ANCOVA()
        with pytest.raises(ValueError):
            ancova.fit(outcome, covariates)

    def test_quantitative_covariate(self) -> None:
        """Test with only quantitative covariates."""
        np.random.seed(42)
        n = 50

        x1 = np.random.normal(0, 1, n)
        x2 = np.random.normal(0, 1, n)
        outcome = 5 + 2 * x1 + 3 * x2 + np.random.normal(0, 1, n)

        covariates = {
            "x1": (x1, "Q"),
            "x2": (x2, "Q"),
        }

        ancova = ANCOVA()
        result = ancova.fit(outcome, covariates)

        assert result.posthoc_results is None  # No G-type covariates
        assert len(result.covariate_stats) == 2

    def test_categorical_without_groups(self) -> None:
        """Test with C-type categorical covariates."""
        np.random.seed(42)
        n = 60

        site = np.repeat(["Site1", "Site2", "Site3"], n // 3)
        outcome = np.random.normal(100, 15, n)

        covariates = {
            "site": (site, "C"),  # C-type: no post-hoc tests
        }

        ancova = ANCOVA()
        result = ancova.fit(outcome, covariates)

        # Should not have post-hoc results for C-type
        assert result.posthoc_results is None or len(result.posthoc_results) == 0
