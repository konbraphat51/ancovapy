"""Tests for Bayesian ANCOVA implementation."""

import numpy as np
import pytest

from ancovapy import BayesianANCOVA
from ancovapy.bayesian_ancova import BayesianANCOVAResult


class TestBayesianANCOVA:
    """Test suite for BayesianANCOVA class."""

    def test_initialization(self) -> None:
        """Test BayesianANCOVA initialization."""
        bancova = BayesianANCOVA(
            hypothesis_type="two-sided",
            mcmc_samples=100,
            mcmc_tune=50,
            mcmc_chains=2,
        )
        assert bancova.hypothesis_type == "two-sided"
        assert bancova.mcmc_samples == 100
        assert bancova.mcmc_tune == 50
        assert bancova.mcmc_chains == 2

    def test_fit_basic(self) -> None:
        """Test basic Bayesian ANCOVA fitting."""
        np.random.seed(42)
        n = 40

        # Create simple dataset
        groups = np.repeat(["A", "B"], n // 2)
        age = np.random.normal(45, 10, n)
        outcome = np.random.normal(100, 15, n)
        outcome[groups == "B"] += 10
        outcome += 0.5 * (age - 45)

        covariates = {
            "age": (age, "Q"),
            "group": (groups, "G"),
        }

        # Use fewer samples for faster testing
        bancova = BayesianANCOVA(
            mcmc_samples=100,
            mcmc_tune=50,
            mcmc_chains=2,
            random_seed=42,
        )
        result = bancova.fit(outcome, covariates)

        # Check result structure
        assert isinstance(result, BayesianANCOVAResult)
        assert len(result.covariate_stats) > 0
        assert result.rhat_max > 0
        assert result.ess_bulk_min > 0
        assert result.divergences >= 0

    def test_fit_with_group_comparisons(self) -> None:
        """Test Bayesian ANCOVA with group comparisons."""
        np.random.seed(42)
        n = 60

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

        bancova = BayesianANCOVA(
            mcmc_samples=100,
            mcmc_tune=50,
            mcmc_chains=2,
            random_seed=42,
        )
        result = bancova.fit(outcome, covariates)

        # Check group comparisons exist
        assert result.group_comparisons is not None
        assert len(result.group_comparisons) > 0

    def test_fit_postpre(self) -> None:
        """Test Bayesian post-pre design analysis."""
        np.random.seed(42)
        n = 40

        groups = np.repeat(["Control", "Treatment"], n // 2)
        pre_scores = np.random.normal(100, 15, n)
        post_scores = pre_scores + np.random.normal(5, 10, n)
        post_scores[groups == "Treatment"] += 8

        bancova = BayesianANCOVA(
            mcmc_samples=100,
            mcmc_tune=50,
            mcmc_chains=2,
            random_seed=42,
        )
        result = bancova.fit_postpre(pre_scores, post_scores, groups)

        # Check result includes adjusted means with credible intervals
        assert result.adjusted_means is not None
        assert len(result.adjusted_means) == 2

        for group, (mean, hdi_low, hdi_high) in result.adjusted_means.items():
            assert isinstance(mean, float)
            assert isinstance(hdi_low, float)
            assert isinstance(hdi_high, float)
            assert hdi_low <= mean <= hdi_high

    def test_one_sided_hypothesis(self) -> None:
        """Test one-sided hypothesis testing."""
        np.random.seed(42)
        n = 40

        groups = np.repeat(["A", "B"], n // 2)
        outcome = np.random.normal(100, 15, n)
        outcome[groups == "B"] += 10

        covariates = {"group": (groups, "G")}

        bancova = BayesianANCOVA(
            hypothesis_type="one-sided",
            mcmc_samples=100,
            mcmc_tune=50,
            mcmc_chains=2,
            random_seed=42,
        )
        result = bancova.fit(outcome, covariates)

        assert isinstance(result, BayesianANCOVAResult)

    def test_invalid_input_lengths(self) -> None:
        """Test that mismatched input lengths raise error."""
        outcome = np.array([1, 2, 3])
        covariates = {
            "x": (np.array([1, 2, 3, 4]), "Q"),
        }

        bancova = BayesianANCOVA(mcmc_samples=10, mcmc_chains=1)
        with pytest.raises(ValueError):
            bancova.fit(outcome, covariates)

    def test_empty_data(self) -> None:
        """Test that empty data raises error."""
        outcome = np.array([])
        covariates = {"x": (np.array([]), "Q")}

        bancova = BayesianANCOVA(mcmc_samples=10, mcmc_chains=1)
        with pytest.raises(ValueError):
            bancova.fit(outcome, covariates)

    def test_covariate_stats_properties(self) -> None:
        """Test properties of covariate statistics."""
        np.random.seed(42)
        n = 40

        x = np.random.normal(0, 1, n)
        outcome = 5 + 2 * x + np.random.normal(0, 1, n)

        covariates = {"x": (x, "Q")}

        bancova = BayesianANCOVA(
            mcmc_samples=100,
            mcmc_tune=50,
            mcmc_chains=2,
            random_seed=42,
        )
        result = bancova.fit(outcome, covariates)

        stat = result.covariate_stats[0]
        # Check probability properties
        assert 0 <= stat.prob_positive <= 1
        assert 0 <= stat.prob_negative <= 1
        # Should sum to approximately 1
        assert abs(stat.prob_positive + stat.prob_negative - 1.0) < 0.01

    def test_rope_decision(self) -> None:
        """Test ROPE decision making."""
        np.random.seed(42)
        n = 40

        groups = np.repeat(["A", "B"], n // 2)
        outcome = np.random.normal(100, 15, n)
        # Small effect
        outcome[groups == "B"] += 0.5

        covariates = {"group": (groups, "G")}

        bancova = BayesianANCOVA(
            mcmc_samples=100,
            mcmc_tune=50,
            mcmc_chains=2,
            random_seed=42,
        )

        # Define ROPE
        result = bancova.fit(outcome, covariates, rope=(-2, 2))

        # Check that ROPE decision is made
        if result.group_comparisons:
            for comp in result.group_comparisons:
                assert comp.rope_decision in ["accept", "reject", "undecided", None]

    def test_diagnostics_range(self) -> None:
        """Test that diagnostics are in expected ranges."""
        np.random.seed(42)
        n = 40

        x = np.random.normal(0, 1, n)
        outcome = 5 + 2 * x + np.random.normal(0, 1, n)

        covariates = {"x": (x, "Q")}

        bancova = BayesianANCOVA(
            mcmc_samples=200,
            mcmc_tune=100,
            mcmc_chains=2,
            random_seed=42,
        )
        result = bancova.fit(outcome, covariates)

        # R-hat should be close to 1
        assert 0.9 < result.rhat_max < 1.2  # Allow some slack for test
        # ESS should be positive
        assert result.ess_bulk_min > 0
        # Divergences should be non-negative
        assert result.divergences >= 0
