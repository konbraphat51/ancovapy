# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0] - 2026-01-31

### Added

#### Core Features
- Classical ANCOVA implementation using statsmodels
- Bayesian ANCOVA implementation using PyMC
- Support for three covariate types:
  - Q: Quantitative (continuous variables)
  - C: Categorical (without group comparisons)
  - G: Group (categorical with pairwise post-hoc tests)
- Configurable Sum of Squares types (Type I, II, III)
- Post-pre experimental design analysis functions

#### Statistical Methods
- Tukey HSD post-hoc tests for group comparisons
- MCMC-based Bayesian inference with diagnostics
- Highest Density Intervals (HDI) for Bayesian credible intervals
- ROPE (Region of Practical Equivalence) decision making
- One-sided and two-sided hypothesis testing for Bayesian ANCOVA
- Adjusted mean calculations for post-pre designs

#### Type System
- Strict type hints throughout the codebase
- Full mypy compatibility
- Modern Python 3.9+ type annotations

#### Documentation
- Comprehensive README with usage examples
- 10 scientific references and citations
- API reference documentation
- Installation instructions
- Quick start guide

#### Testing
- 20 unit tests with pytest
- Example scripts demonstrating all features:
  - Classical ANCOVA example
  - Bayesian ANCOVA example
  - Post-pre design example
  - Quick start example

#### Development Tools
- uv package manager support
- Black code formatting
- Ruff linting
- mypy type checking
- pytest testing framework

### Dependencies
- numpy >= 1.20.0
- scipy >= 1.7.0
- pandas >= 1.3.0
- statsmodels >= 0.13.0
- pymc >= 5.0.0
- arviz >= 0.15.0

[0.1.0]: https://github.com/konbraphat51/ancovapy/releases/tag/v0.1.0
