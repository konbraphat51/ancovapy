# ancovapy

A Python library for scientifically rigorous ANCOVA (Analysis of Covariance) and Bayesian ANCOVA analysis.

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: BSL-1.0](https://img.shields.io/badge/License-BSL_1.0-blue.svg)](https://www.boost.org/LICENSE_1_0.txt)

## Features

- **Classical ANCOVA**: Frequentist analysis using statsmodels with configurable Sum of Squares types
- **Bayesian ANCOVA**: MCMC-based inference using PyMC with credible intervals and diagnostics
- **Flexible Covariates**: Support for quantitative (Q), categorical (C), and group (G) types
- **Post-hoc Tests**: Automatic Tukey HSD for group comparisons
- **Post-Pre Designs**: Specialized analysis for pre-post experimental designs
- **Strict Typing**: Full type hints validated by mypy
- **Scientific Rigor**: Based on established statistical methodology

## Quick Start

```bash
pip install ancovapy
```

```python
import numpy as np
from ancovapy import ANCOVA, BayesianANCOVA

# Classical ANCOVA
ancova = ANCOVA(ss_type=2)
result = ancova.fit(
    outcome_data,
    covariates={
        "age": (age_data, "Q"),        # Quantitative
        "treatment": (groups, "G"),     # Group (with post-hoc tests)
    }
)
print(f"F = {result.f_statistic:.2f}, p = {result.p_value:.4f}")

# Bayesian ANCOVA
bayesian = BayesianANCOVA(mcmc_samples=2000, random_seed=42)
result = bayesian.fit(outcome_data, covariates)
print(f"R-hat: {result.rhat_max:.3f}, ESS: {result.ess_bulk_min:.0f}")
```

## Covariate Types

Specify covariates as tuples: `(data, type)`

- **"Q"** (Quantitative): Continuous numerical variables
- **"C"** (Categorical): Categorical factors without pairwise comparisons  
- **"G"** (Group): Categorical factors with automatic post-hoc tests

## Documentation

- **[Quick Start & Examples](docs/EXAMPLES.md)**: Comprehensive usage examples
- **[API Reference](docs/API.md)**: Detailed API documentation
- **[Scientific References](docs/REFERENCES.md)**: Citations and further reading
- **[Architecture](docs/ARCHITECTURE.md)**: SOLID principles and design
- **[Verification](docs/VERIFICATION.md)**: Statistical implementation validation
- **[Contributing](CONTRIBUTING.md)**: Development guidelines
- **[Changelog](CHANGELOG.md)**: Version history

## Key Capabilities

### Sum of Squares Types
- Type I (Sequential)
- Type II (Hierarchical) - default
- Type III (Marginal)

### Bayesian Features
- NUTS sampling with diagnostics (R-hat, ESS)
- Highest Density Intervals (HDI)
- ROPE (Region of Practical Equivalence) analysis
- One-sided and two-sided hypothesis testing

### Post-Pre Design
Both classical and Bayesian approaches support:
- Baseline-adjusted mean changes
- Pairwise group differences
- 95% confidence/credible intervals

## Requirements

- Python 3.9+
- numpy >= 1.20.0
- scipy >= 1.7.0
- pandas >= 1.3.0
- statsmodels >= 0.13.0
- pymc >= 5.0.0
- arviz >= 0.15.0

## Development

```bash
git clone https://github.com/konbraphat51/ancovapy.git
cd ancovapy
uv pip install -e ".[dev]"
pytest tests/
```

## License

This project is licensed under the Boost Software License 1.0 - see the [LICENSE](LICENSE) file for details.

## Citation

```bibtex
@software{ancovapy,
  title = {ancovapy: A Python library for ANCOVA and Bayesian ANCOVA},
  author = {ancovapy contributors},
  year = {2026},
  url = {https://github.com/konbraphat51/ancovapy},
  license = {BSL-1.0}
}
```

## Support

- **Issues**: [GitHub Issues](https://github.com/konbraphat51/ancovapy/issues)
- **Discussions**: [GitHub Discussions](https://github.com/konbraphat51/ancovapy/discussions)
