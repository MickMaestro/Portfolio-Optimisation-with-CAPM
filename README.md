# Portfolio Optimisation with CAPM

## Overview

This project implements portfolio optimisation using the Capital Asset Pricing Model (CAPM) framework. The analysis constructs efficient frontiers for German DAX stocks by performing CAPM regression analysis, calculating covariance matrices, and solving quadratic programming problems to find optimal portfolio allocations under different constraints.

## Features

- **Data Collection**: Automated stock and market data fetching using Yahoo Finance API
- **CAPM Regression**: Individual stock analysis calculating:
  - Alpha (excess return)
  - Beta (market sensitivity)
  - Expected returns based on CAPM formula
  - Idiosyncratic risk estimation
- **Portfolio Optimisation**: Quadratic programming implementation using CVXOPT for:
  - Minimum variance portfolios
  - Target return portfolios
  - Efficient frontier construction
- **Constraint Handling**: Support for both equality and inequality constraints
- **Visualisation**: Efficient frontier plots with individual asset positioning

## Requirements

### Dependencies

```
yfinance
pandas
numpy
statsmodels
matplotlib
cvxopt
```

### Python Version
- Python 3.7 or higher
- CVXOPT solver compatibility

## Installation

1. Clone or download the project files
2. Install required packages:
```bash
pip install yfinance pandas numpy statsmodels matplotlib cvxopt
```

## Usage

### Running the Analysis

Execute the main script:

```bash
python "Portfolio Optimisation with CAPM.py"
```

### Configuration

The default configuration analyses the following German stocks:
- SAP.DE (SAP)
- SIE.DE (Siemens)
- DTE.DE (Deutsche Telekom)
- ALV.DE (Allianz)
- BAS.DE (BASF)
- BAYN.DE (Bayer)
- MBG.DE (Mercedes-Benz Group)
- BMW.DE (BMW)
- VOW3.DE (Volkswagen)
- ADS.DE (Adidas)

Against the DAX40 market index (^GDAXI) with a 2% risk-free rate.

To modify the portfolio, edit the `company_stocks` list:

```python
company_stocks = [
    'SAP.DE',    # Your stock choices here
    'SIE.DE',    
    # Add or remove as needed
]
```

## Methodology

### CAPM Analysis

1. **Data Retrieval**: Downloads 5 years of daily stock and market data
2. **Return Calculation**: Computes daily returns using adjusted closing prices
3. **Excess Return Calculation**: Subtracts risk-free rate from stock and market returns
4. **Regression Analysis**: Performs OLS regression for each stock against market returns

### Portfolio Construction

1. **Covariance Matrix**: Constructs using CAPM framework:
   - Diagonal: β²σ²ₘ + σ²ᵢ (systematic + idiosyncratic risk)
   - Off-diagonal: βᵢβⱼσ²ₘ (systematic risk correlation)
2. **Optimisation**: Solves quadratic programming problem to minimise portfolio variance
3. **Constraint Types**:
   - Equality: Exact target return achievement
   - Inequality: Minimum target return achievement

### Efficient Frontier

- **Target Return Range**: From minimum to maximum individual asset returns
- **Portfolio Points**: 10 equally spaced target returns
- **Risk Calculation**: Portfolio standard deviation using covariance matrix
- **Weight Constraints**: No short selling (weights ≥ 0), full investment (Σw = 1)

## Output Files

The programme generates efficient frontier visualisations:

- `efficient_frontier_equality.png` - Equality constraint frontier
- `efficient_frontier_both.png` - Comparison of both constraint types

## Key Functions

### Data Processing Functions

- `download_data()` - Downloads and processes stock and market data
- `capm_regression()` - Performs CAPM regression analysis

### Portfolio Construction Functions

- `construct_covariance_matrix()` - Builds CAPM-based covariance matrix
- `optimize_portfolio()` - Solves quadratic programming for optimal weights
- `compute_efficient_frontier()` - Generates multiple portfolio points
- `plot_efficient_frontier()` - Creates visualisation plots

## CAPM Framework

### Theoretical Foundation

The Capital Asset Pricing Model assumes:
- Investors are rational and risk-averse
- Markets are efficient with no transaction costs
- All investors have homogeneous expectations
- Risk-free borrowing and lending available

### Mathematical Implementation

**CAPM Equation**: E(Rᵢ) = Rf + βᵢ[E(Rₘ) - Rf]

Where:
- E(Rᵢ) = Expected return of asset i
- Rf = Risk-free rate
- βᵢ = Beta of asset i
- E(Rₘ) = Expected market return

**Portfolio Variance**: σ²p = w'Σw

Where Σ is the covariance matrix constructed using CAPM parameters.

## Regression Results

The analysis provides detailed statistics for each stock:
- **Alpha**: Excess return beyond CAPM prediction
- **Beta**: Sensitivity to market movements
- **P-values**: Statistical significance tests
- **R-squared**: Explanatory power of market factor
- **Expected Returns**: CAPM-based annual return estimates
- **Idiosyncratic Risk**: Stock-specific variance

## Optimisation Constraints

### Equality Constraint
- **Return Constraint**: w'μ = μₚ (exact target return)
- **Budget Constraint**: w'1 = 1 (full investment)
- **No Short Selling**: w ≥ 0

### Inequality Constraint
- **Return Constraint**: w'μ ≥ μₚ (minimum target return)
- **Budget Constraint**: w'1 = 1 (full investment)
- **No Short Selling**: w ≥ 0

## Error Handling

The implementation includes:
- Missing data removal and index alignment
- Numerical stability checks in matrix operations
- Optimisation failure detection and reporting
- Graceful handling of singular matrices

## Performance Considerations

- **Data Quality**: Automatic removal of missing values
- **Numerical Stability**: Use of robust optimisation solvers
- **Computational Efficiency**: Vectorised operations where possible
- **Memory Management**: Efficient matrix storage and operations

## Limitations

- **CAPM Assumptions**: May not hold in practice (market efficiency, constant beta)
- **Historical Data**: Past relationships may not predict future performance
- **Single Factor Model**: Ignores other risk factors beyond market risk
- **No Transaction Costs**: Real-world costs not considered
- **Static Analysis**: Assumes constant parameters over time

## Future Enhancements

Potential improvements include:
- Multi-factor models (Fama-French, Carhart)
- Time-varying beta estimation
- Transaction cost incorporation
- Dynamic portfolio rebalancing
- Risk-adjusted performance metrics (Sharpe ratio, Treynor ratio)
- Monte Carlo simulation for robustness testing

## Author

Ibukunoluwa Michael Adebanjo
