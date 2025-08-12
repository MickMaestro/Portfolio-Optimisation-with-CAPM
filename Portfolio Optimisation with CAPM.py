import yfinance as yf
import pandas as pd
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt
from cvxopt import matrix, solvers

# 1. Data Collection
def download_data(companies, market_index, period='5y', interval='1d'):
    # Download data
    market_data = yf.download(market_index, period=period, interval=interval)
    stock_data = yf.download(companies, period=period, interval=interval)
    
    # Calculate returns
    price_col = 'Adj Close' if 'Adj Close' in market_data.columns else 'Close'
    market_data['Return'] = market_data[price_col].pct_change()
    
    if isinstance(stock_data.columns, pd.MultiIndex):
        if 'Adj Close' in stock_data.columns.levels[0]:
            stock_returns = stock_data['Adj Close'].pct_change()
        else:
            stock_returns = stock_data['Close'].pct_change()
    else:
        stock_returns = stock_data[price_col].pct_change()
    
    # Remove invalid values
    market_data = market_data.dropna()
    stock_returns = stock_returns.dropna()
    
    # Ensure same dates
    common_index = market_data.index.intersection(stock_returns.index)
    market_data = market_data.loc[common_index]
    stock_returns = stock_returns.loc[common_index]
    
    return market_data, stock_returns

# 2. Linear Regression for Expected Returns and Risk
def capm_regression(stock_returns, market_data, risk_free_rate):
    # Perform CAPM regression for each stock.
    # Convert annual risk-free rate to daily. 260 trading days per year
    daily_rf = (1 + risk_free_rate)**(1/260) - 1
    
    # Get excess returns
    market_excess_return = market_data['Return'] - daily_rf
    stock_excess_returns = stock_returns.subtract(daily_rf, axis=0)
    
    # Make DataFrame to store results
    regression_results = pd.DataFrame(
        index=stock_returns.columns,
        columns=['Alpha', 'Beta', 'P-Value (Alpha)', 'P-Value (Beta)', 'R-squared']
    )
    
    # Store residuals
    residuals = {}
    
    # Perform regression for each stock
    stocks = list(stock_returns.columns)
    i = 0
    while i < len(stocks):
        stock = stocks[i]
        Y = stock_excess_returns[stock]
        X = market_excess_return
        X_with_const = sm.add_constant(X)
        
        model = sm.OLS(Y, X_with_const).fit()
        
        regression_results.loc[stock, 'Alpha'] = model.params[0]
        regression_results.loc[stock, 'Beta'] = model.params[1]
        regression_results.loc[stock, 'P-Value (Alpha)'] = model.pvalues[0]
        regression_results.loc[stock, 'P-Value (Beta)'] = model.pvalues[1]
        regression_results.loc[stock, 'R-squared'] = model.rsquared
        
        residuals[stock] = model.resid
        i += 1
    
    # Calculate expected returns and idiosyncratic risk
    average_market_return = market_data['Return'].mean()
    expected_returns = pd.Series(index=regression_results.index)
    idiosyncratic_risk = pd.Series(index=regression_results.index)
    
    stocks = list(regression_results.index)
    i = 0
    while i < len(stocks):
        stock = stocks[i]
        beta = regression_results.loc[stock, 'Beta']
        expected_returns[stock] = risk_free_rate + beta * (average_market_return - risk_free_rate)
        idiosyncratic_risk[stock] = np.var(residuals[stock])
        i += 1
    
    return regression_results, expected_returns, idiosyncratic_risk, market_excess_return.var()

# 3. Portfolio Optimization
def construct_covariance_matrix(companies, betas, market_variance, idiosyncratic_risk):
    # Build covariance matrix using CAPM parameters.
    n = len(companies)
    cov_matrix = np.zeros((n, n))
    
    # Fill the matrix
    i = 0
    while i < n:
        j = 0
        while j < n:
            beta_i = betas[companies[i]]
            
            if i == j:  # Diagonal elements
                cov_matrix[i, i] = (beta_i ** 2) * market_variance + idiosyncratic_risk[companies[i]]
            else:  # Off-diagonal elements
                beta_j = betas[companies[j]]
                cov_matrix[i, j] = beta_i * beta_j * market_variance
            j += 1
        i += 1
    
    return cov_matrix

def optimize_portfolio(cov_matrix, expected_returns, target_return, equality_constraint=True):
    # Find the minimum risk portfolio for a given target return.
    n = len(expected_returns)
    P = matrix(2 * cov_matrix)
    q = matrix(np.zeros(n))
    
    # No short selling constraint: w >= 0
    G = matrix(-np.eye(n))
    h = matrix(np.zeros(n))
    
    # Set up equality/inequality constraints
    if equality_constraint:
        # Equality: wᵀμ = μp, wᵀ1 = 1
        A = matrix(np.vstack([
            np.ones(n),
            expected_returns.values
        ]))
        b = matrix([1.0, target_return])
    else:
        # Inequality: wᵀμ ≥ μp, wᵀ1 = 1
        # Add one row to G and h for wᵀμ ≥ μp
        G_extended = np.vstack([
            -np.eye(n),
            -expected_returns.values
        ])
        h_extended = np.hstack([
            np.zeros(n),
            -target_return
        ])
        G = matrix(G_extended)
        h = matrix(h_extended)
        
        # Only sum of weights = 1 as equality constraint
        A = matrix(np.ones(n)).T
        b = matrix([1.0])
    
    # Solve QuadraticProgramming problem
    solvers.options['show_progress'] = False
    solution = solvers.qp(P, q, G, h, A, b)
    
    if solution['status'] != 'optimal':
        return None, None, None
    
    # Get weights
    weights = np.array(solution['x']).flatten()
    
    # Calculate portfolio risk and actual return
    portfolio_risk = np.sqrt(weights.T @ cov_matrix @ weights)
    actual_return = np.sum(weights * expected_returns.values)
    
    return weights, portfolio_risk, actual_return

# 4. Efficient Frontier
def compute_efficient_frontier(cov_matrix, expected_returns, num_points=10, equality_constraint=True):
    # Compute the efficient frontier by varying the target return.
    min_return = min(expected_returns)
    max_return = max(expected_returns)
    
    # Create a range of target returns
    target_returns = np.linspace(min_return, max_return, num_points)
    
    # Lists for storing results
    portfolio_risks = []
    achieved_returns = []
    all_weights = []
    
    # find the minimum risk portfolio for each target return, 
    i = 0
    while i < len(target_returns):
        target = target_returns[i]
        weights, risk, actual_return = optimize_portfolio(
            cov_matrix, expected_returns, target, equality_constraint
        )
        
        if weights is not None:
            portfolio_risks.append(risk)
            achieved_returns.append(actual_return)
            all_weights.append(weights)
        i += 1
    
    return portfolio_risks, achieved_returns, all_weights

def plot_efficient_frontier(risks, returns, asset_risks, asset_returns, asset_names, constraint_type):
    # plot the efficient frontier and individual assets.
    plt.figure(figsize=(10, 6))
    plt.plot(risks, returns, 'o-', linewidth=2, color='green', 
             label=f'Efficient Frontier ({constraint_type})')
    plt.scatter(asset_risks, asset_returns, c='red', marker='.', s=100, label='Individual Assets')
    
    # add labels to individual assets
    i = 0
    while i < len(asset_names):
        plt.annotate(asset_names[i], (asset_risks[i], asset_returns[i]), 
                    xytext=(5, 5), textcoords='offset points')
        i += 1
    
    plt.xlabel('Portfolio Risk (Standard Deviation)')
    plt.ylabel('Portfolio Expected Return')
    plt.title('Efficient Frontier')
    plt.grid(True)
    plt.legend()
    
    return plt

# Main execution
if __name__ == "__main__":
    # Define companies and parameters
    company_stocks = [
        'SAP.DE',    # SAP
        'SIE.DE',    # Siemens
        'DTE.DE',    # Deutsche Telekom
        'ALV.DE',    # Allianz
        'BAS.DE',    # BASF
        'BAYN.DE',   # Bayer
        'MBG.DE',    # Mercedes-Benz Group
        'BMW.DE',    # BMW
        'VOW3.DE',   # Volkswagen
        'ADS.DE',    # Adidas
    ]
    market_index = '^GDAXI'  # DAX
    risk_free_rate = 0.02    # 2% annual
    
    # Step 1: Download and process all the data
    print("Step 1: Downloading all data")
    market_data, stock_returns = download_data(company_stocks, market_index)
    
    # Step 2: Do CAPM regression
    print("Step 2: CAPM regression")
    regression_results, expected_returns, idiosyncratic_risk, market_variance = capm_regression(
        stock_returns, market_data, risk_free_rate
    )
    
    # Display regression results
    print("\nRegression Results:")
    print(regression_results)
    
    #Calculate annual values for easier interpretation
    annual_expected_returns = expected_returns * 260
    annual_idiosyncratic_risk = idiosyncratic_risk * 260
    
    # Combine results into a summary DataFrame
    summary_results = pd.DataFrame({
        'Alpha': regression_results['Alpha'],
        'Beta': regression_results['Beta'],
        'Expected Return (Annual)': annual_expected_returns,
        'Idiosyncratic Risk (Annual)': annual_idiosyncratic_risk
    })
    
    #Show all rows and format floats
    pd.set_option('display.max_rows', None)
    pd.set_option('display.float_format', lambda x: '{:.6f}'.format(x))
    
    print("\nFinal Results:")
    print(summary_results.to_string())
    
    # Step 3: Covariance matrix
    print("\nStep 3: Covariance matrix")
    cov_matrix = construct_covariance_matrix(
        company_stocks, regression_results['Beta'], market_variance, idiosyncratic_risk
    )
    
    # Step 4: Calculate and plot efficient frontiers for the 2 constraint types
    print("\nStep 4: Efficient frontiers")
    
    # Individual asset risks for plotting
    asset_risks = []
    asset_returns = []
    i = 0
    while i < len(company_stocks):
        stock = company_stocks[i]
        asset_risk = np.sqrt(cov_matrix[i, i]) * np.sqrt(260)
        asset_risks.append(asset_risk)
        asset_returns.append(expected_returns[stock] * 260)
        i += 1
    
    # 1. Equality constraint (wᵀμ = μp)
    risks_eq, returns_eq, weights_eq = compute_efficient_frontier(
        cov_matrix, expected_returns, 10, True
    )
    
    # Annualize for plotting
    risks_eq_annual = []
    returns_eq_annual = []
    i = 0
    while i < len(risks_eq):
        risks_eq_annual.append(risks_eq[i] * np.sqrt(260))
        returns_eq_annual.append(returns_eq[i] * 260)
        i += 1
    
    # 2. Inequality constraint (wᵀμ ≥ μp)
    risks_ineq, returns_ineq, weights_ineq = compute_efficient_frontier(
        cov_matrix, expected_returns, 10, False
    )
    
    # Annualize for plotting
    risks_ineq_annual = []
    returns_ineq_annual = []
    i = 0
    while i < len(risks_ineq):
        risks_ineq_annual.append(risks_ineq[i] * np.sqrt(260))
        returns_ineq_annual.append(returns_ineq[i] * 260)
        i += 1
    
    # Plot both efficient frontiers
    plt1 = plot_efficient_frontier(
        risks_eq_annual, returns_eq_annual, 
        asset_risks, asset_returns, 
        company_stocks, "Equality Constraint"
    )
    plt1.savefig('efficient_frontier_equality.png', dpi=300)
    
    plt2 = plot_efficient_frontier(
        risks_ineq_annual, returns_ineq_annual, 
        asset_risks, asset_returns, 
        company_stocks, "Inequality Constraint"
    )
    
    # Add equality constraint frontier for comparison
    plt2.plot(risks_eq_annual, returns_eq_annual, '--', color='orange', 
             label='Efficient Frontier (Equality Constraint)')
    plt2.legend()
    plt2.savefig('efficient_frontier_both.png', dpi=300)
    plt.show()
