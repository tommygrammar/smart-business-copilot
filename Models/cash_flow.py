import numpy as np
from Data.business_data import historical_data


def weekly_cashflow_sales_analysis(unit_price, initial_cash, n_forecast_days  ):
    # -------------------------------
    # Step 1: Generate Synthetic Historical Data for Sales and Expenses
    # -------------------------------


    # Assume historical period of 60 days
    n_history = 730

    # For sales: assume daily sales units follow N(20, 5) (with non-negative truncation)
    historical_sales = historical_data['sales']

    # For expenses: assume daily operating expenses follow N(150, 20)
    historical_expenses = historical_data['operational_costs']

    # Assume a fixed unit price for product sales, e.g., 100 currency units per sale.

    # Assume initial cash balance (e.g., starting working capital)


    # Calibrate parameters from historical data:
    mean_sales = np.mean(historical_sales)
    std_sales = np.std(historical_sales)
    mean_expenses = np.mean(historical_expenses)
    std_expenses = np.std(historical_expenses)

    # -------------------------------
    # Step 2: Monte Carlo Simulation for Next Week's Sales and Cash Flow
    # -------------------------------


    n_simulations = 100000  # number of Monte Carlo paths

    # Arrays to hold simulation results
    total_weekly_sales = np.zeros(n_simulations)   # total units sold in the week
    final_cash_flows = np.zeros(n_simulations)       # final cash flow at the end of the week

    for sim in range(n_simulations):
        weekly_sales = 0.0
        weekly_expenses = 0.0
        # Simulate each day independently for the next week
        for day in range(n_forecast_days):
            # Simulate daily sales: sample from N(mean_sales, std_sales), truncate negative values.
            daily_sales = max(np.random.normal(mean_sales, std_sales), 0)
            weekly_sales += daily_sales
            
            # Simulate daily expenses: sample from N(mean_expenses, std_expenses), truncate negative values.
            daily_expense = max(np.random.normal(mean_expenses, std_expenses), 0)
            weekly_expenses += daily_expense
        
        # Total revenue for the week is sales units * unit_price
        weekly_revenue = weekly_sales * unit_price
        # Final cash flow: initial cash plus weekly revenue minus weekly expenses.
        final_cash = initial_cash + weekly_revenue - weekly_expenses
        
        total_weekly_sales[sim] = weekly_sales
        final_cash_flows[sim] = final_cash

    # -------------------------------
    # Step 3: Analysis of Simulation Results
    # -------------------------------

    # Sales Analysis: Compute mean and 95% confidence intervals
    mean_weekly_sales = np.mean(total_weekly_sales)
    sales_ci_lower, sales_ci_upper = np.percentile(total_weekly_sales, [2.5, 97.5])

    # Cash Flow Analysis: Compute mean and 95% confidence intervals
    mean_cash_flow = np.mean(final_cash_flows)
    cashflow_ci_lower, cashflow_ci_upper = np.percentile(final_cash_flows, [2.5, 97.5])

    # Risk Analysis: Proportion of simulations where final cash flow is negative
    negative_cashflow_count = np.sum(final_cash_flows < 0)
    negative_cashflow_probability = (negative_cashflow_count / n_simulations) * 100

    # -------------------------------
    # Step 4: Findings Summary (Non-Technical Output)
    # -------------------------------

    output_summary = (
  f" # {n_forecast_days} days Cash Flow Analysis:\n"
    f"-------------------------------------------\n"
   f" **Interpretation:**\n"
    f"-------------\n"

    f"- Your business is likely to sell around **{mean_weekly_sales:.2f}** units in the upcoming week (with a 95% confidence interval between **{sales_ci_lower:.2f} and {sales_ci_upper:.2f} units)**.\n\n"
    f"- Considering the current revenue **(at {unit_price} per unit)** and typical daily operating expenses, the projected final cash flow is expected to be **{mean_cash_flow:.2f}** units, with a 95% confidence interval of **[{cashflow_ci_lower:.2f}, {cashflow_ci_upper:.2f}]**.\n\n"
    f"- There is a **{negative_cashflow_probability:.2f}%** chance that the cash flow will be negative, indicating potential financial risk if conditions do not improve.\n\n"


    )

    #print(output_summary)
    return output_summary
