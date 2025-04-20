import numpy as np
from Data.business_data import historical_data


def weekly_cashflow_sales_analysis():
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
    unit_price = 3.4

    # Assume initial cash balance (e.g., starting working capital)
    initial_cash = 0

    # Calibrate parameters from historical data:
    mean_sales = np.mean(historical_sales)
    std_sales = np.std(historical_sales)
    mean_expenses = np.mean(historical_expenses)
    std_expenses = np.std(historical_expenses)

    # -------------------------------
    # Step 2: Monte Carlo Simulation for Next Week's Sales and Cash Flow
    # -------------------------------

    n_forecast_days = 7   # forecast period: one week
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
    print(negative_cashflow_probability)

    if negative_cashflow_probability <= 5 and negative_cashflow_probability > 0:
        statement = "kuna a small chance cashflow itaenda vibaya. Lakini bado iko very strong as of now."
    elif negative_cashflow_probability == 0:
        statement = "cash flow iko fiti as of now"
    elif negative_cashflow_probability > 5 and negative_cashflow_probability < 20:
        statement = "chance a cashflow mbaya iko this week but si nyingi sana but bado uchunge. "
    elif negative_cashflow_probability > 20 and negative_cashflow_probability < 50:
        statement = "kuna chance kubwa cashflow itaenda vibaya this week. Chunga sana"
    elif negative_cashflow_probability > 50 and negative_cashflow_probability <= 100:
        statement = "sai cashflow ni mbaya. Tafuta options kama unaweza kuwa nazo."
    

    # -------------------------------
    # Step 4: Findings Summary (Non-Technical Output)
    # -------------------------------

    output_summary = (
  f" ## Weekly Sales and Cash Flow Analysis:\n"
    f"-------------------------------------------\n"
    f"- Uko likely kuuza at least**{sales_ci_lower:.2f} units** na at most inaweza fika around **{sales_ci_upper:.2f} units**.\n"
    f"- The projected final cash flow iko likely kufika at least **KES {cashflow_ci_lower:.2f}** na at most around **KES {cashflow_ci_upper:.2f}**.\n"
    f"- Based on current conditions and the sales forecast, **{statement}**\n"
    )

    #print(output_summary)
    return output_summary
