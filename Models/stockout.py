#how likely is it that i run out of stock
import numpy as np
import math
from scipy.stats import norm
from Data.business_data import historical_data

def stockouts(simulation_days):
    
    # -------------------------------
    # Simulation Parameters
    # -------------------------------

    n_simulations = 100000      # number of Monte Carlo paths
    starting_inventory = historical_data['inventory'][-1]

    # Demand parameters (synthetic, representing a small business in Kenya)
    mean_daily_demand = np.mean(historical_data['sales'])
    std_daily_demand = np.std(historical_data['sales'])

    # -------------------------------
    # Monte Carlo Simulation for Stockout
    # -------------------------------

    # We'll simulate daily demand using a normal distribution truncated at 0.
    # For each simulation, we accumulate daily demand until it exceeds the starting inventory.
    # We record the day at which stockout occurs. If stock never runs out within the simulation period,
    # we mark the simulation with simulation_days+1 (indicating no stockout within period).

    stockout_days = np.zeros(n_simulations)  # to store the day when stockout occurs

    for sim in range(n_simulations):
        cumulative_demand = 0.0
        day_stockout = simulation_days + 1  # default: no stockout within simulation period
        for day in range(1, simulation_days + 1):
            # Sample daily demand; ensure non-negative by taking max with zero.
            daily_demand = max(np.random.normal(mean_daily_demand, std_daily_demand), 0)
            cumulative_demand += daily_demand
            if cumulative_demand >= starting_inventory:
                day_stockout = day
                break
        stockout_days[sim] = day_stockout

    # -------------------------------
    # Analysis of Simulation Results
    # -------------------------------

    # Calculate the proportion of simulations that experienced a stockout within the simulation period.
    stockout_occurrences = np.sum(stockout_days <= simulation_days)
    stockout_probability = (stockout_occurrences / n_simulations) * 100  # as percentage

    # For simulations that had a stockout, calculate the average time to stockout.
    if stockout_occurrences > 0:
        avg_time_to_stockout = np.mean(stockout_days[stockout_days <= simulation_days])
        lower_bound, upper_bound = np.percentile(stockout_days[stockout_days <= simulation_days], [2.5, 97.5])
    else:
        avg_time_to_stockout = float('nan')
        lower_bound, upper_bound = float('nan'), float('nan')

    # -------------------------------
    # Prepare and Print Findings (Non-Technical Summary)
    # -------------------------------

    output_summary = (


    f"# Stockout Analysis\n"
    f"Based on 100,000 simulations with historical analysis,the product is likely to run out of stock within **{simulation_days}** days with a probability of **{stockout_probability:.2f}%**.\n"
    f"For cases where stockout occurs, the average time to stockout is approximately **{avg_time_to_stockout:.2f}** days,with a 95% confidence range between **{lower_bound:.2f}** and **{upper_bound:.2f}** days. These findings directly inform inventory planning decisions to avoid overstocking or understocking.\n"
    
    )

    #print(output_summary)
    return output_summary

