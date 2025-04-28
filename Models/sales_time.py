#what time do buyres buy
import numpy as np
import math
from datetime import datetime, timedelta
from scipy.stats import beta, poisson

def time_analysis():

    # -------------------------------
    # Step 1: Define Simulation Parameters
    # -------------------------------

    np.random.seed(42)

    # Simulation time: 120 days (approx. 4 months)
    start_date = datetime(2025, 1, 1)
    n_days = 120
    dates = [start_date + timedelta(days=i) for i in range(n_days)]

    # Define Poisson lambdas by day-of-week (0=Monday, 6=Sunday):
    # Example values: Weekdays: 20, Saturday: 30, Sunday: 25
    def lambda_for_day(dt):
        if dt.weekday() < 5:  # Monday-Friday
            return 20
        elif dt.weekday() == 5:  # Saturday
            return 30
        else:  # Sunday
            return 25

    # Define customer segments:
    # Group A (60%): Likely to buy early (between 8:00 and 14:00), modeled by Beta(2,5)
    # Group B (40%): Likely to buy later (between 14:00 and 20:00), modeled by Beta(5,2)
    def sample_purchase_time(segment):
        if segment == "A":
            # Sample from Beta(2,5) and scale to [8,14]
            t = beta.rvs(2, 5)
            return 8 + t * 6
        else:  # segment == "B"
            # Sample from Beta(5,2) and scale to [14,20]
            t = beta.rvs(5, 2)
            return 14 + t * 6

    # -------------------------------
    # Step 2: Simulate Purchase Events for Each Day
    # -------------------------------

    # We'll collect a list of purchase events.
    # Each purchase event will be recorded as: {date, time (in hours, 24h format), day_of_week, segment}
    purchase_events = []

    for current_date in dates:
        day_lambda = lambda_for_day(current_date)
        # Number of purchase events on this day, sampled from Poisson
        n_events = poisson.rvs(day_lambda)
        
        for _ in range(n_events):
            # Randomly assign a customer segment based on 60:40 ratio
            segment = "A" if np.random.rand() < 0.6 else "B"
            # Sample purchase time based on segment behavior
            purchase_time = sample_purchase_time(segment)
            purchase_events.append({
                "date": current_date.strftime("%Y-%m-%d"),
                "day_of_week": current_date.strftime("%A"),
                "time": purchase_time,  # in hours, e.g., 13.5 means 1:30 PM
                "segment": segment
            })

    # Convert purchase times to a NumPy array for analysis.
    purchase_times = np.array([event["time"] for event in purchase_events])
    purchase_segments = np.array([event["segment"] for event in purchase_events])

    # -------------------------------
    # Step 3: Analyze Temporal Purchase Behavior
    # -------------------------------

    # Overall analysis: compute mean, standard deviation, and percentiles.
    overall_mean = np.mean(purchase_times)
    overall_std = np.std(purchase_times)
    percentiles = np.percentile(purchase_times, [2.5, 25, 50, 75, 97.5])

    # Analysis by customer segment:
    group_A_times = purchase_times[purchase_segments == "A"]
    group_B_times = purchase_times[purchase_segments == "B"]

    mean_A = np.mean(group_A_times) if len(group_A_times) > 0 else float('nan')
    std_A = np.std(group_A_times) if len(group_A_times) > 0 else float('nan')
    percentiles_A = np.percentile(group_A_times, [2.5, 25, 50, 75, 97.5]) if len(group_A_times) > 0 else None

    mean_B = np.mean(group_B_times) if len(group_B_times) > 0 else float('nan')
    std_B = np.std(group_B_times) if len(group_B_times) > 0 else float('nan')
    percentiles_B = np.percentile(group_B_times, [2.5, 25, 50, 75, 97.5]) if len(group_B_times) > 0 else None

    # -------------------------------
    # Step 4: Prepare Output Summary (Findings of the Analysis)
    # -------------------------------

    output_summary = (
    f"# Temporal Purchase Behavior Analysis (Aggregated over {n_days} days):\n"

    f"Based on 120 days of simulated purchase events:\n"
    f"- Overall, customers are likely to purchase around **{overall_mean:.2f}** hours (on a 24-hour clock), with considerable spread.\n"
    f"- For Group A, which represents approximately 60% of customers, purchases peak in the early period **(between 8:00 and 14:00 hours)**, with a median around **{percentiles_A[2]:.2f} hours.**\n"
    f"- For Group B, representing about 40% of customers, the peak shifts later **(between 14:00 and 20:00 hours)**, with a median around {percentiles_B[2]:.2f} hours.**\n"

    )

    #print(output_summary)
    return output_summary
