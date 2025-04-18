#how likely could this supply chain fail
import numpy as np
import math
from scipy.stats import norm

def supply_chain_fragility():

    # -------------------------------
    # Simulation Parameters
    # -------------------------------
    np.random.seed(42)

    n_simulations = 100000    # Number of simulation paths
    cycle_length = 5          # Order cycle horizon in days for lead-time evaluation
    baseline_lead_time = 3  # Baseline lead time in days (no disruption)

    # Markov chain parameters for supply chain state:
    # Two states: "normal" and "disrupted"
    # Transition probabilities:
    p_nn = 0.93   # P(normal -> normal)
    p_ns = 0.07   # P(normal -> disrupted)
    p_ss = 0.75   # P(disrupted -> disrupted)
    p_sn = 0.25   # P(disrupted -> normal)

    # Additional delay if a disruption is encountered:
    # Use a discrete probability tree:
    # With probability 0.4, extra delay = 2 days
    # With probability 0.4, extra delay = 4 days
    # With probability 0.2, extra delay = 7 days
    def sample_additional_delay():
        r = np.random.rand()
        if r < 0.4:
            return 2.0
        elif r < 0.8:
            return 4.0
        else:
            return 7.0

    # Define the effective lead time threshold for supply chain failure impact:
    lead_time_threshold = 4  # days; effective lead time above this indicates failure

    # -------------------------------
    # Monte Carlo Simulation: Supply Chain Disruption and Lead Time
    # -------------------------------

    # This simulation models an order cycle of 'cycle_length' days for one supplier.
    # We simulate a discrete-time Markov chain. Each simulation path:
    # - Starts in "normal" state.
    # - For each day in the cycle, transitions according to the given probabilities.
    # - If at any day the state becomes "disrupted", we flag the cycle as disrupted.
    # - If disrupted, an additional delay is sampled from the probability tree.
    # - Effective lead time = baseline_lead_time + (additional delay if disruption occurred, else 0).

    effective_lead_times = np.zeros(n_simulations)

    for sim in range(n_simulations):
        state = "normal"
        disrupted = False
        # Simulate the state over the order cycle period
        for day in range(cycle_length):
            if state == "normal":
                # From normal, possible transition to disruption with probability p_ns
                if np.random.rand() < p_ns:
                    state = "disrupted"
                    disrupted = True
                else:
                    state = "normal"
            else:  # state == "disrupted"
                # Remain disrupted with probability p_ss, or recover with probability p_sn
                if np.random.rand() < p_ss:
                    state = "disrupted"
                    disrupted = True
                else:
                    state = "normal"
        # Determine effective lead time
        if disrupted:
            extra_delay = sample_additional_delay()
        else:
            extra_delay = 0.0
        effective_lead_times[sim] = baseline_lead_time + extra_delay

    # -------------------------------
    # Analysis of Simulation Results
    # -------------------------------

    # Calculate the proportion of simulations where effective lead time exceeds the threshold.
    failures = effective_lead_times > lead_time_threshold
    failure_probability = np.mean(failures) * 100  # expressed as percentage

    # Compute mean effective lead time over all simulation paths.
    mean_lead_time = np.mean(effective_lead_times)

    # Compute the 95% confidence interval (percentile interval) for effective lead times.
    ci_lower, ci_upper = np.percentile(effective_lead_times, [2.5, 97.5])

    # -------------------------------
    # Findings Summary (Non-Technical Output)
    # -------------------------------
    output_summary = (f"# Supply Chain Fragility Findings:\n"
    f"---------\n"
    f"Interpretation:\n"
    f"There is a {failure_probability:.2f}% chance that the effective lead time for receiving an order (comprising the baseline lead time plus any additional delay from disruptions) will exceed {lead_time_threshold:.2f} days.\n" )

    #print(output_summary)
    return output_summary
