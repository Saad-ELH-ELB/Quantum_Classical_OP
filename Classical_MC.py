import numpy as np
from prep_pdfs import *

def MC_uniform_mean(x_min, x_max, num_samples):
    # Generate random samples from a uniform distribution
    samples = np.random.uniform(x_min, x_max, num_samples)
    # Calculate the mean of the samples
    mean_estimate = np.mean(samples)
    return mean_estimate

def european_call_black_MC(F, strike, B, maturity, sigma, num_samples):
    # Generate random normal samples
    Z = np.random.normal(size=(num_samples,))
    # Simulate stock price paths
    F_T = F * np.exp((- 0.5 * sigma**2) * maturity + sigma * np.sqrt(maturity) * Z)
    # Calculate option payoff
    payoff = np.maximum(F_T - strike, 0)

    # Discounted expected payoff
    option_price = B * np.mean(payoff)
    return option_price


def european_call_FGMC(F, strike, B, model_params, flag, maturity, num_samples):
    # Only keep the last column of Sim
    Sim = simulation(model_params, num_samples, maturity, flag)
    if Sim.ndim == 1:
        X_last = Sim
    else:
        X_last = Sim[:, -1]

    # Define adjusement term
    h = lambda t: -np.real(log_characteristic_function(model_params, flag, -1j, t))

    price_sim = B * np.mean(np.maximum(np.zeros(len(X_last)), F* np.exp(
        X_last + h(maturity)) - strike * np.ones(len(X_last))))

    return price_sim
