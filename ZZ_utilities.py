import numpy as np
import math
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
from scipy.stats import norm


def int_to_binary(value, length):
    binary_representation = bin(value)
    binary_string = binary_representation[2:]
    padded_binary_string = binary_string.zfill(length)
    return padded_binary_string

def get_zero_indices(binary_string):
    zero_indices = []
    for i, bit in enumerate(binary_string):
        if bit == '0':
            zero_indices.append(i)
    return zero_indices


def binary_to_float(binary_str):
    result = 0
    for i, bit in enumerate(binary_str):
        if bit == '1':
            result += 2 ** (-(i+1))
    return result

def get_twos_complement(value, length):
    twos_comp = 2 ** length - int(np.ceil(value))
    twos_comp_bin = int_to_binary(twos_comp, length)
    tow_comp_list = [int(bit) for bit in list(twos_comp_bin)]
    return tow_comp_list


def payoff_function(x, strike, F):
    res = max(0, F*x-strike)
    return res

def european_call_bm(F, K, B, sigma, T):
    d1 = (np.log(F / K) +  0.5 * sigma**2 * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)

    call_price = B*(F * norm.cdf(d1) - K  * norm.cdf(d2))
    return call_price


def fit_function(steps, a, t):
    return a + steps * t

def plot_error_fit(N_queries_list, error_list, title, show_plt = False, save_plt = True, Q_or_C ='Q'):
    """
    Plots and fits the error versus the number of queries on a log-log scale.

    Parameters:
    - N_queries_list: List of oracle query counts.
    - error_list: List of corresponding errors.
    - title: Title for the plot, also used in the saved file name.
    - show_plt: If True, displays the plot. Default is False.
    - save_plt: If True, saves the plot as a PNG file. Default is True.
    - Q_or_C: Determines the initial guess for curve fitting ('Q' for quantum-like, 'C' for classical-like behavior).

    Outputs:
    - Saves or displays a log-log plot of the error and its fitted curve.
    """

    # Set initial guess for fitting parameters based on Q_or_C.
    if Q_or_C == 'Q':
        initial_guess_q = [1.0, -1] # Quantum-like scaling guess.
    else :
        initial_guess_q = [1.0, -0.5] # Classical-like scaling guess.

    # Perform curve fitting (linear fit in log-log space).
    params_q, covariance_q = curve_fit(fit_function, xdata=np.log(N_queries_list), ydata=np.log(error_list),
                                       p0=initial_guess_q)
    a_fit_q, t_fit_q = params_q
    #
    # Plot the original data points.
    plt.plot(N_queries_list, error_list, 'o-', color='orange', label='Error')
    # Plot the fitted curve using the fit parameters.
    plt.plot(N_queries_list, [math.exp(fit_function(math.log(step), a_fit_q, t_fit_q)) for step in N_queries_list],
             label=f"Fit: y = {a_fit_q:.2f} * x^{t_fit_q:.2f}", color= 'red', linestyle='--')
    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel('Number of queries (log scale)', fontsize=14)
    plt.ylabel('Error (log scale)', fontsize=14)

    plt.grid(True, which="both", ls="--", alpha=0.5)

    plt.legend()
    if save_plt:
        plt.savefig("error_fit_" + title + ".png")
    if show_plt:
        plt.show()


if  __name__ == "__main__":
    value = 1
    n_qubit = 5
    mine = int_to_binary(value, n_qubit)
    twos_complement = '{:b}'.format(value).rjust(n_qubit, '0')
    twos_complement= [int(bit) for bit in list(twos_complement)]
    print(twos_complement)
    print(type(twos_complement))
    print(type(mine))