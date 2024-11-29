import numpy as np
from qiskit.circuit.library import RYGate
from scipy.interpolate import interp1d
from qiskit import QuantumCircuit
import math
from ZZ_utilities import *
from prep_pdfs import compute_pdf_OUlevy, compute_pdf_lognormal

def create_A_gate(n_qubit, stud_case, probabilities=None):
    qc= QuantumCircuit(n_qubit)
    if stud_case=="U":
        # Apply Hadamard gates to all qubits
        qc.h(qc.qubits[0:])
    elif stud_case == "OP":
        if probabilities is None:
            raise ValueError(f"`probabilities` must be provided for the study case Option pricing : A")
        # Load the discretized probabilities into the quantum circuit using load_dist : the distribution loading procedure
        load_dist(qc, probabilities, n_qubit)
    else :
        ValueError(f"Study case not defined: in A")
    gate_A = qc.to_gate(label="A")
    return gate_A

def create_inv_A_gate(n_qubit, stud_case, probabilities=None):
    qc = QuantumCircuit(n_qubit)
    if stud_case == "U":
        qc.h(qc.qubits[0:])
    elif stud_case == "OP":
        if probabilities is None:
            raise ValueError(f"`probabilities` must be provided for the study case Option pricing : Inv_A")
        inv_load_dist(qc, probabilities, n_qubit)
    else:
        ValueError(f"Study case not defined: in Inv_A")

    gate_inv_A = qc.to_gate(label="inv_A")
    return gate_inv_A

def load_dist(myqc, probabilities, n_qubit):
    # This function loads a discrete probability distribution into a quantum state.
    # It prepares the quantum state with amplitudes corresponding to the given probabilities.

    # Normalize the probabilities to ensure they sum to 1.
    total = sum(probabilities)
    probabilities = [p / total for p in probabilities]

    r_i = myqc.qubits[0:n_qubit] # The qubits that will store the quantum state.

    # Initialization : create 2 regions discretization
    m = 1
    theta = 2 * np.arccos(math.sqrt(sum(probabilities[0:2 ** (n_qubit - m)])))
    myqc.ry(theta, r_i[0])


    while m < n_qubit:
        # discretizes the distribution from 2^m regions to 2^(m+1) regions.
        for k in range(2 ** m):
            binary_string = int_to_binary(k, m)
            zeros_indices = get_zero_indices(binary_string)
            theta_k = get_theta(k, m, probabilities, n_qubit)

            if len(zeros_indices) == 0:
                myqc.append(RYGate(theta_k).control(m), [r_i[i] for i in range(m + 1)])
            else:
                myqc.x([r_i[k] for k in zeros_indices])
                myqc.append(RYGate(theta_k).control(m), [r_i[i] for i in range(m + 1)])
                myqc.x([r_i[k] for k in zeros_indices])
        m += 1

def inv_load_dist(myqc, probabilities, n_qubit):
    total = sum(probabilities)
    probabilities = [p / total for p in probabilities]

    r_i = myqc.qubits[0:n_qubit]

    m = n_qubit - 1
    while m > 0:
        for k in range(2 ** m):
            binary_string = int_to_binary(k, m)
            zeros_indices = get_zero_indices(binary_string)
            theta_k = get_theta(k, m, probabilities, n_qubit)

            if len(zeros_indices) == 0:
                myqc.append(RYGate(-theta_k).control(m), [r_i[i] for i in range(m + 1)])
            else:
                myqc.x([r_i[k] for k in zeros_indices])
                myqc.append(RYGate(-theta_k).control(m), [r_i[i] for i in range(m + 1)])
                myqc.x([r_i[k] for k in zeros_indices])

        m -= 1

    theta = 2 * np.arccos(math.sqrt(sum(probabilities[0: 2 ** (n_qubit - 1)])))
    myqc.ry(-theta, r_i[0])

def compute_probabilities_black(maturity, model_params, tol, n_qubit):
    sigma = model_params["sigma"]
    # Calculate N, the number of points for the discretization based on the number of qubits.
    N = 2 ** n_qubit

    # Compute the maximum and minimum values of the distribution using the Black-Scholes formula.
    x_max = math.exp( - sigma ** 2 / 2 * maturity + sigma * np.sqrt(maturity)* tol)
    x_min = math.exp(- sigma ** 2 / 2 * maturity - sigma * np.sqrt(maturity) * tol)
    X = np.linspace(x_min, x_max, num=N)

    # Define the mean (mu) and standard deviation (std_div) for the log-normal distribution.
    mu = - 1 / 2 * sigma ** 2 * maturity
    std_div = sigma * np.sqrt(maturity)

    # Compute the log-normal probability density function (pdf) for each X value.
    probabilities = [compute_pdf_lognormal(x, mu, std_div) for x in X]
    return(probabilities,  X)

def compute_probabilities_levy(maturity, model_params, tol, n_qubit, flag):
    N = 2 ** n_qubit
    # Compute the PDF and X values for the Ornstein-Uhlenbeck (OU) Levy process.
    pdf, x = compute_pdf_OUlevy(model_params, maturity, flag)

    # Filter the X values where the PDF is greater than or equal to the tolerance (tol).
    x_used = x[(pdf >= tol)]

    # Create a range of X values between the minimum and maximum of the used X values with N points.
    X = np.linspace(x_used[0], x_used[-1], num=N)

    # Interpolate the probability density function (pdf) to match the newly discretized X values.
    interp_method = 'cubic'
    probabilities = interp1d(x, pdf, kind=interp_method)(X)

    return(probabilities,  X)

def get_theta(k, m, P, n):
    if m > n:
        print("m is bigger than n_qubits")
        return
    # p(k, m)
    p_k = sum(P[k * 2 ** (n - m): (k + 1) * 2 ** (n - m)])
    # p(2k, m+1)
    p_2k = sum(P[2 * k * 2 ** (n - m - 1): (2 * k + 1) * 2 ** (n - m - 1)])
    # f(k,m)
    f_k_m = p_2k / p_k
    theta = 2 * np.arccos(math.sqrt(f_k_m))
    return theta
