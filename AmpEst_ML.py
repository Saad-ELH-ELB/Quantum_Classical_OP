import time
import math
from qiskit import QuantumCircuit, transpile
from qiskit_aer import Aer
from prep_Grover_operator import *
from ZZ_utilities import *
from prep_State_Payoff_encoding import *
from scipy.optimize import brute
import pandas as pd

def ML_AmpEst(n_qubit, N_shots, M, stud_case, *F_args, exact=True):
    """
        Perform Canonical Amplitude Estimation (Can_AmpEst).
            m_steps (int): Number of qubits for the phase estimation process.
        :param n_qubit: (int) Number of qubits used to encode the distribution.
        :param N_shots: (int) Number of shots (simulations or experiments) to run.
        :param M: (int): Number of steps.
        :param stud_case: See function R
        :param F_args: Additional arguments passed to the gate creation functions.
        :param exact: (bool)  Whether to use the exact version of the payoff encoding (default is True).
        :return:
        mean_candidate_list: Contains the results of the amplitude estimation.
        N_queries_list: Tracks how many Grover iterations (queries) were performed in each circuit execution.
        times: Stores the execution time for each circuit run
        """
    h_goodmeas_list = []
    m_grover_list= []
    N_shots_list = []
    times = []

    for k in range(-1,M):
        start = time.time()
        if k == -1 :
            m_grover = 0
        else :
            m_grover = 2 ** k

        qc = QuantumCircuit(n_qubit + 2 , 1)

        F_gate = create_F_gate(n_qubit, stud_case, *F_args, exact=exact)
        Q_gate = create_Q_gate(n_qubit, stud_case, *F_args, exact=exact)

        qc.append(F_gate, qc.qubits[0:n_qubit+2])

        for j in range(m_grover):
            qc.append(Q_gate, qc.qubits[0:n_qubit + 2])

        # Measure the state of the first qubit and store the result in the classical bit.
        qc.measure(qc.qubits[0], qc.clbits[0])

        backend = Aer.get_backend("qasm_simulator")
        new_circuit = transpile(qc, backend)
        job = backend.run(new_circuit, shots=N_shots)
        counts = job.result().get_counts()

        # Extract the number of successful "1" outcomes from the measurement (successful amplitude estimate).
        h_goodmeas = counts.get('1', 0)

        # Store results for later analysis.
        h_goodmeas_list.append(h_goodmeas)
        m_grover_list.append(m_grover)
        N_shots_list.append(N_shots)
        end = time.time()
        times.append(end - start)

    # Compute the mean candidate and number of queries based on the measurements.
    mean_candidate_list, N_queries_list = compute_means(h_goodmeas_list, m_grover_list, N_shots_list)
    return mean_candidate_list, N_queries_list, times

def compute_means(h_goodmeas_list, m_grover_list, N_shots_list):
    """
      This function calculates the amplitude estimates (mu_candidate_list) and the total number of
      oracle queries used (N_queries_list) during a multi-level amplitude estimation process.

      Parameters:
      - h_goodmeas_list: List of the number of successful "good" measurements for each Grover iteration.
      - m_grover_list: List of Grover iteration counts.
      - N_shots_list: List of shot counts for each level of measurement.

      Returns:
      - mu_candidate_list: List of estimated probabilities (amplitudes).
      - N_queries_list: List of the total number of oracle queries for each level.
      """
    confidence_level = 5 # confidence level to determine the search range

    mu_candidate_list = []
    N_queries_list = []

    range_min = 0  # Initial lower bound of the search range for amplitude estimation.
    range_max = 1  # Initial upper bound of the search range for amplitude estimation.

    # Iterate over the levels of Grover iterations.
    for idx_grover in range(len(m_grover_list)):

        # Define the likelihood function to maximize.
        def likelihood(mu):
            """
           Compute the likelihood of a given amplitude `mu` based on observed measurements.
            """
            theta = np.arcsin(np.sqrt(mu))  # Compute the angle corresponding to amplitude `mu`.
            ret = -1  # Initialize likelihood (negative to fit optimization format).
            for n in range(idx_grover + 1):  # Loop through Grover levels up to current index.
                h_goodmeas = h_goodmeas_list[n]  # Successful measurements at level `n`.
                N_shots = N_shots_list[n]  # Number of shots at level `n`.
                m_grover = m_grover_list[n]  # Grover iteration count at level `n`.
                arg = (2 * m_grover + 1) * theta  # Argument for sine and cosine.
                # Update likelihood using measurement probabilities at level `n`.
                ret *= math.sin(arg) ** (2 * h_goodmeas) * math.cos(arg) ** (2 * (N_shots - h_goodmeas))
            return ret

        # Define the search range for the optimization process.
        search_range = (range_min, range_max)

        # Use brute-force search to find the amplitude that maximizes the likelihood function.
        search_result = brute(likelihood, [search_range])

        mu_candidate = search_result[0]
        mu_candidate_list.append(mu_candidate)
        # Compute the total number of oracle queries used up to this level.
        N_queries = Compute_n_queries_ML(idx_grover, N_shots_list, m_grover_list)
        N_queries_list.append(N_queries)

        # Compute the error in the amplitude estimate.
        mu_error = Compute_error(idx_grover, N_shots_list, mu_candidate, m_grover_list)

        # Update the search range based on the confidence level.
        range_max = min(mu_candidate+confidence_level*mu_error,1)
        range_min = max(mu_candidate-confidence_level*mu_error,0)

    return mu_candidate_list, N_queries_list

def Compute_error(M, N_shots_list, mu, m_grover_list):
    """
       Compute the standard error of the amplitude estimate `mu` based on the
       Fisher information accumulated over multiple Grover iterations.

       Parameters:
       - M: The index of the current Grover iteration level.
       - N_shots_list: List of the number of measurement shots for each level.
       - mu: The current amplitude estimate.
       - m_grover_list: List of Grover iteration counts (number of Grover operator applications) for each level.

       Returns:
       - The standard error of the amplitude estimate.
       """

    fisher_info = 0  # Initialize Fisher information.

    # Accumulate Fisher information across all Grover levels up to the current level.
    for k in range(M + 1):
        Nk = N_shots_list[k]  # Number of shots at level `k`.
        mk = m_grover_list[k]  # Grover iteration count at level `k`.
        fisher_info += Nk / (mu * (1 - mu)) * (2 * mk + 1) ** 2

    # Standard error is the inverse square root of Fisher information.
    return np.sqrt(1 / fisher_info)

def Compute_n_queries_ML(M, N_shots_list, m_grover_list):
    N_queries = 0  # Initialize total query count.

    # Sum the number of queries for each Grover level up to `M`.
    for k in range(M + 1):
        Nk = N_shots_list[k]  # Number of shots at level `k`.
        mk = m_grover_list[k]  # Grover iteration count at level `k`.
        # Each shot at level `k` uses (2 * mk + 1) oracle queries.
        N_queries += Nk * (2 * mk + 1)
    return N_queries









