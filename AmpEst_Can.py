import math
import time
from qiskit import QuantumCircuit, transpile
from qiskit.circuit.library import QFT
from qiskit_aer import Aer
from ZZ_utilities import *
from prep_Grover_operator import create_controlled_Q_gate, create_F_gate


def Can_AmpEst(n_qubit,N_shots, m_steps ,stud_case, *F_args, exact=True):
    """
    Perform Canonical Amplitude Estimation (Can_AmpEst).
        m_steps (int): Number of qubits for the phase estimation process.
    :param n_qubit: (int) Number of qubits used to encode the distribution.
    :param N_shots: (int) Number of shots (simulations or experiments) to run.
    :param m_steps: (int): Number of qubits for the phase estimation process.
    :param stud_case: See function R
    :param F_args: Additional arguments passed to the gate creation functions.
    :param exact: (bool)  Whether to use the exact version of the payoff encoding (default is True).
    :return:
    """

    # Determine the size of the quantum circuit based on the study case
    if stud_case == "OP" and exact == False:
        qc = QuantumCircuit(1 + n_qubit + 2 + n_qubit + m_steps, m_steps)
    else:
        qc = QuantumCircuit(n_qubit + 2 + m_steps, m_steps)
    start = time.time()

    F_gate = create_F_gate(n_qubit, stud_case, *F_args, exact=exact)
    CQ_gate = create_controlled_Q_gate(n_qubit, stud_case, *F_args, exact=exact)

    # Apply the F gate to all qubits except the last `m_steps` phase estimation qubits
    qc.append(F_gate, qc.qubits[0:-m_steps])
    # Apply Hadamard gates to the last `m_steps` qubits for phase estimation
    qc.h(qc.qubits[-m_steps:])

    # Iteratively apply controlled-Q gates for phase estimation
    for h in range(m_steps):
        M = 2 ** h
        for k in range(M):
            qc.append(CQ_gate, [qc.qubits[-m_steps + h]] + qc.qubits[0:-m_steps])

    # Apply the inverse Quantum Fourier Transform (QFT) to the phase estimation qubits
    qc.append(QFT(m_steps, do_swaps=True).inverse(), qc.qubits[-m_steps:])

    # Measure the phase estimation qubits
    qc.measure(qc.qubits[-m_steps::], qc.clbits[::-1])

    # Run the circuit on the QASM simulator backend
    backend = Aer.get_backend("qasm_simulator")
    new_circuit = transpile(qc, backend)
    job = backend.run(new_circuit, shots=N_shots)
    counts = job.result().get_counts()

    # Determine the phase from the results
    result = get_key_of_median_value(counts)[::-1] # Get the binary representation of the median count
    if binary_to_float(result) < 0.5:
        phase = 2 * math.pi * binary_to_float(result)
    else:
        phase = 2 * math.pi * (1 - binary_to_float(result))

    end = time.time()

    duration = end - start

    return duration, phase

def get_key_of_median_value(d):
    # Step 1: Sort the dictionary by keys
    sorted_dict = sorted(d.items())

    # Step 2: Expand the sorted dictionary into a list
    expanded_list = []
    for key, count in sorted_dict:
        expanded_list.extend([key] * count)

    # Step 3: Find the median index
    n = len(expanded_list)
    median_index = n // 2  # integer division gives the middle index

    # Step 4: Return the median key
    return expanded_list[median_index]

def Compute_n_queries_Can(m_steps, N_shots):

    # Start with queries from the initial F gate application, which is applied once per shot.
    N_queries = 1 * N_shots

    # Add queries from Grover steps for each iteration level
    for k in range(m_steps):
        N_queries += 2 * 2 ** k * N_shots
    return N_queries

