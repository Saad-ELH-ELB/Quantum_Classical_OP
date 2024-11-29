import numpy as np
from qiskit.circuit.library import *
from qiskit import QuantumCircuit, transpile
from qiskit_aer import Aer
from prep_State_Payoff_encoding import R, inv_R, compute_rotation_angles
from prep_State_Dist_loading import create_A_gate, create_inv_A_gate

def create_controlled_Q_gate(n_qubit, stud_case, *F_args, exact=True):
    # This function adds a control qubit to the gate Q
    # See the parameters in the function Q
    Q_gate = create_Q_gate(n_qubit, stud_case, *F_args, exact=exact)
    CQ_gate = Q_gate.control(1)
    return (CQ_gate)

def create_Q_gate(n_qubit, stud_case, *F_args, exact=True):
    """
        Creates the Q gate for the grover operator.

        Parameters:
            n_qubit (int): Number of qubits for the distribution loading.
            stud_case (str): Study case type, e.g., "OP" (Option Pricing).
            *F_args: Additional arguments required by the Q function.
            exact (bool): Indicates whether the exact version of the encoding is used (default is True).

        Returns:
            QuantumGate: A quantum gate labeled "Q" that can be appended to a quantum circuit.
        """

    #For the rest see the comment in create_F_gate
    if stud_case == "OP" and exact == False:
        qc = QuantumCircuit(1 + n_qubit + 2 + n_qubit)

    else :
        qc = QuantumCircuit(n_qubit + 2)
    Q(qc, n_qubit, stud_case, *F_args, exact=exact)
    gate_Q = qc.to_gate(label="Q")
    return(gate_Q)

def Q(qc, n_qubit, stud_case, *F_args, exact=True):
    """
    :param qc: quantum circuit
    :param n_qubit: int number of qubits used to load the distribution
    :param stud_case: str "U": computation of the mean of a uniform distribution,
                        "OP": Option pricing study case
    :param F_args: See functions Grover_operator.F and payoff_encoding.R
    :param exact: (bool) Whether to use the exact version of the payoff encoding (default is True).

    :return:  None: The function applies the grover operator Q = - F Z inv_F V to the quantum circuit qc in place

    :remark: the minus sign is included in the creation of Z
    """


    # r_X is the state created after the two state preparation steps
    r_X = qc.qubits[:n_qubit+1]
    qc.append(V_gate(n_qubit), r_X)
    inv_F(qc, n_qubit, stud_case, *F_args, exact=exact)
    qc.append(minus_Z_gate(n_qubit), r_X)
    F(qc, n_qubit, stud_case, *F_args, exact=exact)


def create_F_gate(n_qubit, stud_case, *F_args, exact=True):
    """
    Creates the gate F for quantum amplitude estimation or encoding.

    Parameters:
        n_qubit (int): Number of qubits for the Distribution Loading.
        stud_case (str): Study case type, e.g., "OP" (Option Pricing).
        *F_args: Additional arguments required by the F function.
        exact (bool): Indicates whether the exact version of the encoding is used (default is True).

    Returns:
        QuantumGate: A quantum gate labeled "F" that can be appended to a quantum circuit.
    """

    if stud_case == "OP" and exact == False:
        # Option pricing with approximate encoding requires additional qubits:
        # - `n_qubit + 1` for comparison in payoff encoding(see the function Payoff_Encoding.compare)
        qc = QuantumCircuit(1 + n_qubit + 2 + n_qubit)
    else :
        qc = QuantumCircuit(n_qubit + 2)

    # This modifies the circuit `qc` in place based on the study case and arguments
    F(qc, n_qubit, stud_case, *F_args, exact=exact)

    # Convert the modified circuit into a quantum gate
    gate_F = qc.to_gate(label="F")

    return (gate_F)

def F(qc, n_qubit, stud_case, *F_args, exact=True):
    """
    :param qc: quantum circuit
    :param n_qubit: (int) number of qubits used to load the distribution
    :param stud_case: (str) "U": computation of the mean of a uniform distribution,
                        "OP": Option pricing study case
    :param F_args: See function payoff_encoding.R
    :param exact:  (bool) Whether to use the exact version of the payoff encoding (default is True).

    :return: None: The function applies the grover operator F to the quantum circuit qc in place
    """

    # In the case of option pricing,
    # we need the discretized pdf as input in order
    # to apply the Distribution loading procedure
    # In this case, F_args includes probabilities and R_arg as defined in function R
    if stud_case == "OP":
        probabilities = F_args[0]
        R_args = F_args[1:]
        A_gate = create_A_gate(n_qubit, stud_case, probabilities)
    else:
        R_args = F_args
        A_gate = create_A_gate(n_qubit, stud_case)

    qc.append(A_gate, qc.qubits[1:n_qubit + 1])
    R(qc, n_qubit, stud_case, *R_args, exact=exact)

def inv_F(qc, n_qubit, stud_case, *F_args, exact=True):
    # create the inv_F operator
    if stud_case == "OP":
        probabilities = F_args[0]
        R_args = F_args[1:]
        inv_A_gate = create_inv_A_gate(n_qubit, stud_case, probabilities)
    else:
        R_args = F_args
        inv_A_gate = create_inv_A_gate(n_qubit, stud_case)

    inv_R(qc, n_qubit, stud_case, *R_args, exact=exact)
    qc.append(inv_A_gate, qc.qubits[1:n_qubit + 1])


def V_gate(n_qubit):
    matrix = V_mat(n_qubit)
    gate = UnitaryGate(matrix, label="V")
    return gate
def minus_Z_gate(n_qubit):
    matrix = minus_Z_mat(n_qubit)
    gate = UnitaryGate(matrix, label="Z")
    return gate

def minus_Z_mat(n_qubit):
    zero_n = np.zeros(2 ** (n_qubit + 1))
    zero_n[0] = 1
    result = I_mat(2 ** (n_qubit + 1)) - 2 * np.outer(zero_n, zero_n)
    return(-result) # Negation needed since Q = - F Z inv_F V
def V_mat(n_qubit):
    _1_ = np.array([0, 1])
    tensorproduct = np.kron(I_mat(2 ** n_qubit), np.outer(_1_, _1_))
    result = I_mat(2 ** (n_qubit + 1)) - 2 * tensorproduct
    return(result)
def I_mat(n):
    return np.eye(n)


if  __name__ == "__main__":
    # This code is important to check that the state encode the needed information,
    # we can get an approximation of the expected value and the convergence rate is comparable
    # to the one in classical MC

    n_qubit = 4
    N_shots = 10**7
    stud_case = "U"
    x_min = 0
    x_max = 0.5
    X = np.linspace(x_min, x_max, num=2 ** n_qubit)
    rot_angles = compute_rotation_angles(X, stud_case)

    qc = QuantumCircuit(n_qubit + 2, 1)
    F_gate = create_F_gate(n_qubit, stud_case, rot_angles, exact=True)
    qc.append(F_gate, qc.qubits[0:n_qubit + 2])
    qc.measure(qc.qubits[0], qc.clbits[0])
    backend = Aer.get_backend("qasm_simulator")
    new_circuit = transpile(qc, backend)
    job = backend.run(new_circuit, shots=N_shots)
    counts = job.result().get_counts()
    h_goodmeas = counts.get('1', 0)
    print(h_goodmeas/N_shots, (x_max+x_min)/2)

