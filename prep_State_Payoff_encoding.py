import math
import numpy as np
from ZZ_utilities import *
from qiskit.circuit.library.boolean_logic import OR

def R(qc, n_qubit ,stud_case, *R_args, exact=True):
    """

    :param qc: Quantum circuit
    :param n_qubit: (int) number of qubit used to load the distribution
    :param stud_case: (str) "U": computation of the mean of a uniform distribution,
                        "OP": Option pricing study case
    :param R_args: A variable number of positional arguments depends on the number of parameters
     needed for exact_encoding and approx_encoding
    :param exact: (bool) Whether to use the exact version of the payoff encoding (default is True).
    :return: Apply the operator R to the quantum circuit qc
    """
    if exact :
        rot_angles = R_args[0]
        exact_encoding(qc, n_qubit, rot_angles)

    else:
        approx_encoding(qc, n_qubit, stud_case, *R_args)

def inv_R(qc, n_qubit, stud_case , *R_args, exact=True):
    """
    :return: Apply the operator inv_R to the quantum circuit qc
    """
    if exact:
        rot_angles = R_args[0]
        inv_exact_encoding(qc, n_qubit, rot_angles)
    else:
        inv_approx_encoding(qc, n_qubit, stud_case, *R_args)

# Exact rotation
def exact_encoding(qc, n_qubit, rot_angles):
    """
    :param qc: Quantum circuit
    :param n_qubit: (int) number of qubit used for distribution loading
    :param rot_angles: (float[]) rotation angles for all j in [0, 2**(n_qubit)-1]

    :return: Apply the exact payoff encoding to the quantum circuit qc
    """
    # the register where the distribution is loaded
    r_i = qc.qubits[1:n_qubit + 1]
    # the ancilla qubit where we aim encoding the payoff
    r_v = qc.qubits[0]
    #conditional qubit used as control for the rotation
    r_cond = qc.qubits[n_qubit + 1]

    # Aplly 2**(n_qubit) controlled rotations but only one will be executed
    for j in range(2**(n_qubit)):
        j_bin = int_to_binary(j, n_qubit)
        zeros_indices = get_zero_indices(j_bin)

        if len(zeros_indices) != 0:
            for h in zeros_indices:
                qc.x(r_i[h])

        theta_j = rot_angles[j]

        qc.mcx(control_qubits=r_i, target_qubit = r_cond)

        qc.cry(theta_j, r_cond, r_v)

        # Restore the conditional qubit to its initial state to use it in the next step
        qc.mcx(control_qubits=r_i, target_qubit = r_cond)

        # Restore the i_register to its initial state
        if len(zeros_indices) != 0:
            for h in zeros_indices:
                qc.x(r_i[h])

def inv_exact_encoding(qc, n_qubit, rot_angles):
    # the inverse of the exact encoding
    r_i = qc.qubits[1:n_qubit + 1]
    r_v = qc.qubits[0]
    r_cond = qc.qubits[n_qubit + 1]

    for i in range(2 ** (n_qubit)):
        i_bin = int_to_binary(i, n_qubit)
        zeros_indices = get_zero_indices(i_bin)

        if len(zeros_indices) != 0:
            for h in zeros_indices:
                qc.x(r_i[h])

        theta_x = rot_angles[i]

        qc.mcx(control_qubits=r_i, target_qubit=r_cond)

        qc.cry(-theta_x, r_cond, r_v)

        qc.mcx(control_qubits=r_i, target_qubit=r_cond)

        if len(zeros_indices) != 0:
            for h in zeros_indices:
                qc.x(r_i[h])

def compute_rotation_angles(X, stud_case, op_params=None):
    """
    :param X: the possible values for the random variable X = [x_min ,...., x_max]
    :param stud_case: (str) "U": computation of the mean of a uniform distribution,
                        "OP": Option pricing study case
    :param op_params: defautl is None, in the case of Option pricing, op_params=[strike, F0]
    :return: (float[]) rotation angles corresponding to all the x_j in [x_min, ..., x_max]
    """
    thetas = []
    if stud_case == "U":
        N = len(X)
        for j in range(N):
            x = X[j]
            # The payoff in the case of Uniform law is x.
            theta_x = 2*math.asin(math.sqrt(x))
            thetas.append(theta_x)
    elif stud_case == "OP":
        if op_params is None:
            raise ValueError(f"`op_params` must be provided for the study case Option pricing : compute_rotation_angles")
        strike, F0 = op_params
        N = len(X)
        x_max = X[-1]
        v_max = payoff_function(x_max, strike, F0)
        for j in range(N):
            x = X[j]
            v_x = payoff_function(x, strike, F0)
            # we devide by v_max, because the amplitude should be <1
            theta_x = 2*math.asin(math.sqrt(v_x/v_max))
            thetas.append(theta_x)
    else:
        ValueError(f"Study case not defined: in compute_rotation_angles")
    return thetas

# Approximate rotation
def approx_encoding(qc, n_qubit, stud_case, x_min, x_max, approx_param, op_params= None):
    """
        Approximate encoding of the payoff.
        Parameters:
            qc (QuantumCircuit): Quantum circuit to apply the encoding.
            n_qubit (int): Number of qubits used for distribution loading.
            stud_case (str): Study case type, can be "U" (uniform) or "OP" (option pricing).
            x_min (float): Minimum value of the range for X.
            x_max (float): Maximum value of the range for X.
            approx_param (float): Approximation parameter for the approximate payoff rotation.
            op_params (optional): Parameters specific to "OP" study case (strike, F0). Required if stud_case is "OP".
        Returns:
            None: The function modifies the quantum circuit `qc` in place.
        """
    r_v = qc.qubits[0] # Ancilla qubit (target qubit for rotation)
    r_i = qc.qubits[1:n_qubit + 1] # Distribution register qubits (control qubits for rotations)
    if stud_case == "U":
            for k in range(n_qubit):
                qc.cry(theta=2 * approx_param * 2 ** (-(k+1)) * (x_max - x_min), control_qubit=r_i[k], target_qubit= r_v)
            qc.ry(math.pi / 2, r_v)
            qc.ry(2 * x_min * approx_param, r_v)
    elif stud_case == "OP":
        if op_params is None:
            # Option pricing requires additional parameters (strike, F0)
            raise ValueError(
                f"`op_params` must be provided for the study case Option pricing : Approx_encoding")
        strike, F0 = op_params
        r_comp = qc.qubits[n_qubit + 1]
        r_cond = qc.qubits[n_qubit + 2]

        t = (2 ** n_qubit - 1) / (x_max - x_min) * (strike / F0 - x_min)

        # Perform comparison (sets r_comp based on the condition)
        compare(qc, t, n_qubit)

        for k in range(n_qubit):
            qc.ccx(control_qubit1=r_comp, control_qubit2=r_i[k], target_qubit=r_cond)
            qc.cry(theta=approx_param * 2 ** (n_qubit - 1 - k) * 4 * F0 * (x_max - x_min) / (
                    (F0 * x_max - strike) * (2 ** n_qubit - 1)), control_qubit=r_cond, target_qubit=r_v)

            # Uncompute the CCX gate
            qc.ccx(control_qubit1=r_comp, control_qubit2=r_i[k], target_qubit=r_cond)

        qc.cry(theta=-approx_param * 2 * 2 * (strike - F0 * x_min) / (F0 * x_max - strike), control_qubit=r_comp,
               target_qubit=r_v)
        qc.ry(-2 * approx_param, r_v)
        qc.ry(math.pi / 2, r_v)

        # Undo the comparison operation
        inv_compare(qc, t, n_qubit)
    else:
        ValueError(f"Study case not defined: in Approx_encoding")

def inv_approx_encoding(qc, n_qubit, stud_case, x_min, x_max, approx_param, op_params= None):
    # This function implements the inverse of the approx_encoding function.
    r_v = qc.qubits[0]
    r_i = qc.qubits[1:n_qubit + 1]
    if stud_case == "U":
        for k in range(n_qubit):
            qc.cry(theta=- 2 * approx_param * 2 ** (-(k+1)) * (x_max - x_min), control_qubit=r_i[k],
                   target_qubit=r_v)
        qc.ry(- math.pi / 2, r_v)
        qc.ry(-2 * x_min * approx_param, r_v)
    elif stud_case == "OP":
        if op_params is None:
            raise ValueError(
                f"`op_params` must be provided for the study case Option pricing : Approx_encoding")
        strike, F0 = op_params
        r_comp = qc.qubits[n_qubit + 1]
        r_cond = qc.qubits[n_qubit + 2]

        t = (2 ** n_qubit - 1) / (x_max - x_min) * (strike / F0 - x_min)
        compare(qc, t, n_qubit)

        for k in range(n_qubit):
            qc.ccx(control_qubit1=r_comp, control_qubit2=r_i[k], target_qubit=r_cond)
            qc.cry(theta=-approx_param * 2 ** (n_qubit - 1 - k) * 4 * F0 * (x_max - x_min) / (
                        (F0 * x_max - strike) * (2 ** n_qubit - 1)), control_qubit=r_cond, target_qubit=r_v)
            qc.ccx(control_qubit1=r_comp, control_qubit2=r_i[k], target_qubit=r_cond)

        qc.cry(theta=approx_param * 2 * 2 * (strike - F0 * x_min) / (F0 * x_max - strike), control_qubit=r_comp,
               target_qubit=r_v)
        qc.ry(2 * approx_param, r_v)
        qc.ry(-math.pi / 2, r_v)

        inv_compare(qc, t, n_qubit)
    else:
        ValueError(f"Study case not defined: in Inv_approx_encoding")

def compare(qc, t, n_qubit):
    # This function implements a comparison of the input value with a threshold `t`
    # using quantum arithmetic and sets a comparison qubit (`r_comp`) accordingly.

    r_i = qc.qubits[1:n_qubit + 1]  # Input qubits to be compared.
    r_a = qc.qubits[-n_qubit:]      # Ancilla qubits for intermediate results.
    r_comp = qc.qubits[n_qubit + 1] # Comparison result qubit.

    # Convert the threshold `t` into its two's complement representation for comparison.
    two_comp = get_twos_complement(t, n_qubit)

    for k in range(n_qubit):
        if k == 0:
            if two_comp[k] == 1:
                qc.cx(r_i[k], r_a[k])
        elif k < n_qubit - 1:
            if two_comp[k] == 1:
                qc.compose(OR(2), [r_i[k], r_a[k - 1], r_a[k]],
                           inplace=True)
            else:
                qc.ccx(r_i[k], r_a[k - 1], r_a[k])
        else:
            if two_comp[k] == 1:
                qc.compose(OR(2), [r_i[k], r_a[k - 1], r_comp],
                           inplace=True)
            else:
                qc.ccx(r_i[k], r_a[k - 1], r_comp)

def inv_compare(qc, t, n_qubit):
    r_i = qc.qubits[1:n_qubit + 1]
    r_a = qc.qubits[-n_qubit:]
    r_comp = qc.qubits[n_qubit + 1]

    two_comp = get_twos_complement(t, n_qubit)

    for k in reversed(range(n_qubit)):
        if k == 0:
            if two_comp[k] == 1:
                qc.cx(r_i[k], r_a[k])
        elif k < n_qubit - 1:
            if two_comp[k] == 1:
                qc.compose(OR(2), [r_i[k], r_a[k - 1], r_a[k]],
                           inplace=True)
            else:
                qc.ccx(r_i[k], r_a[k - 1], r_a[k])
        else:
            if two_comp[k] == 1:
                qc.compose(OR(2), [r_i[k], r_a[k - 1], r_comp],
                           inplace=True)
            else:
                qc.ccx(r_i[k], r_a[k - 1], r_comp)



