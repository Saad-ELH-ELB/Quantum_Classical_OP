from prep_State_Dist_loading import *
from AmpEst_ML import *
from Classical_MC import european_call_black_MC

"""
Option pricing under Black model study case
"""
stud_case = "OP"
n_qubit = 6
N_shots = 100
M = 6

sigma = 0.2
model_params = {"sigma": sigma}
maturity = 1
strike = 100
F0 = 110
B_t0 = 0.95
tol = 5
op_params = [strike, F0]
probabilities, X = compute_probabilities_black(maturity, model_params, tol, n_qubit)
rot_angles = compute_rotation_angles(X, stud_case, op_params)
x_max = X[-1]
v_max = payoff_function(x_max, strike, F0)
exact_price = european_call_bm(F0, strike, B_t0, sigma, maturity)

# Maximum Likelihood amplitude estimation
# title = "Black_OP_ML_AmpEst_Exact"
# mean_candidate_list, N_queries_list, times = ML_AmpEst(n_qubit,N_shots, M ,stud_case, probabilities, rot_angles, exact=True)
# print(mean_candidate_list)
# error_list = [np.abs(mean_candidate * B_t0 * v_max - exact_price) for mean_candidate in mean_candidate_list]
#
# plot_error_fit(N_queries_list, error_list, title, show_plt = True, save_plt = False, Q_or_C ='Q')


# Classical MC
title = "Black_OP_MC"
MC_error_list = []
MC_N_queries_list = []
for k in range(8):
    num_samples = 10**k
    MC_price= european_call_black_MC(F0, strike, B_t0, maturity, sigma, num_samples)
    MC_error_list.append(np.abs(MC_price-exact_price))
    MC_N_queries_list.append(num_samples)

plot_error_fit(MC_N_queries_list, MC_error_list, title, show_plt = True, save_plt = False, Q_or_C ='C')
