from scipy.special import gamma
from prep_State_Dist_loading import *
from AmpEst_ML import *
from Classical_MC import european_call_FGMC



"""
Option pricing under Levy driven OU  model study case
"""
stud_case = "OP"
n_qubit = 6
N_shots = 100
M = 6

delta_times = np.ones(1)
tol = 5 * 10 ** (-4)
# Process parameters
x0 = 0
sigma = 0.201
k = 0.256
alpha = 0.7
# Parameters
beta = (1 - alpha) / k
c = 1 / gamma(1 - alpha) * beta ** (1 - alpha)
b = 1
a = np.exp(-b * delta_times)

flag = 4
model_params = {'x0': x0, 'b': b, 'beta': beta, 'c': c, 'alpha': alpha, 'sigma': sigma, 'k': k}

strike = 100
F0 = 110
B_t0 = 0.95
op_params = [strike, F0]

probabilities, X = compute_probabilities_levy(delta_times[-1], model_params, tol, n_qubit, flag)
rot_angles = compute_rotation_angles(X, stud_case, op_params)
x_max = X[-1]
v_max = payoff_function(x_max, strike, F0)


# Maximum Likelihood amplitude estimation
title = "LevyOU_OP_ML_AmpEst_Exact"
mean_candidate_list, N_queries_list, times = ML_AmpEst(n_qubit,N_shots, M ,stud_case, probabilities, rot_angles, exact=True)
error_list = [np.abs(mean_candidate_list[i] - mean_candidate_list[i-1]) for i in range(1, len(mean_candidate_list)) ]

plot_error_fit(N_queries_list[1:], error_list, title, show_plt = True, save_plt = False, Q_or_C ='Q')
#

# Classical MC
title = "LevyOU_OP_MC"
MC_price_list = []
MC_N_queries_list = []
for k in range(8):
    num_samples = 10**k
    MC_price=  european_call_FGMC(F0, strike, B_t0, model_params, flag, delta_times[-1], num_samples)
    MC_price_list.append(MC_price)
    MC_N_queries_list.append(num_samples)

MC_error_list = [np.abs(MC_price_list[i] - MC_price_list[i-1]) for i in range(1, len(MC_price_list)) ]
plot_error_fit(MC_N_queries_list[1:], MC_error_list, title, show_plt = True, save_plt = False, Q_or_C ='C')

