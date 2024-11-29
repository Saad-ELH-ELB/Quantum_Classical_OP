from AmpEst_Can import *
from AmpEst_ML import *
from Classical_MC import MC_uniform_mean

"""
Uniform distribution study case
"""
stud_case = "U"
n_qubit = 4
N_shots = 100
M = 7
x_min = 0
x_max = 0.5
exact_mean = (x_max+x_min)/2

N = 2 ** n_qubit
X = np.linspace(x_min, x_max, num=N)
rot_angles = compute_rotation_angles(X, stud_case)
# Maximum Likelihood amplitude estimation
title = "U_ML_AmpEst_Exact"
mean_candidate_list, N_queries_list, times = ML_AmpEst(n_qubit,N_shots, M ,stud_case, rot_angles, exact=True)
print(mean_candidate_list)
error_list = [np.abs(mean_candidate-exact_mean) for mean_candidate in  mean_candidate_list]

plot_error_fit(N_queries_list, error_list, title, show_plt = True, save_plt = False, Q_or_C ='Q')

# Canonical amplitude estimation
title = "U_Can_AmpEst_Exact"
error_list = []
N_queries_list = []
for m in range(1,M):
    m_steps = m
    time, phase = Can_AmpEst(n_qubit,N_shots, m_steps ,stud_case, rot_angles, exact=True)
    N_queries = Compute_n_queries_Can(m_steps, N_shots)
    mean = (1 - math.cos(phase))/2
    error_list.append(np.abs(mean-exact_mean))
    N_queries_list.append(N_queries)

plot_error_fit(N_queries_list, error_list, title, show_plt = True, save_plt = False, Q_or_C ='Q')

# Classical MC
title = "U_MC"
MC_error_list = []
MC_N_queries_list = []
for k in range(8):
    num_samples = 10**k
    MC_mean= MC_uniform_mean(x_min, x_max, num_samples)
    MC_error_list.append(np.abs(MC_mean-exact_mean))
    MC_N_queries_list.append(num_samples)

plot_error_fit(MC_N_queries_list, MC_error_list, title, show_plt = True, save_plt = False, Q_or_C ='C')
