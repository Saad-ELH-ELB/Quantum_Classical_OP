# Import libraries
import numpy as np
from numpy.fft import fft
from scipy.special import hyp2f1, gamma
from scipy.interpolate import interp1d
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import math
import warnings


def log_characteristic_function(params, flag, u, t):
    """
    Log-Characteristic Function computation

    INPUT
    params:   dictionary containing the process params
    flag:     1 -> OU-TS
              2 -> TS-OU
              3 -> OU-NTS
              4 -> NTS-OU
    u:        input parameter of the characteristic function
    t:        input parameter of the characteristic function
    """

    # Helper function
    I = lambda s, alpha, beta1, beta2: -1j / (alpha * s) * ((beta2) ** (-alpha) * (beta2 - 1j * s) ** (alpha + 1) * hyp2f1(1, 1, 1 - alpha, -1j * beta2 / s) - (beta1) ** (-alpha) * (beta1 - 1j * s) ** (alpha + 1) * hyp2f1(1, 1, 1 - alpha, -1j * beta1 / s))

    if flag == 1 or flag == 2:

        # Set parameters for readability
        gamma_l = params['gamma_l']
        b = params['b']
        beta_p = params['beta_p']
        beta_n = params['beta_n']
        c_p = params['c_p']
        c_n = params['c_n']
        alpha_p = params['alpha_p']
        alpha_n = params['alpha_n']
        a = np.exp(-b*t)

        # OU-TS
        if flag == 1:
            log_char_f = 1j*u*gamma_l*(1-np.exp(-b*t))/b - c_p*beta_p**alpha_p*gamma(1-alpha_p)/(alpha_p*b)*(I(u,alpha_p,beta_p,beta_p/a) + np.log(a) + 1j*u*(1-a)*alpha_p/beta_p) - c_n*beta_n**alpha_n*gamma(1-alpha_n)/(alpha_n*b)*(I(-u,alpha_n,beta_n,beta_n/a) + np.log(a) - 1j*u*(1-a)*alpha_n/beta_n)

        # TS-OU
        else:
            f = lambda s,c,beta,alpha : c * gamma(-alpha) * beta**alpha * ((1-1j*s/beta)**alpha - 1 + 1j*s*alpha/beta)
            phi_l = lambda s : 1j*s*gamma_l + f(s, c_p, beta_p, alpha_p) + f(-s, c_n, beta_n, alpha_n)
            log_char_f = phi_l(u) - phi_l(a*u)

    elif flag == 3 or flag == 4:

        # Set parameters for readability
        b = params['b']
        beta = params['beta']
        c = params['c']
        sigma = params['sigma']
        alpha = params['alpha']
        k = params['k']
        a = np.exp(-b * t)

        # OU-NTS
        if flag == 3:
            log_char_f = -c*beta**alpha*gamma(1-alpha)/(2*alpha*b) * (I(1j*sigma**2*u**2/2, alpha, beta, beta/a**2) + np.log(a**2))

        # NTS-OU
        else:
            phi_l = lambda s: (1-alpha)/(k*alpha) * (1 - (1 + k*(s**2*sigma**2/2)/(1 - alpha))**alpha)
            log_char_f = phi_l(u) - phi_l(a*u)

    else:
        log_char_f = -1
        print('Wrong value for flag \n')

    return log_char_f


def heuristic_estimation(log_char_f, a_tilde, delta_times):
    """
    Heuristic estimation to compute optimal w_tilde and b_tilde

    INPUT
    log_char_f:   log characteristic function of the process
    a_tilde:      optimal a_tilde parameter
    delta_times:  time interval
    """

    # Defining u large enough
    u = np.arange(500,10**4 + 1,step=500)

    # Evaluating the characteristic function
    phi = np.exp(log_char_f(u-1j*a_tilde, delta_times))

    # Performing linear regression
    w_tilde = np.arange(0.001, 1, step=0.01)
    check = np.zeros(len(w_tilde))
    b_tilde = np.zeros(len(w_tilde))
    y = -np.log(abs(phi))
    if np.any(np.isinf(y)):
        b_tilde_opt = np.inf
        w_tilde_opt = 0
    else:
        for i in range(len(w_tilde)):
            x = abs(u)**w_tilde[i]
            lm = LinearRegression().fit(x.reshape(-1,1), y)
            b_tilde[i] = lm.coef_
            check[i] = lm.score(x.reshape(-1,1), y) # R^2

        # Choose the best fitting regression
        b_tilde_opt = b_tilde[np.argmax(check)]
        w_tilde_opt = w_tilde[np.argmax(check)]

        # Compute the best fitting model
        x = abs(u) ** w_tilde_opt
        y = -np.log(abs(phi))
        lm = LinearRegression().fit(x.reshape(-1,1), y)

        # Plot the best fitting model
        plt.figure(figsize=(10, 10))
        plt.plot(u, -np.log(abs(phi)), 'bo')
        plt.plot(u, lm.intercept_+b_tilde_opt*abs(u)**w_tilde_opt, 'r--')
        plt.title("best fitting model")
        # plt.show()

    return b_tilde_opt, w_tilde_opt


def compute_cdf(psi, a_tilde, R, h, M, delta_time):
    """
    Computation of the CDF with FFT method

    INPUT
    psi:          log characteristic function of the process
    a_tilde:      optimal a_tilde parameter
    R:            cdf parameter
    h:            discretization parameter
    M:            discretization parameter
    delta_times:  time interval
    """

    # Discretization Parameters
    N = 2**M
    dgamma = 2*math.pi/(h*N)
    u_N = h*(N-1)/2
    u_1 = -u_N
    x_N = dgamma*(N-1)/2
    x_1 = -x_N

    # Grids computation
    x = np.arange(x_1, x_N+0.00001, step=dgamma)
    u = np.arange(u_1, u_N+0.00001, step=h)

    # Compute f and discretize it
    f = lambda u : np.exp(psi(u - 1j*a_tilde, delta_time))/(1j*u + a_tilde)
    f_j = f(u) * np.exp(-1j * x_1 * h * np.arange(N))

    # Compute the integral on the pre-set grid
    f_hat = np.real(h*np.exp(-1j*u_1*x)*fft(f_j))
    F = R - np.exp(-a_tilde*x)/(2*math.pi) * f_hat

    return F,x


def clean_cdf(F,x,dt,flag):
    """
    Cleans the cdf from numerical artifacts

    INPUT
    F:      numerically obtained cdf
    x:      grid on which F is computed
    flag:   indicates the process
    """

    # Discard points where the x is bigger than D*dt and smaller than -D*dt
    D = 35 * int(flag == 1) + 40 * int(flag == 2) + 40 * int(flag == 3) + 15 * int(flag == 4)
    x0 = x[(x >= -D * dt) & (x <= D * dt)]
    F0 = F[(x >= -D * dt) & (x <= D * dt)]

    # Discard points where the cdf is bigger than 1 or smaller than 0
    F_01 = F0[(F0 >= 0) & (F0 <= 1)]
    x_01 = x0[(F0 >= 0) & (F0 <= 1)]

    # Discard points where F is not increasing
    indices_clean = [0]  # initialize with 0
    for i in range(1, len(F_01)):
        if F_01[i] >= F_01[indices_clean[-1]]:
            indices_clean.append(i)
    F_02 = F_01[indices_clean]
    x_02 = x_01[indices_clean]

    # Reset symmetric grid
    xm = min(abs(x_02[0]), x_02[-1])
    x_clean = x_02[(x_02 >= -xm) & (x_02 <= xm)]
    F_clean = F_02[(x_02 >= -xm) & (x_02 <= xm)]

    return F_clean,x_clean

def transform_cdf(F,x,params, maturity,flag):

    x_pos = x[(x >= 0)]
    h =  -np.real(log_characteristic_function(params, flag, -1j, maturity))

    y = - h + np.log(x_pos)

    tail_indexes_right = y > x[-1]
    tail_indexes_left = y < x[0]
    inside_indexes = ~(tail_indexes_left | tail_indexes_right)



    y_inside = y[inside_indexes]
    interp_method = 'cubic'
    F_y_inside = interp1d(x, F, kind=interp_method)(y_inside)
    x_inside = x_pos[inside_indexes]

    return F_y_inside,x_inside


def compute_pdf_OUlevy(params, delta_times, flag):
    """
    INPUT
    params:       dictionary containing the process params
    Nsim:         number of simulations
    delta_times:  vector of delta times
    flag:         1 -> OU-TS
                  2 -> TS-OU
                  3 -> OU-NTS
                  2 -> NTS-OU
    """
    warnings.filterwarnings("ignore")

    # Computing the log characteristic function
    log_char_f = lambda u,t : log_characteristic_function(params, flag, u, t)

    # Computing optimal parameters
    if flag == 3 or flag == 4:
        a_tilde = 0.5
        p_star = 1
    else:
        a_tilde = np.sign(params['beta_p']-params['beta_n']) * np.maximum(params['beta_p'],params['beta_n'])/2
        p_star = np.maximum(params['beta_p'],params['beta_n'])
    b_tilde, w_tilde = heuristic_estimation(log_char_f, a_tilde, delta_times)

    # Computing the cdf with FFT
    M = 12
    N = 2**M
    k = (math.pi*p_star/(b_tilde*N**w_tilde))**(1/(w_tilde+1))
    if k == 0:
        h = 0.1
    else:
        h = min(0.1, k)
    R = int(a_tilde>0)+int(a_tilde == 0)/2
    F,x = compute_cdf(log_char_f, a_tilde, R, h, M, delta_times)

    # Cleaning the cdf
    F,x = clean_cdf(F, x, delta_times, flag)


    # trasformin the cdf in order to get the cdf of exp(X + h)
    F, x = transform_cdf(F, x, params, delta_times, flag)


    # Computing the pdf
    dF = F[2:] - F[:-2]
    dx = x[2:] - x[:-2]

    pdf = dF / dx

    return pdf, x[1:-1]


def lastStep(x_grid, cdf_grid, num_simulations):
    """
    Simplified final step of the fast MC algorithm

    INPUT
    x_grid:             grid of points for which the CDF has been evaluated , after cleaning
    cdf_grid:           values of the cdf for the points in x_grid
    num_simulations:    number of path
    """

    # Extract standard normal rv
    u = np.random.rand(num_simulations, 1)

    # Separate u
    tail_indexes_right = u > cdf_grid[-1]
    tail_indexes_left = u < cdf_grid[0]
    inside_indexes = ~(tail_indexes_left | tail_indexes_right)

    # Interpolation (spline)
    u_inside = u[inside_indexes]
    interp_method = 'cubic'
    paths_inside = interp1d(cdf_grid, x_grid, kind=interp_method)(u_inside)
    paths = paths_inside

    # Left tail y = a * exp(b * x)
    if np.any(tail_indexes_left):
        b_left = (np.log(cdf_grid[1]) - np.log(cdf_grid[0])) / (x_grid[1] - x_grid[0])
        a_left = cdf_grid[0] * np.exp(-b_left * x_grid[0])
        paths_left_tail = (np.log(u[tail_indexes_left]) - np.log(a_left)) / b_left
        paths = np.concatenate((paths, paths_left_tail))

    # Right tail y = 1 - a * exp(-b * x)
    if np.any(tail_indexes_right):
        b_right = (np.log(1 - cdf_grid[-1]) - np.log(1 - cdf_grid[-2])) / (x_grid[-2] - x_grid[-1])
        a_right = (1 - cdf_grid[-2]) * np.exp(b_right * x_grid[-2])
        paths_right_tail = (np.log(1 - u[tail_indexes_right]) - np.log(a_right)) / (-b_right)
        paths = np.concatenate((paths, paths_right_tail))

    return paths


def simulation(params, Nsim, delta_times, flag):
    """
    FastMC Simulation algorithm

    INPUT
    params:       dictionary containing the process params
    Nsim:         number of simulations
    delta_times:  vector of delta times
    flag:         1 -> OU-TS
                  2 -> TS-OU
                  3 -> OU-NTS
                  2 -> NTS-OU
    """

    # Computing the log characteristic function
    log_char_f = lambda u,t : log_characteristic_function(params, flag, u, t)

    # Computing optimal parameters
    if flag == 3 or flag == 4:
        a_tilde = 0.5
        p_star = 1
    else:
        a_tilde = np.sign(params['beta_p']-params['beta_n']) * np.maximum(params['beta_p'],params['beta_n'])/2
        p_star = np.maximum(params['beta_p'],params['beta_n'])
    b_tilde, w_tilde = heuristic_estimation(log_char_f, a_tilde, delta_times)

    # Computing the cdf with FFT
    #M = 19 # used for negative alphas and to check improvements
    M = 12
    N = 2**M
    k = (math.pi*p_star/(b_tilde*N**w_tilde))**(1/(w_tilde+1))
    if k == 0:
        h = 0.1
    else:
        h = min(0.1, k)
    R = int(a_tilde>0)+int(a_tilde == 0)/2
    F,x = compute_cdf(log_char_f, a_tilde, R, h, M, delta_times)

    # Cleaning the cdf
    F,x = clean_cdf(F, x, delta_times, flag)

    # Computing the paths
    X = lastStep(x, F, Nsim)

    return X


def compute_pdf_lognormal(x, mu, std_div):
    pdf = np.exp(-(np.log(x) - mu) ** 2 / (2 * std_div ** 2)) / (x * std_div * np.sqrt(2 * np.pi))
    return pdf


if __name__ == "__main__":
    np.random.seed(42)
    delta_times = np.ones(1)
    # Process parameters
    x0 = 0
    sigma = 0.201
    k = 0.256
    alphas = np.arange(0.7,0.8, 0.2)

    for i, alpha in enumerate(alphas):
        print(alpha)

        if alpha > 0:
            # Parameters
            beta = (1 - alpha) / k
            c = 1 / gamma(1 - alpha) * beta ** (1 - alpha)

            # NTS-OU SIMULATION
            print('NTS-OU simulation, alpha = ', alpha, '\n')
            b = 1
            a = np.exp(-b * delta_times)
            flag = 4
            params_NTS_OU = {'x0': x0, 'b': b, 'beta': beta, 'c': c, 'alpha': alpha, 'sigma': sigma, 'k': k}
            pdf, x = compute_pdf_OUlevy(params_NTS_OU, delta_times, flag)




