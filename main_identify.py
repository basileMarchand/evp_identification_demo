from scipy.optimize import least_squares
from grad_computer import grad_function
from cost_function import cost_function, load_stress
import json
import numpy as np

# Load experimental data


time0, stress_0_exp = load_stress("data_exp_0.json")
time45, stress_45_exp = load_stress("data_exp_45.json")
time90, stress_90_exp = load_stress("data_exp_90.json")


nb_parallel_process = 5
finite_difference = "fd1"


def fcost(p):
    return cost_function(p, ((time0, stress_0_exp), (time45, stress_45_exp), (time90, stress_90_exp)), (1., 1., 1.))


def grad(p):
    return grad_function(p, fcost, nb_parallel_process, finite_difference)


p_init = np.array([500_000, 1_500., 50_000., 1_400.])

bounds = ((100_000., 500., 5_000., 400.), (1_000_000, 1_500., 50_000., 1_400.))

opt = least_squares(fcost, p_init, jac=grad,
                    ftol=1.e-8, xtol=1.e-8, method="dogbox", verbose=2, bounds=bounds)

print(f"""Identified parameters : 
      C  = {opt.x[0]:.2f}
      D  = {opt.x[1]:.2f}
      C1 = {opt.x[2]:.2f}
      R0 = {opt.x[3]:.2f}
""")
