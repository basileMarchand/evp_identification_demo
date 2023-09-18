import json
import numpy as np
from scipy.interpolate import interp1d

from model_to_identify import model_0, model_45, model_90


def load_stress(fname):
    with open(fname) as fid:
        data = json.load(fid)
    return np.array(data["time"]), np.array(data["stress"])


def cost_function(p, exp_stress: tuple, w=None):
    print(f"Start cost function evaluation for {p}")
    if w is None:
        w = (1., 1., 1.)
    # Evaluate models
    # 🚧 the three following lines can be computed in parallel 🚧
    time_0, stress_0 = model_0(p)
    time_45, stress_45 = model_45(p)
    time_90, stress_90 = model_90(p)

    time_0_exp, stress_0_exp = exp_stress[0]
    time_45_exp, stress_45_exp = exp_stress[1]
    time_90_exp, stress_90_exp = exp_stress[2]
    d_0 = stress_0_exp - interp1d(time_0, stress_0)(time_0_exp)
    d_45 = stress_45_exp - interp1d(time_45, stress_45)(time_45_exp)
    d_90 = stress_90_exp - interp1d(time_90, stress_90)(time_90_exp)
    print(f"... end cost function evaluation for {p}")
    return w[0]*d_0 + w[1]*d_45 + w[2]*d_90


if __name__ == "__main__":

    time0, stress_0_exp = load_stress("data_exp_0.json")
    time45, stress_45_exp = load_stress("data_exp_45.json")
    time90, stress_90_exp = load_stress("data_exp_90.json")

    out1 = cost_function(np.array([5_000, 1_500., 5_000., 600.]), ((
        time0, stress_0_exp), (time45, stress_45_exp), (time90, stress_90_exp)), w=(1./3., 1./3., 1./3.))
    out2 = cost_function(np.array([395_000, 1_000., 22_000., 600.]), ((
        time0, stress_0_exp), (time45, stress_45_exp), (time90, stress_90_exp)), w=(1./3., 1./3., 1./3.))

    import matplotlib.pyplot as plt

    plt.semilogy(time0, out1, label="test 1")
    plt.semilogy(time0, out2, label="test 2")
    plt.show()
