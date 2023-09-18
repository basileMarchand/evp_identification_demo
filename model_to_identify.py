import numpy as np
from evpsim.behavior.criterion import MisesCriterion
from evpsim.behavior.flow import NortonFlow
from evpsim.behavior.isotropic_hardening import LinearIsotropicHardening
from evpsim.behavior.kinematic_hardening import LinearKinematicHardening, NonLinearKinematicHardening
from evpsim.behavior.potential import PotentialEVP
from evpsim.behavior.genevp import GeneralizedElastoViscoPlastic
from evpsim.tools.elasticity_helper import transverse_elasticity

from evpsim.simulator import SimuLoad, MaterialSimulator, LocalFrame


def model_theta(x, local):
    C, D, C1, H, R0 = x.ravel()
    criterion = MisesCriterion()
    flow = NortonFlow(600., 3.)
    iso = LinearIsotropicHardening(H)
    kin1 = NonLinearKinematicHardening(C, D, 0)
    kin2 = LinearKinematicHardening(C1, 1)
    pot = PotentialEVP(R0, criterion, flow, iso, [kin1, kin2], name="ep")
    elas = transverse_elasticity(166_000., 205_700., 0.3, 0.3, 79_115.)
    beha = GeneralizedElastoViscoPlastic(elas, [pot])

    load = SimuLoad()
    load.addComponent('eto33', (0., 0.025, 0.05, 0.075, 0.1),
                      (0., 0.015, 0., -0.015, 0.), repeat=2)

    load.addComponent('eto33', (0., 0.25, 0.5, 0.75, 1.),
                      (0., 0.015, 0., -0.015, 0.), repeat=2)

    simu = MaterialSimulator()
    simu.setDTime(1./399.)
    simu.setMaterial(beha)
    simu.setLoad(load)
    simu.setLocalFrame(local)
    simu.outputCycles([x for x in range(100)])
    _ = simu.updateOutputShapes()

    time, _ = simu.compute(method="lm", options={"xtol": 1.e-2})
    return time, simu._stress_history[:, 2, 2].ravel()


def model_0(x):
    local = LocalFrame(np.array([1., 0., 0.]),
                       np.array([0., 1., 0.]))
    return model_theta(x, local)


def model_45(x):
    local = LocalFrame(np.array([1., 0., 1.]),
                       np.array([0., 1., 0.]))
    return model_theta(x, local)


def model_90(x):
    local = LocalFrame(np.array([0., 0., 1.]),
                       np.array([0., 1., 0.]))
    return model_theta(x, local)


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    time0, stress0 = model_0(np.array([5_000, 1_500., 5_000., 0., 600.]))
    time1, stress1 = model_0(np.array([500_000, 1_500., 50_000., 3000., 1_400.]))
    plt.plot(time0, stress0, label="test 1")
    plt.plot(time1, stress1, label="test 2")
    plt.show()
