import numpy as np

from lvp.algorithms.accelerated_lvp import AcceleratedLVP
from lvp.algorithms.local_voting_protocol import LocalVotingProtocol
from lvp.algorithms.round_robin import RoundRobin
from lvp.models.parameters import Parameters

LVP = "LVP"
ALVP = "ALVP"
ROUND_ROBIN = "ROUND_ROBIN"

METHODS_TO_CLASSES = {
    LVP: LocalVotingProtocol,
    ALVP: AcceleratedLVP,
    ROUND_ROBIN: RoundRobin
}


def run_load_balancing(method, num_steps, params, generate, productivities):
    load_balancing = METHODS_TO_CLASSES[method](params=params)
    load_balancing.run(num_steps=num_steps, generate=generate, productivities=productivities)
    return load_balancing


if __name__ == "__main__":
    generate = True
    num_agents = 5
    productivities = [10] * num_agents
    num_steps = 20

    pars = Parameters()
    pars.n = num_agents
    pars.theta_hat = []
    Adj = np.array([
        [0, 0, 1, 1, 0],
        [0, 0, 0, 1, 1],
        [1, 0, 0, 0, 1],
        [1, 1, 0, 0, 0],
        [0, 1, 1, 0, 0]
    ])
    pars.b = Adj / 2

    pars.neib_add = 5
    pars.add_neib_val = 0.3
    pars.params_dict = {
        "L": 7.1,
        "mu": 0.9,
        "h": 0.2,
        "eta": 0.8,
        "gamma": [[0.07, 0.09, 0.11][0]],
        "alpha": [0.07, 0.09, 0.11][1]
    }
    run_load_balancing(ROUND_ROBIN, num_steps, pars, generate, productivities)
