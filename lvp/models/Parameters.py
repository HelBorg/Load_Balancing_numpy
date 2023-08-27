import numpy as np


class Parameters:
    n: int
    b: np.matrix
    product: np.matrix
    queue_gen: np.matrix
    theta_hat: np.matrix = np.matrix([[0], [0], [0]])
    neib_add: int
    add_neib_val: float

    params_dict: dict