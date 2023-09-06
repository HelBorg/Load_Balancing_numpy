import logging
from multiprocessing import Lock, Manager
from typing import Dict

import numpy as np

from lvp.MainParallel import ParallelProcessing
from lvp.models.Agent import Agent


class AlvpParallel(ParallelProcessing):
    def __init__(
            self,
            alpha: float,
            mu: float,
            eta: float,
            h: float,
            L: float,
            n_jobs: int = -1,
            waiting_time: int = 1
    ):
        super().__init__(n_jobs, waiting_time)
        self.alpha = alpha
        self.mu = mu
        self.eta = eta
        self.h = h
        self.L = L

    def process(
            self,
            agent_id: int,
            x: np.matrix,
            nesterov_step: float,
            gamma: list,
            D,
            b,
            response_dict: dict
    ) -> None:
        """
        """
        logging.basicConfig(filename=loggs_path + f'/_loggs_{agent_id}_alvp.log', filemode='a', level=logging.INFO)
        x_n = 1 / (gamma[0] + self.alpha * (self.mu - self.eta)) \
              * (self.alpha * gamma[0] * nesterov_step + gamma[1] * x.item((agent_id, 0)))

        # Create matrix with repeating x and x_n on main diagonal
        x = x.astype(float)
        x[agent_id] = x_n
        y = (D - b) * x
        y_vec = y.item((0, 0))

        nesterov_step = 1 / gamma[0] * \
                             ((1 - self.alpha) * gamma[0] * nesterov_step
                              + self.alpha * (self.mu - self.eta) * x_n
                              - self.alpha * y_vec)

        x_avg = x_n - self.h * y_vec

        H = self.h - self.h * self.h * self.L / 2

        if H - self.alpha * self.alpha / (2 * gamma[1]) < 0:
            logging.warning(H)
            print(f"H {H}")
            logging.exception(f"Oh no: {H - self.alpha * self.alpha / (2 * gamma[1])}")
            print(f"Oh no: {H - self.alpha * self.alpha / (2 * gamma[1])}")
            raise BaseException()

        response_dict[agent_id] = {
            "x_avg": x_avg,
            "nesterov_step": nesterov_step
        }

        logging.info(f"Agent {agent_id} ended counting lvp")


