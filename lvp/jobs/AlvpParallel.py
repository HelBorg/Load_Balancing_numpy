import logging

import numpy as np

from lvp.jobs.MainParallel import ParallelProcessing


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
            loggs_path: str,
            response_dict: dict
    ) -> None:
        """
        """
        logging.basicConfig(filename=loggs_path + f'/_loggs_{agent_id}_alvp.log', filemode='a', level=logging.INFO)
        x_agent = x.item((agent_id, 0))
        x_n = 1 / (gamma[0] + self.alpha * (self.mu - self.eta)) \
              * (self.alpha * gamma[0] * nesterov_step + gamma[1] * x_agent)

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
            "x": x_agent - x_avg,
            "nesterov_step": nesterov_step
        }

        logging.info(f"Agent {agent_id} ended counting lvp")

    @classmethod
    def extract_response_to_array(self, response, parameter, keys):
        return np.matrix([response[key][parameter] for key in keys]).transpose()
