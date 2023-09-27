import logging

import numpy as np

from lvp.jobs.MainParallel import ParallelProcessing


class LvpParallel(ParallelProcessing):
    def __init__(
            self,
            h: float,
            n_jobs: int = -1,
            waiting_time: int = 1
    ):
        super().__init__(n_jobs, waiting_time)
        self.h = h

    def process(
            self,
            agent_id: int,
            x: np.matrix,
            D,
            b,
            loggs_path: str,
            response_dict: dict
    ) -> None:
        """
        """
        logging.basicConfig(filename=loggs_path + f'/_loggs_{agent_id}_alvp.log', filemode='a', level=logging.INFO)

        lvp = (D - b) * x

        response_dict[agent_id] = lvp.item(0)

        logging.info(f"Agent {agent_id} ended counting lvp")


    @classmethod
    def extract_response_to_array(self, response, keys):
        return np.matrix([response[key] for key in keys]).transpose()
