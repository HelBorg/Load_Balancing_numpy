import logging
import os

import numpy as np
import pandas as pd

from lvp.models.agent import Agent
from lvp.models.parameters import Parameters

DEFAULT_CACHE_PATH = "/cache/"
DEFAULT_LOGGS_PATH = DEFAULT_CACHE_PATH + "logs/"
DEFAULT_NEIGH_FILE = DEFAULT_CACHE_PATH + "alg_params/neigh.pkl"
DEFAULT_NOISE_FILE = DEFAULT_CACHE_PATH + "alg_params/noise.pkl"


class LbAlgorithm:
    """
    Abstract class for implementing different algorithms
    to solve load balancing problem
    """

    def __init__(self, params: Parameters, **kwargs):
        """
        :param parameters: parameters of class Parameters
        """
        self.n = params.n
        self.agents = []

        # Results
        self.theta_hat = []
        self.sequence = []
        self.sequence_2 = []

        # Set up logging
        file_path = os.path.realpath(__file__)
        project_dir = "\\".join(file_path.split("\\")[:-3])
        loggs_id = max([int(i) for i in os.listdir(project_dir + DEFAULT_LOGGS_PATH) if i.isdigit()] + [-1])
        self.loggs_path = project_dir + DEFAULT_LOGGS_PATH + f"{int(loggs_id) + 1}/"
        os.mkdir(self.loggs_path)
        logging.basicConfig(filename=f'{self.loggs_path}_loggs_lvp.log', filemode='w', level=logging.INFO,
                            # force=True
                            )

    def run(
            self,
            num_steps: int = 100,
            productivities: list = [],
            generate: bool = True,
            is_logging: bool = False,
            neigh_file: str = DEFAULT_NEIGH_FILE
    ):
        self.create_agents(num_steps, generate, productivities)
        self.prepare(generate, neigh_file, num_steps)

        for step in range(num_steps):
            if is_logging:
                logging.info(f"\n\nStep: {step}")
                for agent in self.agents:
                    df = [task.to_dict() for task in agent.tasks]
                    logging.info(pd.DataFrame(df).sum())

            # Add some neighbour edges
            self.extract_step_info(step)

            # Get new tasks
            [agent.update_with_new_tasks(step) for agent in self.agents]
            self.sequence.append([agent.get_real_queue_length() for agent in self.agents])
            self.sequence_2.append([agent.theta_hat for agent in self.agents])

            # Complete some tasks
            [agent.complete_tasks() for agent in self.agents]

            self.algorithm_step(is_logging, step)

            print(f"Step {step} is completed")

    def create_agents(self, num_steps, generate, productivities):
        task_pool = self.create_task_pool(num_steps, generate)
        self.agents = [
            Agent(id, productivities[id], num_steps=num_steps, generate=generate, task_pool=task_pool)
            for id in range(self.n)
        ]
        self.theta_hat = np.matrix([len(agent.tasks) for agent in self.agents]).transpose()

    def create_task_pool(self, num_steps: int, generate: bool):
        pass

    def prepare(self, generate, neigh_file, num_steps):
        raise NotImplementedError('subclasses must override prepare()!')

    def extract_step_info(self, step):
        pass

    def algorithm_step(self, is_logging, step):
        raise NotImplementedError('subclasses must override algorithm_step()!')

    def log(self, is_logging: bool, text: str):
        if is_logging:
            logging.info(text)
