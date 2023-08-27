import logging
from multiprocessing import Lock, Manager
from typing import Dict

from lvp.MainParallel import ParallelProcessing
from lvp.models.Agent import Agent


class LvpParallel(ParallelProcessing):
    # def __init__(self, request_dic, *args, **kwargs):
    #     super().__init__(*args, **kwargs)
    #     self.request_dic = request_dic

    def process(self, agent: Agent, requests_dic, get_tasks: int, step: int, response_dict: dict) -> None:
        """
        Agent distribute it's tasks among neibours
        :param agent:  agent instance
        :param requests_dic: dictionary of requests
        :param get_tasks: result of counting local voting protocol for agent
        :return: dict {id send to: tasks to send}
        """
        agent_id = agent.id
        ag_tasks = len(agent.tasks)

        logging.basicConfig(filename=f'cache/loggs/_loggs_{agent_id}.log', filemode='a', level=logging.INFO)
        logging.info(f"\n\nStep {step}")

        requests_neib = [ind for ind in requests_dic.keys() if ind in agent.neighb]
        response = []
        for req_id in requests_neib:
            if get_tasks == 0:
                break
            num_tasks = min(self._enter_crit_sec(req_id, get_tasks), ag_tasks)

            if not num_tasks:
                continue

            logging.info(f"Agent {agent_id} send {num_tasks} tasks to {req_id}")
            response.append((req_id, num_tasks))
            get_tasks -= num_tasks
            ag_tasks -= num_tasks

        response_dict[agent_id] = response
        logging.info(f"Agent {agent_id} ended sending")

    def init_child(self, par_lock_: Lock, request_dic_: Dict[int, int] = None, *args) -> None:
        """
        Initiation used by each process to create shared variables
        :param par_lock_: lock used to update shared variable
        :param request_dic_: shared variable
        """
        super(LvpParallel, self).init_child(par_lock_)
        global request_dic
        request_dic = request_dic_

    def get_shared_vars(self, manager: Manager, shared_vars):
        """
        Create and return variable that would be shared among processes
        :param manager: multiprocessing manager used to create processes
        :param shared_vars: vars to share between different proccesses
        :return: created variable
        """
        request_dic = manager.dict()
        request_dic.update(shared_vars)
        return (request_dic,)

    def critical_section(self, req_id: int, can_send: int) -> int:
        """
        Change shared dictionary requests
        :param req_id: neighbour from whom want to take tasks
        :param can_send: number of tasks that can send
        :return: number of tasks to send
        """
        if req_id not in request_dic:
            return 0

        req = abs(request_dic[req_id])
        if req > can_send:
            request_dic[req_id] = -(req - can_send)
            send = can_send
        else:
            del request_dic[req_id]
            send = req
        return send