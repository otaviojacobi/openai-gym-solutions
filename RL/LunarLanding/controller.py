import gym
import random
import math
from itertools import product
import numpy as np
from typing import Tuple, List, Dict, Union

#TODO: Remove this from here and find a better way
env = gym.make('LunarLander-v2')
#SENSORS
'''
[ posicao_horizontal, 
  posicao_vertical, 
  velocidade_horizontal, 
  velocidade_vertical, 
  angulo,
  velocidade_angular,
  contat_perna_esquerda
  contat_perna_direita ]
'''
StateType = Tuple[int, ...]
class State:
    def __init__(self, features: List[float]) -> None:
        self.features = features

    @staticmethod
    def discretization_levels() -> Tuple[int, ...]:
        return (2,2,2,3,2,2,2,2)


    @staticmethod
    def enumerate_all_possible_states() -> Tuple[StateType]:

        levels = State.discretization_levels()
        levels_possibilities = [(j for j in range(i)) for i in levels]

        return tuple([i for i in product(*levels_possibilities)])


    def discretize_features(self) -> StateType:
        

        discretized = []
        #FEATURE 0
        if abs(self.features[0]) <= 0.04:
            discretized.append(1)
        else:
            discretized.append(0)
        #FEATURE 1
        if self.features[1] <= 0.04:
            discretized.append(1)
        else:
            discretized.append(0)
        #FEATURE 2
        if self.features[2] <= 0.04:
            discretized.append(1)
        else:
            discretized.append(0)
        #FEATURE 3
        if self.features[3] <= 0.01:
            discretized.append(2)
        elif self.features[3] <= 0.08:
            discretized.append(1)
        else:
            discretized.append(0)

        #FEATURE 4
        if abs(self.features[4] <= 0.02):
            discretized.append(1)
        else:
            discretized.append(0)

        #FEATURE 5
        if abs(self.features[5] <= 0.02):
            discretized.append(1)
        else:
            discretized.append(0)
        #FEATURE 6
        discretized.append(int(self.features[6]))
        #FEATURE 7
        discretized.append(int(self.features[7]))

        return tuple(discretized)


class QTable:
    def __init__(self, table=None) -> None:
        if table is None:
            self.q_table = self.init_q_table()
        else:
            self.q_table = table
    def init_q_table(self) -> Dict[StateType, List[float]]:

        new_q_table = {}
        pos_states = State(range(8)).enumerate_all_possible_states()

        for state in pos_states:
            new_q_table[state] = [0.0 for i in range(4)]

        return new_q_table

    def get_q_value(self, key: StateType, action: Union[None, int] = None) -> float:

        if action is None:
            return self.q_table[key]
        return self.q_table[key][action]

    def set_q_value(self, key: StateType, action: int, new_q_value: float) -> None:
        self.q_table[key][action] = new_q_value

    @staticmethod
    def load(path: str) -> "QTable":
        return np.load(path).item()

    def save(self, path: str, *args) -> None:
        np.save(path, self.q_table)

class Controller:
    def __init__(self, q_table_path: Union[None, str] = None) -> None:
        if q_table_path is None:
            self.q_table = QTable()
        else:
            self.q_table = QTable(QTable.load(q_table_path))

    def update_q(self, new_state: StateType, old_state: StateType, action: int, reward: int, it: int) -> None:

        learning_rate = 0.1
        discount = 0.99

        old_value = self.q_table.get_q_value(old_state, action)
        max_next_state = max(self.q_table.get_q_value(new_state))

        new_q_value = ((1-learning_rate)*old_value) + learning_rate * (reward + discount*max_next_state)

        self.q_table.set_q_value(old_state, action, new_q_value)

    def take_action(self, state: StateType, it: int) -> int:

            exp_rate = 0.995**it

            if exp_rate < 0.01:
                exp_rate = 0.01

            if random.random() < exp_rate:
                action = env.action_space.sample()
            else:
                action = np.argmax(self.q_table.get_q_value(state))


            return action
        

    def take_best_action(self, state: StateType) -> int:
        if self.q_table.get_q_value(state, 0) >= self.q_table.get_q_value(state, 1):
            return 0
        else:
            return 1