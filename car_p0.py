import gym
import random
import math
from itertools import product
import numpy as np


env = gym.make('CartPole-v0')

ANGLE_LIMITS = (math.radians(1), math.radians(3))
SPEED_LIMITS =  (1.5, 2)

class State:
    def __init__(self, features):
        self.features = features

    @staticmethod
    def discretization_levels():
        return (1,1,5,3)


    @staticmethod
    def enumerate_all_possible_states():
        levels = State.discretization_levels()
        levels_possibilities = [(j for j in range(i)) for i in levels]

        return tuple([i for i in product(*levels_possibilities)])


    def discretize_features(self):

        if self.features[2] <= ANGLE_LIMITS[0] and self.features[2] >= -ANGLE_LIMITS[0]:
            angle = 0
        elif self.features[2] > ANGLE_LIMITS[0] and self.features[2] <= ANGLE_LIMITS[1]:
            angle = 1
        elif self.features[2] < -ANGLE_LIMITS[0] and self.features[2] >= -ANGLE_LIMITS[1]:
            angle = 2
        elif self.features[2] > ANGLE_LIMITS[1]:
            angle = 3
        elif self.features[2] < -ANGLE_LIMITS[1]:
            angle = 4
        else:
            print(math.degrees(self.features[2]))

        
        if self.features[2] <= SPEED_LIMITS[0] and self.features[2] >= -SPEED_LIMITS[0]:
            vel = 0
        elif self.features[2] > SPEED_LIMITS[0]:
            vel = 1
        elif self.features[2] < -SPEED_LIMITS[0]:
            vel = 2

        return (0, 0, angle, vel)

    def __str__(self):
        disc = self.discretize_features()
        return "({},{},{},{})".format(disc[0], disc[1], disc[2], disc[3])


class QTable:
    def __init__(self):
        self.q_table = self.init_q_table()

    def init_q_table(self):

        new_q_table = {}
        pos_states = State(range(4)).enumerate_all_possible_states()

        for state in pos_states:
            new_q_table[state] = [0.0 for i in range(2)]

        return new_q_table

    def get_q_value(self, key, action=None):
        if action is None:
            return self.q_table[key]
        return self.q_table[key][action]

    def set_q_value(self, key, action, new_q_value):
        self.q_table[key][action] = new_q_value

    @staticmethod
    def load(path):
        return np.load(path).item()

    def save(self, path, *args):
        return np.save(path, self.q_table)

    def __str__(self):
        string = ""
        for key in self.q_table.keys():
            string += str(key)
            string += str(self.q_table[key])
            string += '\n'

        return string


class Controller:
    def __init__(self, q_table_path = None):
        if q_table_path is None:
            self.q_table = QTable()
        else:
            self.q_table = QTable.load(q_table_path)

    def update_q(self, new_state, old_state, action, reward):

        learning_rate = 0.1
        discount = 0.99

        old_value = self.q_table.get_q_value(old_state, action)
        max_next_state = max(self.q_table.get_q_value(new_state))

        new_q_value = ((1-learning_rate)*old_value) + learning_rate * (reward + discount*max_next_state)

        #print(new_q_value)

        self.q_table.set_q_value(old_state, action, new_q_value)

    def take_action(self, state, it):
        #print(discretized)
        #print(self.q_table.q_table[discretized][0])
        if self.q_table.get_q_value(state, 0) >= self.q_table.get_q_value(state, 1):
            if random.random() > 0.9:
                return 0
            else:
                return 1
        else:
            if random.random() > 0.9:
                return 0
            else:
                return 1

    def take_best_action(self, state):
        if self.q_table.get_q_value(state, 0) >= self.q_table.get_q_value(state, 1):
            return 0
        else:
            return 1


def main():
    done = False

    print(env.observation_space.low)
    print(env.observation_space.high)
    #Step may take two actions -> 0 to go right
    #                          -> 1 to go left

    controller = Controller()
    
    num_streaks = 0
    max_streaks = -1
    for episode in range(10000):
        
        old_state = State(env.reset()).discretize_features()
        
        for it in range(250):

            #env.render()

            action = controller.take_action(old_state, it)

            observation, reward, done, _ = env.step(action)
            new_state = State(observation).discretize_features()

            if(reward > 1):
                print(reward)

            controller.update_q(new_state, old_state, action, reward)

            old_state = new_state

            if done:
                if (it >= 10):
                    #print("what")
                    num_streaks += 1
                    if num_streaks > max_streaks:
                        max_streaks = num_streaks
                else:
                    num_streaks = 0
                break
        
        if num_streaks > 50:
            break


    print(controller.q_table)
    print(max_streaks)

    env.reset()
    done = False

    old_state = State(env.reset()).discretize_features()

    while not done:
        env.render()
        action = controller.take_best_action(old_state)
        observation, reward, done, _ = env.step(action)
        new_state = State(observation).discretize_features()
        old_state = new_state



if __name__ == '__main__':
    main()