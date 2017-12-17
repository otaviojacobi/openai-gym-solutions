import gym
import random
import math
from itertools import product
import numpy as np
import matplotlib.pyplot as plt

env = gym.make('CartPole-v1')

LIMITS = list(zip(env.observation_space.low, env.observation_space.high))
LIMITS[1] = [-0.5, 0.5]
LIMITS[3] = [-math.radians(50), math.radians(50)]


class State:
    def __init__(self, features):
        self.features = features

    @staticmethod
    def discretization_levels():
        return (1,1,6,3)


    @staticmethod
    def enumerate_all_possible_states():
        levels = State.discretization_levels()
        levels_possibilities = [(j for j in range(i)) for i in levels]

        return tuple([i for i in product(*levels_possibilities)])


    def discretize_features(self):
        
        discretized = []
        levels = State.discretization_levels()

        for idx in range(len(self.features)):
            if self.features[idx] <= LIMITS[idx][0]:
                discretized.append(0)
            elif self.features[idx] >= LIMITS[idx][1]:
                discretized.append(levels[idx]-1)     
            else:
                bound_width = LIMITS[idx][1] - LIMITS[idx][0]
                offset = (levels[idx]-1)*LIMITS[idx][0]/bound_width
                scaling = (levels[idx]-1)/bound_width
                bucket_index = int(round(scaling*self.features[idx] - offset))
                discretized.append(bucket_index)

        return tuple(discretized)

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

    def update_q(self, new_state, old_state, action, reward, it):

        #learning_rate = Controller.get_learning_rate(it)
        learning_rate = 0.2
        #print(learning_rate)
        discount = 0.99

        old_value = self.q_table.get_q_value(old_state, action)
        max_next_state = max(self.q_table.get_q_value(new_state))

        new_q_value = ((1-learning_rate)*old_value) + learning_rate * (reward + discount*max_next_state)


        self.q_table.set_q_value(old_state, action, new_q_value)

    def take_action(self, state, it):
            exp_rate = 0.99**it

            if random.random() < exp_rate:
                action = env.action_space.sample()
            else:
                action = np.argmax(self.q_table.get_q_value(state))
            return action
        

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
    for episode in range(300):
        
        old_state = State(env.reset()).discretize_features()
        done = False

        for it in range(250):

            #env.render()

            action = controller.take_action(old_state, episode)

            observation, reward, done, _ = env.step(action)
            new_state = State(observation).discretize_features()

            controller.update_q(new_state, old_state, action, reward, episode)

            old_state = new_state

            if done:
                print("Finished {} with {} steps.".format(episode, it))
                if (it >= 199):
                    num_streaks += 1
                    if num_streaks > max_streaks:
                        max_streaks = num_streaks
                else:
                    num_streaks = 0
                break
        
    
        if max_streaks > 50:
            break


    print(controller.q_table)
    print(max_streaks)

    old_state = State(env.reset()).discretize_features()

    done = False

    points = 0
    while not done:
        if points >= 500:
            break
        env.render()
        action = controller.take_best_action(old_state)
        observation, reward, _, _ = env.step(action)
        new_state = State(observation).discretize_features()
        old_state = new_state
        points += 1
    print(points)

    plt.plot(plot)
    plt.show()



if __name__ == '__main__':
    main()