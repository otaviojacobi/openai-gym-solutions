import gym
import random
import math
from itertools import product
import numpy as np
from typing import Tuple, List, Dict, Union
import copy
from time import sleep

#TODO: Remove this from here and find a better way
#env = gym.make('CartPole-v0')
#LIMITS = list(zip(env.observation_space.low, env.observation_space.high))
#LIMITS[1] = [-0.5, 0.5]
#LIMITS[3] = [-math.radians(50), math.radians(50)]


class Controller:
    def __init__(self) -> None:
        self.env = gym.make('CartPole-v0')
        self.episode = 1
        self.sensors = self.env.reset()


    def take_action(self, parameters: List) -> int:

        features = self.compute_features(self.sensors)

        ##print(features)
        #print(parameters)

        parameters = np.array_split(parameters, 2)

        action_scores = []
        for parameter in parameters:
            action_scores.append(np.matmul(features, parameter))

        return np.argmax(action_scores)


    def compute_features(self, sensors):

        ang_norm = (sensors[2] + 0.418879020)/(0.83775804)
        vel_norm = (sensors[3] + 3)/6
            

        return np.array([1, ang_norm, vel_norm])
        #raise NotImplementedError("This Method Must Be Implemented")

    def run_episode(self, weights):
        self.sensors = self.env.reset()
        done = False
        it = 0

        while not done:
            it += 1 
            action = self.take_action(weights)
            self.sensors, _, done, _ = self.env.step(action)
            if done:
                return it
        return it
    
    def play(self, weights):
        self.sensors = self.env.reset()
        done = False
        it = 0

        while not done:
            it += 1 
            action = self.take_action(weights)
            self.env.render()
            self.sensors, _, done, _ = self.env.step(action)
            sleep(0.03)

    def learn(self, weights) -> list:

        def gera_vizinho(pesos, var=0.01) -> list:
            perturbacao = []
            for peso in range(len(pesos)):
                #if peso//5==var: 
                perturbacao.append(random.normalvariate(0,var))
                #else:
                    #perturbacao.append(0)

            perturbacao=np.array(perturbacao)
            #print(perturbacao)
            soma = pesos + perturbacao
            return soma.tolist()

        estado_atual = weights
        # ADJUSTABLE ###########  
        T=100                  #
        N=50                  #
        temp_minima=20
        tx_resfriamento = 0.999 #
        ########################
        estado_atual_valor = self.run_episode(estado_atual)
        best_point = estado_atual
        best_point_value = estado_atual_valor # So i can always keep track of the best
        print(estado_atual_valor)
        while(T >= temp_minima):
            T *= tx_resfriamento
            for i in range(1,N):
                estado_candidato = gera_vizinho(estado_atual)
                estado_candidato_valor = self.run_episode(estado_candidato)

                if estado_candidato_valor > best_point_value:
                    best_point = estado_candidato
                    best_point_value = estado_atual_valor

                if best_point_value == 200:
                    print(best_point)
                    print(best_point_value)
                    print("BATATA")
                    return best_point

                delta = estado_candidato_valor - estado_atual_valor
                if delta > 0:
                    estado_atual = estado_candidato
                    estado_atual_valor = copy.deepcopy(estado_candidato_valor)
                else:
                    if random.uniform(0,1) < math.exp(delta/T):
                        estado_atual = estado_candidato
                        estado_atual_valor = copy.deepcopy(estado_candidato_valor)
            #print(self.episode)

            print("Temperature " + str(T))
            print("Valor Atual: " + str(estado_atual_valor))
            print(estado_atual)
            print(best_point_value)

        return best_point

def main():
    c = Controller()
    val = c.learn([0 for _ in range(6)])
    c.play(val)

        #raise NotImplementedError("This Method Must Be Implemented")
if __name__ == '__main__':
    main()