import gym
from controller import State, Controller
import config as Config

#TODO: These guys should come from the command line.
ANIMATE = True
MAX_EPISODE = 1000

def main():

    env = gym.make('CartPole-v0')

    done = False

    controller = Controller()
    
    num_streaks = 0
    episode = 0
    while(True):
        
        old_state = State(env.reset()).discretize_features()
        done = False
        it = 0
        while not done:
            it += 1 
            action = controller.take_action(old_state, episode)
            observation, reward, done, _ = env.step(action)
            new_state = State(observation).discretize_features()
            controller.update_q(new_state, old_state, action, reward, episode)
            old_state = new_state
            if done:
                #print("Finished {} with {} steps.".format(episode, it))
                if (it >= Config.DURATION):
                    num_streaks += 1
                else:
                    num_streaks = 0
                break
        
    
        if num_streaks > Config.STREAKS:
            break
        
        if episode >= MAX_EPISODE:
            break

        episode += 1

    if ANIMATE:
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

    #plt.plot(plot)
    #plt.show()



if __name__ == '__main__':
    main()