import gym
from controller import State, Controller
import config as Config

#TODO: These guys should come from the command line.
ANIMATE = True

def play(env,path=None):

        if path:
            controller = Controller(path)
        else:
            controller = Controller()

        old_state = State(env.reset()).discretize_features()
        done = False
        points = 0
        while not done:
            env.render()
            action = controller.take_best_action(old_state)
            observation, reward, done, _ = env.step(action)
            points += reward
            new_state = State(observation).discretize_features()
            old_state = new_state
        print(points)
def main():

    env = gym.make('LunarLander-v2')


    play(env,'hein.npy')
    exit(0)

    done = False

    controller = Controller()
    
    episode = 0
    best_reward = -500
    while(episode < Config.ITERATIONS):
        
        old_state = State(env.reset()).discretize_features()
        done = False
        it = 0
        total_reward = 0
        while not done:
            it += 1 
            action = controller.take_action(old_state, episode)
            observation, reward, done, _ = env.step(action)
            total_reward += reward
            new_state = State(observation).discretize_features()
            controller.update_q(new_state, old_state, action, reward, episode)
            old_state = new_state
            if done:
                if(episode%30 == 0):
                    print("Finished {} with {} steps and {} reward {}.".format(episode, it, total_reward, best_reward))

                if total_reward > best_reward:
                    best_reward = total_reward
                break
        

        episode += 1

    controller.q_table.save("{}.txt".format(best_reward))
    #print("Solved in {} episodes".format(episode))
    if ANIMATE:
        play(env)
        #print(points)


if __name__ == '__main__':
    main()