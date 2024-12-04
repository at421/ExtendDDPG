import gymnasium as gym
import numpy as np
from td3_torch import Agent
from shared.utils import plot_learning_curve

if __name__ == '__main__':
    training_cycle = []
    n_training_cycles = 10
    for cycle in range(1, n_training_cycles+1):
        env = gym.make("LunarLanderContinuous-v3")
        agent = Agent(alpha=0.001, beta=0.001, 
                    input_dims=env.observation_space.shape, tau=0.005,
                    env=env, batch_size=100, layer1_size=400, layer2_size=300,
                    n_actions=env.action_space.shape[0])
        
        n_games = 1500

        reward_range = (-np.inf, np.inf)
        best_score = reward_range[0]
        score_history = []

        #agent.load_models()

        for i in range(n_games):
            observation, info = env.reset()
            done = False
            score = 0
            while not done:
                action = agent.choose_action(observation)
                observation_, reward, done, trunc, info = env.step(action)
                agent.remember(observation, action, reward, observation_, done)
                agent.learn()
                score += reward
                observation = observation_
            score_history.append(score)
            avg_score = np.mean(score_history)
            avg_score_100 = np.mean(score_history[-100:])

            if avg_score_100 > best_score:
                best_score = avg_score
                agent.save_models()

            print('Train cycle', cycle, 'episode ', i, 'score %.2f' % score,
                    'trailing games avg %.3f' % avg_score)
        
        training_cycle.append(score_history)

    filename = 'lunar' + str(n_games) + '_' + str(n_training_cycles) + '.png'
    figure_file = 'plots/' + filename

    # Plot the average mean of the training cycles
    avg_score = np.mean(training_cycle, axis=0)
    x = [i+1 for i in range(n_games)]
    plot_learning_curve(x, avg_score, figure_file)

