import gymnasium as gym
import numpy as np
from trd3_torch import Agent
from shared.utils import save_model_gif

if __name__ == '__main__': 
  # Set up the environment
  env = gym.make("LunarLanderContinuous-v3", render_mode='rgb_array')

  # Set up the agent
  agent = Agent(alpha=0.001, beta=0.001, 
                    input_dims=env.observation_space.shape, tau=0.005,
                    env=env, batch_size=100, layer1_size=400, layer2_size=300,
                    n_actions=env.action_space.shape[0])
  
  # Load the agent
  agent.load_models()
  
  # Save the model as a gif
  save_model_gif(env, agent, 'TrD3')