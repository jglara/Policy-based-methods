from unityagents import UnityEnvironment
import numpy as np
from ddpg import DDPGAgent
import torch
from collections import deque
import matplotlib.pyplot as plt
import yaml

def test_loop(brain_name, env, agent, max_t=2000, **params):
    env_info = env.reset(train_mode=True)[brain_name]
    states = env_info.vector_observations                  # get the current state 
    score = 0                          # initialize the score 
    for i in range(max_t):
        actions = agent.act(states)
        env_info = env.step(actions)[brain_name]
        next_states = env_info.vector_observations         # get next state 
        rewards = env_info.rewards                         # get reward 
        dones = env_info.local_done
        
        score += np.mean(rewards)
        if np.any(dones):
            break
        
        states = next_states

    return score


if __name__ == "__main__":
    # Hyper parameters
    params = yaml.load(open("parameters.yaml"), Loader=yaml.FullLoader)["parameters"]

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    env = UnityEnvironment(file_name='./Reacher_Linux_v2/Reacher.x86_64')

    # get the default brain
    brain_name = env.brain_names[0]
    brain = env.brains[brain_name]

    env_info = env.reset(train_mode=True)[brain_name]
    action_size = brain.vector_action_space_size
    states = env_info.vector_observations
    state_size = states.shape[1]
    num_agents = len(env_info.agents)
    print(f"agents = {num_agents} . Action space: {action_size} observation space: {state_size}")
    agent = DDPGAgent(state_size, action_size, device=device, **params)

    # load parameters
    agent.actor_local.load_state_dict(torch.load("actor.weight"))
    agent.critic_local.load_state_dict(torch.load("critic.weight"))

    score = test_loop(brain_name, env, agent, max_t=1000, **params)
    env.close()
