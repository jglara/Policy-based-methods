from unityagents import UnityEnvironment
import numpy as np
from ddpg import DDPGAgent
import torch
from collections import deque
import matplotlib.pyplot as plt
import yaml

def train_loop(brain_name, env, agent, n_episodes=2000, max_t=1000, goal=30.0, running_average=100, **params):
    """Train an agent with the environment and using hyper parameters
    
    Params
    ======
        brain_name: brain used in the environment
        env: gym environment to interact
        agent: agent to train
        n_episodes (int): maximum number of training episodes
        max_t (int): maximum number of timesteps per episode
        model_wegihts_file: file to save model weights
        params: hyper parameters
    """    

    averaged_scores = []
    scores_window = deque(maxlen=running_average)  # last running_average score
    for i_episode in range(1, n_episodes+1):
        env_info = env.reset(train_mode=True)[brain_name]
        states = env_info.vector_observations                  # get the current state
        scores = np.zeros(len(env_info.agents))
        for i in range(max_t):
            actions = agent.act(states)
            env_info = env.step(actions)[brain_name]
            next_states = env_info.vector_observations         # get next state 
            rewards = env_info.rewards                         # get reward 
            dones = env_info.local_done
            agent.step(states, actions, rewards, next_states, dones)
            
            scores += rewards
            if np.any(dones):
                break

            states = next_states

        score = np.mean(scores)
        scores_window.append(score)       # save most recent averaged score
        averaged_scores.append(score)                        # save most recent score    

        print('\rEpisode {} \tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_window)), end="")
        if i_episode % running_average == 0:
            print('\rEpisode {} \tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_window)))
        if len(scores_window) >= running_average and np.mean(scores_window)>=goal:
            print('\nEnvironment solved in {:d} episodes!\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_window)))
            torch.save(agent.actor_local.state_dict(), "actor.weight")
            torch.save(agent.critic_local.state_dict(), "critic.weight")
            break
    return averaged_scores    


if __name__ == "__main__":
    # Hyper parameters
    params = yaml.load(open("parameters.yaml"), Loader=yaml.FullLoader)["parameters"]

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    env = UnityEnvironment(file_name='./Reacher_Linux_NoVis_v2/Reacher.x86_64')

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

    scores = train_loop(brain_name, env, agent, max_t=1000, goal=10.0, running_average=20, **params)

    # Plot Statistics (Global scores and averaged scores)
    plt.subplot(2, 1, 2)
    plt.plot(np.arange(1, len(scores) + 1), scores)
    plt.ylabel('Reacher Environment Average Score')
    plt.xlabel('Episode #')
    plt.show()

    env.close()

