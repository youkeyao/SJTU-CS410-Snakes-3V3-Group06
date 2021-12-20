import os
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from pathlib import Path
import sys
base_dir = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(base_dir))
from env.chooseenv import make

TEAM_HEAD = -3
TEAM_BODY = -1
OPPONENT_HEAD = -4
OPPONENT_BODY = -2
BEAN = 1

class Net(nn.Module):
    def __init__(self, obs_dim, act_dim):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(obs_dim, 50)
        self.fc1.weight.data.normal_(0, 0.1)   # initialization
        self.out = nn.Linear(50, act_dim)
        self.out.weight.data.normal_(0, 0.1)   # initialization

    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)
        actions_value = self.out(x)
        return actions_value


class DQN(object):
    def __init__(self, obs_dim, act_dim, args):

        self.path = os.path.dirname(os.path.abspath(__file__))
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.eps = args.epsilon
        self.gamma = args.gamma
        self.decay_speed = args.epsilon_speed
        self.batch_size = args.batch_size
        self.target_replace_iter = args.target_replace_iter
        self.memory_capacity = args.memory_capacity

        self.learn_step_counter = 0
        self.memory_counter = 0
        self.memory = np.zeros((self.memory_capacity, obs_dim * 2 + 2))

        self.eval_net, self.target_net = Net(obs_dim, act_dim), Net(obs_dim, act_dim)
        self.optimizer = torch.optim.Adam(self.eval_net.parameters(), lr=args.lr)
        self.loss_func = nn.MSELoss()

    def choose_action(self, x, evaluation=False):
        x = torch.FloatTensor(x)

        p = np.random.random()
        if p > self.eps or evaluation:
            actions_value = self.eval_net.forward(x)
            action = torch.max(actions_value, 1)[1].data.numpy()
        else:
            action = np.random.randint(0, 4, (x.shape[0]))
        
        self.eps *= self.decay_speed
        return action

    def store_transition(self, s, a, r, s_):
        transition = np.hstack((s, [a, r], s_))
        # replace the old memory with new memory
        index = self.memory_counter % self.memory_capacity
        self.memory[index, :] = transition
        self.memory_counter += 1

    def learn(self):
        # target parameter update
        if self.learn_step_counter % self.target_replace_iter == 0:
            self.target_net.load_state_dict(self.eval_net.state_dict())
        self.learn_step_counter += 1

        # sample batch transitions
        sample_index = np.random.choice(self.memory_capacity, self.batch_size)
        b_memory = self.memory[sample_index, :]
        b_s = torch.FloatTensor(b_memory[:, :self.obs_dim])
        b_a = torch.LongTensor(b_memory[:, self.obs_dim:self.obs_dim+1].astype(int))
        b_r = torch.FloatTensor(b_memory[:, self.obs_dim+1:self.obs_dim+2])
        b_s_ = torch.FloatTensor(b_memory[:, -self.obs_dim:])

        # q_eval w.r.t the action in experience
        q_eval = self.eval_net(b_s).gather(1, b_a)  # shape (batch, 1)
        q_next = self.target_net(b_s_).detach()     # detach from graph, don't backpropagate
        q_target = b_r + self.gamma * q_next.max(1)[0].view(self.batch_size, 1)   # shape (batch, 1)
        loss = self.loss_func(q_eval, q_target)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def load_model(self, episode):
        model_eval_path = self.path + "/eval_" + str(episode) + ".pth"
        model_target_path = self.path + "/target_" + str(episode) + ".pth"

        eval = torch.load(model_eval_path)
        target = torch.load(model_target_path)
        self.eval_net.load_state_dict(eval)
        self.target_net.load_state_dict(target)

    def save_model(self, episode):
        model_eval_path = self.path + "/eval_" + str(episode) + ".pth"
        torch.save(self.eval_net.state_dict(), model_eval_path)

        model_target_path = self.path + "/target_" + str(episode) + ".pth"
        torch.save(self.target_net.state_dict(), model_target_path)

# network input
def get_observations(state, agents_index, height, width):
    observations = np.zeros((len(agents_index), 2 + width * height))
    areas = np.zeros((height, width))
    for i in range(1, 8):
        for j in range(len(state[i])):
            if i == 1:
                areas[tuple(state[i][j])] = BEAN
            elif i < 5:
                if j == 0:
                    areas[tuple(state[i][j])] = TEAM_HEAD
                else:
                    areas[tuple(state[i][j])] = TEAM_BODY
            else:
                if j == 0:
                    areas[tuple(state[i][j])] = OPPONENT_HEAD
                else:
                    areas[tuple(state[i][j])] = OPPONENT_BODY
    for i in agents_index:
        observations[i][:2] = state[i+2][0][:]
        observations[i][2:] = areas.flatten()
    return observations

# count reward
def get_reward(info, snake_index, reward, score):
    snakes_position = np.array(info['snakes_position'], dtype=object)
    beans_position = np.array(info['beans_position'], dtype=object)
    snake_heads = [snake[0] for snake in snakes_position]
    step_reward = np.zeros(len(snake_index))
    for i in snake_index:
        if score == 1:
            step_reward[i] += 50
        elif score == 2:
            step_reward[i] -= 25
        elif score == 3:
            step_reward[i] += 10
        elif score == 4:
            step_reward[i] -= 5

        if reward[i] > 0:
            step_reward[i] += 20
        else:
            self_head = np.array(snake_heads[i])
            dists = [np.sqrt(np.sum(np.square(other_head - self_head))) for other_head in beans_position]
            step_reward[i] -= min(dists)
            if reward[i] < 0:
                step_reward[i] -= 10

    return step_reward

def main(args):
    env = make(args.game_name, conf=None)

    num_agents = env.n_player
    print(f'Total agent number: {num_agents}')
    ctrl_agent_index = [0, 1, 2]
    print(f'Agent control by the actor: {ctrl_agent_index}')
    ctrl_agent_num = len(ctrl_agent_index)

    width = env.board_width
    print(f'Game board width: {width}')
    height = env.board_height
    print(f'Game board height: {height}')

    act_dim = env.get_action_dim()
    print(f'action dimension: {act_dim}')
    obs_dim = 2 + width * height
    print(f'observation dimension: {obs_dim}')

    model = DQN(obs_dim, act_dim, args)

    episode = 0
    win = 0

    if args.load_model != 0:
        episode = args.load_model
        model.load_model(args.load_model)

    while episode < args.max_episodes:

        state = env.reset()

        state_to_training = state[0]
        obs = get_observations(state_to_training, ctrl_agent_index, height, width)

        episode += 1
        step = 0
        episode_reward = np.zeros(6)

        while True:
            team_actions = model.choose_action(obs)
            opponent_actions = np.random.randint(0, 4, (num_agents - ctrl_agent_num))
            actions = np.concatenate([team_actions, opponent_actions])
            
            next_state, reward, done, _, info = env.step(env.encode(actions))
            next_state_to_training = next_state[0]
            next_obs = get_observations(next_state_to_training, ctrl_agent_index, height, width)

            reward = np.array(reward)
            episode_reward += reward

            if done:
                if np.sum(episode_reward[:3]) > np.sum(episode_reward[3:]):
                    step_reward = get_reward(info, ctrl_agent_index, reward, score=1)
                elif np.sum(episode_reward[:3]) < np.sum(episode_reward[3:]):
                    step_reward = get_reward(info, ctrl_agent_index, reward, score=2)
                else:
                    step_reward = get_reward(info, ctrl_agent_index, reward, score=0)
            else:
                if np.sum(episode_reward[:3]) > np.sum(episode_reward[3:]):
                    step_reward = get_reward(info, ctrl_agent_index, reward, score=3)
                elif np.sum(episode_reward[:3]) < np.sum(episode_reward[3:]):
                    step_reward = get_reward(info, ctrl_agent_index, reward, score=4)
                else:
                    step_reward = get_reward(info, ctrl_agent_index, reward, score=0)

            done = np.array([done] * ctrl_agent_num)

            for i in range(ctrl_agent_num):
                model.store_transition(obs[i], actions[i], step_reward[i], next_obs[i])

            if model.memory_counter > model.memory_capacity:
                model.learn()                    

            if args.episode_length <= step or (True in done):
                if np.sum(episode_reward[:3]) > np.sum(episode_reward[3:]):
                    win += 1
                print('Ep: ', episode, '| Ep_r: ', episode_reward, '| eps: ', model.eps, '| acr: ', win/(episode-args.load_model))
                if episode % 5000 == 0:
                    model.save_model(episode)
                env.reset()
                break

            obs = next_obs

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--game_name', default="snakes_3v3", type=str)
    parser.add_argument('--max_episodes', default=50000, type=int)
    parser.add_argument('--episode_length', default=200, type=int)

    parser.add_argument('--gamma', default=0.95, type=float)
    parser.add_argument('--lr', default=0.0001, type=float)
    parser.add_argument('--batch_size', default=64, type=int)
    parser.add_argument('--epsilon', default=0.5, type=float)
    parser.add_argument('--epsilon_speed', default=0.99998, type=float)
    parser.add_argument('--target_replace_iter', default=100, type=int)
    parser.add_argument('--memory_capacity', default=20000, type=int)

    parser.add_argument('--load_model', default=0, type=int)

    args = parser.parse_args()
    main(args)
